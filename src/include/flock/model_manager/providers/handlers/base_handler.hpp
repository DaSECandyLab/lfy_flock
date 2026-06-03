#pragma once

#include "flock/core/common.hpp"
#include "flock/metrics/manager.hpp"
#include "flock/model_manager/providers/handlers/handler.hpp"
#include "session.hpp"
#include <algorithm>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <curl/curl.h>
#include <filesystem>
#include <iostream>
#include <fstream>
#include <limits>
#include <random>
#include <thread>
#include <map>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

namespace flock {

class BaseModelProviderHandler : public IModelProviderHandler {
public:
    explicit BaseModelProviderHandler(bool throw_exception)
        : _throw_exception(throw_exception) {}
    virtual ~BaseModelProviderHandler() = default;

    void AddRequest(const nlohmann::json& json, RequestType type = RequestType::Completion) override {
        _request_batch.push_back(json);
        _request_types.push_back(type);
    }

    std::vector<nlohmann::json> CollectCompletions(const std::string& contentType = "application/json") override {
        std::vector<nlohmann::json> completions;
        if (!_request_batch.empty()) completions = ExecuteBatch(_request_batch, true, contentType, RequestType::Completion);
        _request_batch.clear();
        // 与请求队列一起清空类型队列，避免下一个 batch 复用到旧的请求类型。
        _request_types.clear();
        return completions;
    }

    std::vector<nlohmann::json> CollectEmbeddings(const std::string& contentType = "application/json") override {
        std::vector<nlohmann::json> embeddings;
        if (!_request_batch.empty()) embeddings = ExecuteBatch(_request_batch, true, contentType, RequestType::Embedding);
        _request_batch.clear();
        // embedding batch 结束后也同步清空类型队列。
        _request_types.clear();
        return embeddings;
    }


    std::vector<nlohmann::json> CollectTranscriptions(const std::string& contentType = "multipart/form-data") override {
        std::vector<nlohmann::json> transcriptions;
        if (!_request_batch.empty()) {
            std::vector<nlohmann::json> transcription_batch;
            for (size_t i = 0; i < _request_batch.size(); ++i) {
                if (_request_types[i] == RequestType::Transcription) {
                    transcription_batch.push_back(_request_batch[i]);
                }
            }

            if (!transcription_batch.empty()) {
                transcriptions = ExecuteBatch(transcription_batch, true, contentType, RequestType::Transcription);
                // Remove transcription requests from batch
                for (size_t i = _request_batch.size(); i > 0; --i) {
                    if (_request_types[i - 1] == RequestType::Transcription) {
                        _request_batch.erase(_request_batch.begin() + i - 1);
                        _request_types.erase(_request_types.begin() + i - 1);
                    }
                }
            }
        }
        return transcriptions;
    }


public:
protected:
    bool DebugLlmIoEnabled() const {
        const char* env_value = std::getenv("FLOCK_DEBUG_LLM_IO");
        if (env_value == nullptr) {
            return false;
        }
        auto value = std::string(env_value);
        std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) { return std::tolower(c); });
        return !(value.empty() || value == "0" || value == "false" || value == "off" || value == "no");
    }

    static const char* RequestTypeName(RequestType request_type) {
        switch (request_type) {
            case RequestType::Completion:
                return "completion";
            case RequestType::Embedding:
                return "embedding";
            case RequestType::Transcription:
                return "transcription";
        }
        return "unknown";
    }

    void DebugPrintLlmIo(const std::string& label, size_t index, const std::string& payload) const {
        std::cerr << "[FLOCK_DEBUG_LLM_IO] " << label << " #" << index << ":\n"
                  << payload << "\n";
    }

    std::vector<nlohmann::json> ExecuteBatch(const std::vector<nlohmann::json>& jsons, bool async = true, const std::string& contentType = "application/json", RequestType request_type = RequestType::Completion) {
#ifdef __EMSCRIPTEN__
        // WASM: Process requests sequentially using emscripten fetch
        std::vector<nlohmann::json> results(jsons.size());
        bool is_completion = (request_type == RequestType::Completion);
        auto url = is_completion ? getCompletionUrl() : getEmbedUrl();

        for (size_t i = 0; i < jsons.size(); ++i) {
            prepareSessionForRequest(url);
            setParameters(jsons[i].dump(), contentType);
            auto response = postRequest(contentType);

            if (!response.is_error && !response.text.empty() && isJson(response.text)) {
                try {
                    nlohmann::json parsed = nlohmann::json::parse(response.text);
                    checkResponse(parsed, request_type);
                    if (is_completion) {
                        results[i] = ExtractCompletionOutput(parsed);
                    } else {
                        results[i] = ExtractEmbeddingVector(parsed);
                    }
                } catch (const std::exception& e) {
                    trigger_error(std::string("JSON parse error: ") + e.what());
                }
            } else {
                trigger_error("Empty or invalid response: " + response.error_message);
            }
        }
        return results;
#else
        // Native 路径使用 curl multi-handle 并发发起请求。
        // 每个请求的状态都保存在 requests 原始下标里，后面按下标解析结果以保证保序。
        struct CurlRequestData {
            std::string response;
            CURL* easy = nullptr;
            std::string payload;
            curl_mime* mime_form = nullptr;
            std::string temp_file_path;
            bool is_temp_file = false;
            struct curl_slist* headers = nullptr;
            // 记录 easy handle 是否已加入/完成，便于异常和收尾时安全移除。
            bool added = false;
            bool completed = false;
            CURLcode curl_result = CURLE_OK;
            long http_code = 0;
            // 多 terminal 共用 max_in_flight 时，通过原子创建目录占住一个全局 slot。
            std::string shared_slot_path;
            int shared_slot_index = -1;
            int trace_slot_index = -1;
            size_t request_index = 0;
            size_t wave_id = 0;
            size_t effective_request_cap = 0;
            size_t active_terminal_count = 0;
            double started_at_s = 0.0;
            double completed_at_s = 0.0;
        };
        std::vector<CurlRequestData> requests(jsons.size());
        if (jsons.empty()) {
            // 空 batch 直接返回，避免创建无意义的 multi handle。
            return {};
        }
        const bool debug_llm_io = DebugLlmIoEnabled();
        CURLM* multi_handle = curl_multi_init();

        // 解析布尔型环境变量，供 TPCC workload 开关和 trace 开关复用。
        auto env_flag_enabled = [](const char* env_name) {
            const char* env_value = std::getenv(env_name);
            if (env_value == nullptr) {
                return false;
            }
            auto value = std::string(env_value);
            std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) { return std::tolower(c); });
            return !(value.empty() || value == "0" || value == "false" || value == "off" || value == "no");
        };
        // 读取可选的整型 trace 元数据；缺失或解析失败时统一返回 -1。
        auto parse_optional_int_env = [](const char* env_name) -> long long {
            const char* env_value = std::getenv(env_name);
            if (env_value == nullptr) {
                return -1;
            }
            try {
                return std::stoll(env_value);
            } catch (...) {
                return -1;
            }
        };
        // 统一使用系统时钟秒级时间戳，便于跨 terminal 还原全局请求时间线。
        auto system_clock_seconds = []() {
            return std::chrono::duration<double>(
                           std::chrono::system_clock::now().time_since_epoch())
                    .count();
        };

        // Determine URL based on request type
        std::string url;
        bool is_transcription = (request_type == RequestType::Transcription);
        bool is_completion = (request_type == RequestType::Completion);
        if (is_transcription) {
            url = getTranscriptionUrl();
        } else if (is_completion) {
            url = getCompletionUrl();
        } else {
            url = getEmbedUrl();
        }

        // Prepare all requests
        for (size_t i = 0; i < jsons.size(); ++i) {
            requests[i].request_index = i;
            requests[i].easy = curl_easy_init();
            // 通过 CURLOPT_PRIVATE 在完成事件中找回原始 request 对象，避免按完成顺序打乱结果。
            curl_easy_setopt(requests[i].easy, CURLOPT_PRIVATE, &requests[i]);
            curl_easy_setopt(requests[i].easy, CURLOPT_URL, url.c_str());

            if (is_transcription) {
                // Handle transcription requests (multipart/form-data)
                const auto& req = jsons[i];
                if (!req.contains("file_path") || req["file_path"].is_null()) {
                    trigger_error("Missing or null file_path in transcription request");
                }
                if (!req.contains("model") || req["model"].is_null()) {
                    trigger_error("Missing or null model in transcription request");
                }
                auto file_path = req["file_path"].get<std::string>();
                auto model = req["model"].get<std::string>();
                auto prompt = req.contains("prompt") && !req["prompt"].is_null() ? req["prompt"].get<std::string>() : "";
                requests[i].is_temp_file = req.contains("is_temp_file") ? req["is_temp_file"].get<bool>() : false;
                if (requests[i].is_temp_file) {
                    requests[i].temp_file_path = file_path;
                }

                // Set up multipart form data
                requests[i].mime_form = curl_mime_init(requests[i].easy);
                curl_mimepart* field = curl_mime_addpart(requests[i].mime_form);
                curl_mime_name(field, "file");
                curl_mime_filedata(field, file_path.c_str());

                field = curl_mime_addpart(requests[i].mime_form);
                curl_mime_name(field, "model");
                curl_mime_data(field, model.c_str(), CURL_ZERO_TERMINATED);

                field = curl_mime_addpart(requests[i].mime_form);
                curl_mime_name(field, "response_format");
                curl_mime_data(field, "json", CURL_ZERO_TERMINATED);

                if (!prompt.empty()) {
                    field = curl_mime_addpart(requests[i].mime_form);
                    curl_mime_name(field, "prompt");
                    curl_mime_data(field, prompt.c_str(), CURL_ZERO_TERMINATED);
                }

                curl_easy_setopt(requests[i].easy, CURLOPT_MIMEPOST, requests[i].mime_form);

                // headers 挂在 request 上，等请求完成后统一释放，避免 curl 仍使用时被提前释放。
                requests[i].headers = curl_slist_append(requests[i].headers, "Expect:");
                for (const auto& h: getExtraHeaders()) {
                    requests[i].headers = curl_slist_append(requests[i].headers, h.c_str());
                }
                curl_easy_setopt(requests[i].easy, CURLOPT_HTTPHEADER, requests[i].headers);
            } else {
                // JSON 请求先只准备 payload 和 handle；真正发起由下面的调度循环控制。
                requests[i].payload = jsons[i].dump();
                if (debug_llm_io) {
                    DebugPrintLlmIo(std::string("request ") + RequestTypeName(request_type), i, jsons[i].dump(2));
                }
                requests[i].headers = curl_slist_append(requests[i].headers, "Content-Type: application/json");
                for (const auto& h: getExtraHeaders()) {
                    requests[i].headers = curl_slist_append(requests[i].headers, h.c_str());
                }
                curl_easy_setopt(requests[i].easy, CURLOPT_HTTPHEADER, requests[i].headers);
                curl_easy_setopt(requests[i].easy, CURLOPT_POST, 1L);
                curl_easy_setopt(requests[i].easy, CURLOPT_POSTFIELDS, requests[i].payload.c_str());
            }

            // Set response callback
            curl_easy_setopt(
                    requests[i].easy, CURLOPT_WRITEFUNCTION, +[](char* ptr, size_t size, size_t nmemb, void* userdata) -> size_t {
                std::string* resp = static_cast<std::string*>(userdata);
                resp->append(ptr, size * nmemb);
                return size * nmemb; });
            curl_easy_setopt(requests[i].easy, CURLOPT_WRITEDATA, &requests[i].response);
        }

        // request_rate 只控制“发起下一个请求”的节奏，不等待前一个请求完成。
        auto parse_request_rate = []() {
            const char* env_value = std::getenv("FLOCK_VLLM_REQUEST_RATE");
            if (env_value == nullptr) {
                return std::numeric_limits<double>::infinity();
            }

            auto value = std::string(env_value);
            std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) { return std::tolower(c); });
            if (value.empty() || value == "inf" || value == "infinity" || value == "unlimited") {
                return std::numeric_limits<double>::infinity();
            }

            try {
                const auto parsed = std::stod(value);
                return parsed > 0 ? parsed : std::numeric_limits<double>::infinity();
            } catch (...) {
                return std::numeric_limits<double>::infinity();
            }
        };

        // 限制同时在途的请求数；0/inf/unlimited 表示不额外限制，最多等于本 batch 请求数。
        // Flock 这一侧只负责尽快投递请求，真正如何排队/批处理交给 vLLM 服务端调度。
        auto parse_requested_max_in_flight = []() {
            size_t max_in_flight = 0;
            if (const char* env_value = std::getenv("FLOCK_VLLM_MAX_IN_FLIGHT")) {
                try {
                    auto value = std::string(env_value);
                    std::transform(value.begin(), value.end(), value.begin(),
                                   [](unsigned char c) { return std::tolower(c); });
                    if (value.empty() || value == "0" || value == "inf" ||
                        value == "infinity" || value == "unlimited") {
                        max_in_flight = 0;
                    } else {
                        max_in_flight = std::stoul(value);
                    }
                } catch (...) {
                    max_in_flight = 0;
                }
            }
            return max_in_flight;
        };

        auto parse_positive_size_env = [](const char* env_name, size_t default_value) {
            const char* env_value = std::getenv(env_name);
            if (env_value == nullptr) {
                return default_value;
            }
            try {
                const auto parsed = std::stoll(env_value);
                return parsed > 0 ? static_cast<size_t>(parsed) : default_value;
            } catch (...) {
                return default_value;
            }
        };

        auto count_active_terminals_from_coord_dir = [](const std::string& coord_dir) {
            if (coord_dir.empty()) {
                return static_cast<size_t>(0);
            }

            size_t active_terminal_count = 0;
            std::error_code path_error;
            if (!std::filesystem::exists(coord_dir, path_error)) {
                return static_cast<size_t>(0);
            }

            std::filesystem::directory_iterator iterator(
                    coord_dir, std::filesystem::directory_options::skip_permission_denied, path_error);
            for (const auto& entry: iterator) {
                if (path_error) {
                    break;
                }
                if (!entry.is_regular_file()) {
                    continue;
                }
                std::ifstream state_file(entry.path());
                if (!state_file.good()) {
                    continue;
                }
                try {
                    nlohmann::json state_json;
                    state_file >> state_json;
                    if (!state_json.contains("remaining_query_count")) {
                        continue;
                    }
                    long long remaining_query_count = 0;
                    if (state_json["remaining_query_count"].is_number_integer()) {
                        remaining_query_count = state_json["remaining_query_count"].get<long long>();
                    } else if (state_json["remaining_query_count"].is_number_unsigned()) {
                        remaining_query_count =
                                static_cast<long long>(state_json["remaining_query_count"].get<unsigned long long>());
                    } else {
                        continue;
                    }
                    if (remaining_query_count > 0) {
                        active_terminal_count++;
                    }
                } catch (...) {
                    continue;
                }
            }

            return active_terminal_count;
        };

        // 默认使用泊松到达；设置 deterministic 时改为固定间隔，便于复现实验。
        auto deterministic_arrival = []() {
            const char* env_value = std::getenv("FLOCK_VLLM_REQUEST_ARRIVAL");
            if (env_value == nullptr) {
                return false;
            }
            auto value = std::string(env_value);
            std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) { return std::tolower(c); });
            return value == "deterministic";
        }();

        // 在 legacy query-concurrency 路径下支持 query 起始错峰；TPCC workload 也复用这套机制。
        auto maybe_apply_query_request_stagger = []() {
            if (const char* applied_env = std::getenv("FLOCKMTL_QUERY_REQUEST_STAGGER_APPLIED")) {
                if (std::string(applied_env) == "1") {
                    return;
                }
            }

            const char* stagger_env = std::getenv("FLOCKMTL_QUERY_START_STAGGER_MS");
            const char* coord_dir_env = std::getenv("FLOCKMTL_QUERY_STAGGER_COORD_DIR");
            const char* order_env = std::getenv("FLOCKMTL_QUERY_ORDER_INDEX");
            if (stagger_env == nullptr || coord_dir_env == nullptr || order_env == nullptr) {
                return;
            }

            double stagger_ms = 0.0;
            int query_order_index = 0;
            try {
                stagger_ms = std::max(0.0, std::stod(stagger_env));
                query_order_index = std::max(0, std::stoi(order_env));
            } catch (...) {
                return;
            }
            if (stagger_ms <= 0.0) {
                return;
            }

            auto file_exists = [](const std::string& path) {
                std::ifstream file(path);
                return file.good();
            };

            auto read_marker_ms = [](const std::string& path, int64_t& timestamp_ms) {
                std::ifstream file(path);
                if (!file.good()) {
                    return false;
                }
                file >> timestamp_ms;
                return !file.fail();
            };

            auto write_marker_ms = [](const std::string& path, int64_t timestamp_ms) {
                std::ofstream file(path, std::ios::trunc);
                file << timestamp_ms << '\n';
            };

            const auto coord_dir = std::string(coord_dir_env);
            const auto marker_prefix = coord_dir + "/query_" + std::to_string(query_order_index);
            const auto started_marker = marker_prefix + ".started";

            if (file_exists(started_marker)) {
                setenv("FLOCKMTL_QUERY_REQUEST_STAGGER_APPLIED", "1", 1);
                return;
            }

            if (query_order_index > 0) {
                const auto predecessor_prefix =
                        coord_dir + "/query_" + std::to_string(query_order_index - 1);
                const auto predecessor_started = predecessor_prefix + ".started";
                const auto predecessor_done = predecessor_prefix + ".done";
                const auto base_poll_ms = std::max<int64_t>(
                        1,
                        std::min<int64_t>(
                                10,
                                static_cast<int64_t>(std::ceil(stagger_ms / 10.0))));

                while (true) {
                    int64_t predecessor_started_ms = 0;
                    if (read_marker_ms(predecessor_started, predecessor_started_ms)) {
                        const auto now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                                std::chrono::system_clock::now().time_since_epoch())
                                                    .count();
                        const auto allowed_ms =
                                predecessor_started_ms + static_cast<int64_t>(std::ceil(stagger_ms));
                        if (now_ms >= allowed_ms) {
                            break;
                        }
                        const auto remaining_ms = std::max<int64_t>(1, allowed_ms - now_ms);
                        std::this_thread::sleep_for(std::chrono::milliseconds(
                                std::min<int64_t>(remaining_ms, base_poll_ms)));
                        continue;
                    }

                    if (file_exists(predecessor_done)) {
                        break;
                    }

                    std::this_thread::sleep_for(std::chrono::milliseconds(base_poll_ms));
                }
            }

            const auto now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                                        std::chrono::system_clock::now().time_since_epoch())
                                        .count();
            write_marker_ms(started_marker, now_ms);
            setenv("FLOCKMTL_QUERY_REQUEST_STAGGER_APPLIED", "1", 1);
        };

        const auto request_rate = parse_request_rate();
        const auto requested_max_in_flight = parse_requested_max_in_flight();
        const auto initial_requested_cap =
                requested_max_in_flight == 0
                        ? jsons.size()
                        : std::max<size_t>(1, std::min(requested_max_in_flight, jsons.size()));
        const auto global_max_in_flight =
                requested_max_in_flight == 0 ? jsons.size() : std::max<size_t>(1, requested_max_in_flight);
        const auto global_in_flight_coord_dir = []() -> std::string {
            if (const char* env_value = std::getenv("FLOCK_VLLM_GLOBAL_IN_FLIGHT_COORD_DIR")) {
                return std::string(env_value);
            }
            return "";
        }();
        // 多 terminal 共用全局上限时，需要跨进程协调用掉多少个 in-flight slot。
        const bool use_global_in_flight_coord =
                !global_in_flight_coord_dir.empty() && requested_max_in_flight > 0;
        // TPCC batch 模式只对 batch-capable query 生效，用于开启 query-local trace
        // 和“带数量上限的滚动补发”发送逻辑。
        const bool use_tpcc_batch_mode =
                env_flag_enabled("FLOCK_TPCC_ENABLE_WAVE_MODE") && !is_transcription;
        const auto active_terminal_coord_dir = []() -> std::string {
            if (const char* env_value = std::getenv("FLOCK_TPCC_ACTIVE_TERMINAL_COORD_DIR")) {
                return std::string(env_value);
            }
            return "";
        }();
        const auto total_terminal_count =
                parse_positive_size_env("FLOCK_TPCC_TOTAL_TERMINAL_COUNT", static_cast<size_t>(1));
        const auto query_start_active_terminal_count =
                parse_positive_size_env("FLOCK_TPCC_ACTIVE_TERMINAL_COUNT", static_cast<size_t>(0));
        const auto tpcc_vllm_batch_size =
                parse_positive_size_env("FLOCK_TPCC_VLLM_BATCH_SIZE", static_cast<size_t>(0));
        const auto active_terminal_refresh_interval =
                parse_positive_size_env("FLOCK_TPCC_ACTIVE_TERMINAL_REFRESH_INTERVAL", static_cast<size_t>(10));
        const bool use_dynamic_active_terminal_cap =
                use_tpcc_batch_mode && !use_global_in_flight_coord &&
                !active_terminal_coord_dir.empty() && tpcc_vllm_batch_size > 0;
        auto last_observed_active_terminal_count =
                use_dynamic_active_terminal_cap ? count_active_terminals_from_coord_dir(active_terminal_coord_dir)
                                                : query_start_active_terminal_count;
        if (use_dynamic_active_terminal_cap && last_observed_active_terminal_count == 0) {
            last_observed_active_terminal_count = std::max<size_t>(1, total_terminal_count);
        }
        auto min_observed_active_terminal_count =
                use_dynamic_active_terminal_cap ? last_observed_active_terminal_count : static_cast<size_t>(0);
        auto max_observed_active_terminal_count =
                use_dynamic_active_terminal_cap ? last_observed_active_terminal_count : static_cast<size_t>(0);
        auto compute_query_local_cap_from_active_terminals = [&]() -> size_t {
            if (!use_dynamic_active_terminal_cap) {
                return initial_requested_cap;
            }
            const auto observed_active_terminal_count = count_active_terminals_from_coord_dir(active_terminal_coord_dir);
            last_observed_active_terminal_count =
                    observed_active_terminal_count > 0 ? observed_active_terminal_count : std::max<size_t>(1, total_terminal_count);
            min_observed_active_terminal_count =
                    min_observed_active_terminal_count == 0
                            ? last_observed_active_terminal_count
                            : std::min(min_observed_active_terminal_count, last_observed_active_terminal_count);
            max_observed_active_terminal_count = std::max(
                    max_observed_active_terminal_count, last_observed_active_terminal_count);
            const auto dynamic_cap =
                    std::max<size_t>(1, tpcc_vllm_batch_size / std::max<size_t>(1, last_observed_active_terminal_count));
            return std::max<size_t>(1, std::min(dynamic_cap, jsons.size()));
        };
        auto max_in_flight = compute_query_local_cap_from_active_terminals();
        const auto configured_request_cap_at_query_start = max_in_flight;
        const auto active_terminal_count_at_query_start = last_observed_active_terminal_count;
        auto min_observed_request_cap = max_in_flight;
        auto max_observed_request_cap = max_in_flight;
        size_t dynamic_request_cap_refresh_count = 0;
        size_t dispatched_request_count = 0;
        size_t next_active_terminal_refresh_dispatch_count =
                use_dynamic_active_terminal_cap ? active_terminal_refresh_interval : std::numeric_limits<size_t>::max();
        size_t trace_wave_id = 0;
        size_t trace_wave_remaining_budget = std::max<size_t>(1, max_in_flight);
        auto refresh_dynamic_request_cap_if_needed = [&]() -> bool {
            if (!use_dynamic_active_terminal_cap) {
                return false;
            }
            if (dispatched_request_count < next_active_terminal_refresh_dispatch_count) {
                return false;
            }
            const auto previous_request_cap = max_in_flight;
            max_in_flight = compute_query_local_cap_from_active_terminals();
            min_observed_request_cap = std::min(min_observed_request_cap, max_in_flight);
            max_observed_request_cap = std::max(max_observed_request_cap, max_in_flight);
            dynamic_request_cap_refresh_count++;
            next_active_terminal_refresh_dispatch_count =
                    dispatched_request_count + active_terminal_refresh_interval;
            return max_in_flight != previous_request_cap;
        };
        const auto query_trace_output = []() -> std::string {
            if (const char* env_value = std::getenv("FLOCK_TPCC_QUERY_TRACE_OUTPUT")) {
                return std::string(env_value);
            }
            return "";
        }();
        const bool trace_enabled = use_tpcc_batch_mode && !query_trace_output.empty();
        const auto trace_terminal_id = parse_optional_int_env("FLOCK_TPCC_TERMINAL_ID");
        const auto trace_query_id = parse_optional_int_env("FLOCK_TPCC_QUERY_ID");
        const auto trace_step_index = parse_optional_int_env("FLOCK_TPCC_STEP_INDEX");
        const auto trace_function_type = []() -> std::string {
            if (const char* env_value = std::getenv("FLOCK_TPCC_FUNCTION_TYPE")) {
                return std::string(env_value);
            }
            return "";
        }();

        const auto paced_requests = std::isfinite(request_rate);
        std::mt19937 rng(std::random_device{}());
        std::exponential_distribution<double> poisson_interval(paced_requests ? request_rate : 1.0);

        using Clock = std::chrono::steady_clock;
        auto api_start = Clock::now();
        auto next_request_time = api_start;
        size_t completed_requests = 0;
        size_t in_flight = 0;
        int still_running = 0;

        auto interval_duration = [](double seconds) {
            return std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(seconds));
        };

        // 计算下一次允许发起请求的时间点；无限速时无需更新。
        auto schedule_next_arrival = [&]() {
            if (!paced_requests) {
                return;
            }
            const auto interval_seconds = deterministic_arrival ? (1.0 / request_rate) : poisson_interval(rng);
            next_request_time = Clock::now() + interval_duration(interval_seconds);
        };

        // 请求完成或异常退出时释放已占用的全局 slot。
        auto release_global_in_flight_slot = [](CurlRequestData& request) {
            if (!request.shared_slot_path.empty()) {
                std::error_code remove_error;
                std::filesystem::remove_all(request.shared_slot_path, remove_error);
                request.shared_slot_path.clear();
                request.shared_slot_index = -1;
            }
        };

        // 尝试为当前请求占用一个全局 slot；占不到则稍后重试。
        auto try_acquire_global_in_flight_slot = [&](CurlRequestData& request) {
            if (!use_global_in_flight_coord) {
                return true;
            }

            for (size_t slot_idx = 0; slot_idx < global_max_in_flight; ++slot_idx) {
                const auto slot_path = global_in_flight_coord_dir + "/slot_" + std::to_string(slot_idx) + ".lock";
                std::error_code create_error;
                if (std::filesystem::create_directory(slot_path, create_error)) {
                    request.shared_slot_path = slot_path;
                    request.shared_slot_index = static_cast<int>(slot_idx);
                    request.trace_slot_index = static_cast<int>(slot_idx);
                    return true;
                }
                if (create_error && create_error != std::make_error_code(std::errc::file_exists)) {
                    trigger_error(
                            "Failed to claim global in-flight coordination slot: " + slot_path +
                            " (" + create_error.message() + ")");
                }
            }
            return false;
        };

        // 统一做收尾，确保异常路径也不会泄漏 easy handle、mime form 或全局 slot。
        auto cleanup_requests = [&]() {
            for (auto& request: requests) {
                if (is_transcription && request.is_temp_file && !request.temp_file_path.empty()) {
                    std::remove(request.temp_file_path.c_str());
                    request.temp_file_path.clear();
                }
                if (request.added && !request.completed && multi_handle != nullptr && request.easy != nullptr) {
                    curl_multi_remove_handle(multi_handle, request.easy);
                }
                release_global_in_flight_slot(request);
                if (is_transcription && request.mime_form != nullptr) {
                    curl_mime_free(request.mime_form);
                    request.mime_form = nullptr;
                }
                if (request.headers != nullptr) {
                    curl_slist_free_all(request.headers);
                    request.headers = nullptr;
                }
                if (request.easy != nullptr) {
                    curl_easy_cleanup(request.easy);
                    request.easy = nullptr;
                }
            }
            if (multi_handle != nullptr) {
                curl_multi_cleanup(multi_handle);
                multi_handle = nullptr;
            }
        };

        // 输出 TPCC query trace。
        // trace 中会保留逻辑 wave 分桶，但 `request_scheduling_mode` 明确标注
        // 当前调度是 rolling refill，而不是严格的整 wave 屏障。
        auto write_query_trace = [&](const std::string& status, const std::string& error_message) {
            if (!trace_enabled) {
                return;
            }
            try {
                nlohmann::json trace;
                trace["status"] = status;
                trace["error"] = error_message.empty() ? nlohmann::json(nullptr) : nlohmann::json(error_message);
                trace["terminal_id"] = trace_terminal_id >= 0 ? nlohmann::json(trace_terminal_id) : nlohmann::json(nullptr);
                trace["query_id"] = trace_query_id >= 0 ? nlohmann::json(trace_query_id) : nlohmann::json(nullptr);
                trace["step_index"] = trace_step_index >= 0 ? nlohmann::json(trace_step_index) : nlohmann::json(nullptr);
                trace["function_type"] = trace_function_type.empty() ? nlohmann::json(nullptr) : nlohmann::json(trace_function_type);
                trace["request_type"] = RequestTypeName(request_type);
                trace["request_scheduling_mode"] =
                        use_tpcc_batch_mode ? nlohmann::json("rolling_refill_with_cap") : nlohmann::json("default");
                trace["configured_request_cap"] = configured_request_cap_at_query_start;
                trace["configured_wave_size"] = configured_request_cap_at_query_start;
                trace["requested_max_in_flight"] =
                        requested_max_in_flight == 0 ? nlohmann::json(nullptr) : nlohmann::json(requested_max_in_flight);
                trace["uses_global_in_flight_coord"] = use_global_in_flight_coord;
                trace["uses_dynamic_active_terminal_cap"] = use_dynamic_active_terminal_cap;
                trace["active_terminal_refresh_interval"] =
                        use_dynamic_active_terminal_cap ? nlohmann::json(active_terminal_refresh_interval)
                                                        : nlohmann::json(nullptr);
                trace["dynamic_request_cap_refresh_count"] =
                        use_dynamic_active_terminal_cap ? nlohmann::json(dynamic_request_cap_refresh_count)
                                                        : nlohmann::json(0);
                trace["active_terminal_count_at_query_start"] =
                        active_terminal_count_at_query_start > 0 ? nlohmann::json(active_terminal_count_at_query_start)
                                                                 : nlohmann::json(nullptr);
                trace["last_observed_active_terminal_count"] =
                        last_observed_active_terminal_count > 0 ? nlohmann::json(last_observed_active_terminal_count)
                                                                : nlohmann::json(nullptr);
                trace["min_observed_active_terminal_count"] =
                        min_observed_active_terminal_count > 0 ? nlohmann::json(min_observed_active_terminal_count)
                                                               : nlohmann::json(nullptr);
                trace["max_observed_active_terminal_count"] =
                        max_observed_active_terminal_count > 0 ? nlohmann::json(max_observed_active_terminal_count)
                                                               : nlohmann::json(nullptr);
                trace["final_effective_request_cap"] = max_in_flight;
                trace["min_observed_request_cap"] = min_observed_request_cap;
                trace["max_observed_request_cap"] = max_observed_request_cap;
                trace["total_llm_requests"] = jsons.size();

                nlohmann::json waves = nlohmann::json::array();
                if (use_tpcc_batch_mode) {
                    // wave_id 仅用于 trace 分桶；TPCC batch-capable 路径实际采用 rolling refill，
                    // 因此不同 wave_id 的请求时间区间允许重叠，不存在整 wave 同步屏障。
                    std::map<size_t, std::vector<size_t>> wave_to_request_indices;
                    for (size_t request_idx = 0; request_idx < requests.size(); ++request_idx) {
                        wave_to_request_indices[requests[request_idx].wave_id].push_back(request_idx);
                    }
                    trace["wave_count"] = wave_to_request_indices.size();
                    for (const auto& wave_entry: wave_to_request_indices) {
                        const auto wave_id = wave_entry.first;
                        const auto& wave_request_indices = wave_entry.second;
                        nlohmann::json request_jsons = nlohmann::json::array();
                        double wave_started_at = 0.0;
                        bool has_wave_start = false;
                        double wave_completed_at = 0.0;
                        bool all_requests_completed = true;

                        for (const auto request_idx: wave_request_indices) {
                            const auto& request = requests[request_idx];
                            if (request.started_at_s > 0.0) {
                                wave_started_at = has_wave_start ? std::min(wave_started_at, request.started_at_s) : request.started_at_s;
                                has_wave_start = true;
                            }
                            if (request.completed_at_s > 0.0) {
                                wave_completed_at = std::max(wave_completed_at, request.completed_at_s);
                            } else {
                                all_requests_completed = false;
                            }

                            nlohmann::json request_json;
                            request_json["terminal_id"] =
                                    trace_terminal_id >= 0 ? nlohmann::json(trace_terminal_id) : nlohmann::json(nullptr);
                            request_json["query_id"] =
                                    trace_query_id >= 0 ? nlohmann::json(trace_query_id) : nlohmann::json(nullptr);
                            request_json["step_index"] =
                                    trace_step_index >= 0 ? nlohmann::json(trace_step_index) : nlohmann::json(nullptr);
                            request_json["wave_id"] = request.wave_id;
                            request_json["request_index"] = request.request_index;
                            request_json["row_id"] = request.request_index;
                            request_json["function_type"] =
                                    trace_function_type.empty() ? nlohmann::json(nullptr) : nlohmann::json(trace_function_type);
                            request_json["request_type"] = RequestTypeName(request_type);
                            request_json["started_at"] =
                                    request.started_at_s > 0.0 ? nlohmann::json(request.started_at_s) : nlohmann::json(nullptr);
                            request_json["completed_at"] =
                                    request.completed_at_s > 0.0 ? nlohmann::json(request.completed_at_s) : nlohmann::json(nullptr);
                            request_json["duration_ms"] =
                                    (request.started_at_s > 0.0 && request.completed_at_s > 0.0)
                                            ? nlohmann::json((request.completed_at_s - request.started_at_s) * 1000.0)
                                            : nlohmann::json(nullptr);
                            request_json["http_code"] =
                                    request.http_code > 0 ? nlohmann::json(request.http_code) : nlohmann::json(nullptr);
                            request_json["shared_slot_index"] =
                                    request.trace_slot_index >= 0 ? nlohmann::json(request.trace_slot_index)
                                                                  : nlohmann::json(nullptr);
                            request_json["effective_request_cap"] =
                                    request.effective_request_cap > 0 ? nlohmann::json(request.effective_request_cap)
                                                                      : nlohmann::json(nullptr);
                            request_json["active_terminal_count_at_dispatch"] =
                                    request.active_terminal_count > 0 ? nlohmann::json(request.active_terminal_count)
                                                                      : nlohmann::json(nullptr);
                            request_jsons.push_back(request_json);
                        }

                        nlohmann::json wave_json;
                        wave_json["wave_id"] = wave_id;
                        wave_json["request_count"] = wave_request_indices.size();
                        wave_json["is_barrier_wave"] = false;
                        wave_json["started_at"] = has_wave_start ? nlohmann::json(wave_started_at) : nlohmann::json(nullptr);
                        wave_json["completed_at"] =
                                (all_requests_completed && wave_completed_at > 0.0)
                                        ? nlohmann::json(wave_completed_at)
                                        : nlohmann::json(nullptr);
                        wave_json["duration_ms"] =
                                (has_wave_start && all_requests_completed && wave_completed_at > 0.0)
                                        ? nlohmann::json((wave_completed_at - wave_started_at) * 1000.0)
                                        : nlohmann::json(nullptr);
                        wave_json["requests"] = request_jsons;
                        waves.push_back(wave_json);
                    }
                } else {
                    trace["wave_count"] = 0;
                }

                trace["waves"] = waves;
                auto trace_path = std::filesystem::path(query_trace_output);
                if (trace_path.has_parent_path()) {
                    std::filesystem::create_directories(trace_path.parent_path());
                }
                std::ofstream trace_file(trace_path, std::ios::trunc);
                trace_file << trace.dump(2);
            } catch (const std::exception& trace_exc) {
                std::cerr << "[FLOCK_TPCC_TRACE] Failed to write query trace: " << trace_exc.what() << '\n';
            }
        };

        // 回收已经完成的 easy handle，只记录完成状态；响应内容稍后仍按 requests 下标解析。
        auto drain_completed_requests = [&]() {
            int msgs_left = 0;
            while (auto* msg = curl_multi_info_read(multi_handle, &msgs_left)) {
                if (msg->msg == CURLMSG_DONE) {
                    CurlRequestData* request = nullptr;
                    curl_easy_getinfo(msg->easy_handle, CURLINFO_PRIVATE, &request);
                    if (request != nullptr && !request->completed) {
                        request->completed = true;
                        request->completed_at_s = system_clock_seconds();
                        request->curl_result = msg->data.result;
                        curl_easy_getinfo(msg->easy_handle, CURLINFO_RESPONSE_CODE, &request->http_code);
                        curl_multi_remove_handle(multi_handle, msg->easy_handle);
                        release_global_in_flight_slot(*request);
                        completed_requests++;
                    }
                    if (in_flight > 0) {
                        in_flight--;
                    }
                }
            }
        };

        // 将一个待发送请求真正放入 curl multi。
        // 这是“完成一个补一个”的核心入口：只要 slot 和节流条件满足，就立即发下一个。
        auto add_request_to_multi = [&](size_t request_idx) {
            if (request_idx == 0 && !requests[request_idx].added) {
                maybe_apply_query_request_stagger();
            }
            if (trace_wave_remaining_budget == 0) {
                trace_wave_id++;
                trace_wave_remaining_budget = std::max<size_t>(1, max_in_flight);
            }
            requests[request_idx].wave_id = trace_wave_id;
            requests[request_idx].effective_request_cap = max_in_flight;
            requests[request_idx].active_terminal_count =
                    use_dynamic_active_terminal_cap ? last_observed_active_terminal_count
                                                    : query_start_active_terminal_count;
            if (!try_acquire_global_in_flight_slot(requests[request_idx])) {
                return false;
            }
            const auto add_result = curl_multi_add_handle(multi_handle, requests[request_idx].easy);
            if (add_result != CURLM_OK) {
                release_global_in_flight_slot(requests[request_idx]);
                trigger_error(std::string("Failed to add request to curl multi handle: ") + curl_multi_strerror(add_result));
            }
            requests[request_idx].added = true;
            requests[request_idx].started_at_s = system_clock_seconds();
            in_flight++;
            if (trace_wave_remaining_budget > 0) {
                trace_wave_remaining_budget--;
            }
            schedule_next_arrival();
            return true;
        };

        // 计算下一次 `curl_multi_wait` 的超时。
        // 如果启用了 request_rate 节流，则超时会被“下一次允许发请求的时间点”截断。
        auto compute_timeout_ms = [&](size_t pending_request_limit, size_t next_request_idx) {
            long timeout_ms = 1000;
            if (paced_requests && next_request_idx < pending_request_limit && in_flight < max_in_flight) {
                const auto wait_ms =
                        std::chrono::duration_cast<std::chrono::milliseconds>(next_request_time - Clock::now()).count();
                timeout_ms = static_cast<long>(std::max<long long>(0, std::min<long long>(wait_ms, 1000)));
            }
            return timeout_ms;
        };

        // Native 路径和 TPCC batch-capable 路径都使用 rolling window：
        // 只要 in-flight 低于上限，就立即补发下一个待处理请求。
        auto run_rolling_window = [&]() {
            size_t next_request_idx = 0;
            while (completed_requests < jsons.size()) {
                auto now = Clock::now();
                bool blocked_on_global_slot = false;
                while (next_request_idx < jsons.size() && in_flight < max_in_flight && (!paced_requests || now >= next_request_time)) {
                    if (!add_request_to_multi(next_request_idx)) {
                        blocked_on_global_slot = true;
                        break;
                    }
                    next_request_idx++;
                    dispatched_request_count++;
                    if (refresh_dynamic_request_cap_if_needed()) {
                        trace_wave_remaining_budget = 0;
                    }
                    now = Clock::now();
                }

                curl_multi_perform(multi_handle, &still_running);
                drain_completed_requests();

                if (completed_requests >= jsons.size()) {
                    break;
                }

                const auto timeout_ms = compute_timeout_ms(jsons.size(), next_request_idx);
                if (in_flight > 0) {
                    // 有在途请求时等待网络事件；timeout 会被下一次 paced arrival 截断。
                    int numfds = 0;
                    curl_multi_wait(multi_handle, NULL, 0, timeout_ms, &numfds);
                } else if (paced_requests && next_request_idx < jsons.size()) {
                    // 没有在途请求但还没到下一次到达时间时，直接 sleep 到发起时间。
                    std::this_thread::sleep_until(next_request_time);
                } else if (blocked_on_global_slot) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
                }
            }
        };

        try {
            run_rolling_window();

            auto api_end = Clock::now();
            double api_duration_ms = std::chrono::duration<double, std::milli>(api_end - api_start).count();

            int64_t batch_input_tokens = 0;
            int64_t batch_output_tokens = 0;

            // results 与 jsons/requests 等长并共享下标，因此即使完成顺序不同，返回顺序仍等于请求入队顺序。
            std::vector<nlohmann::json> results(jsons.size());
            for (size_t i = 0; i < requests.size(); ++i) {
                const auto http_code = requests[i].http_code;

                if (requests[i].curl_result != CURLE_OK) {
                    trigger_error(std::string("Provider request failed: ") + curl_easy_strerror(requests[i].curl_result) +
                                  " (URL: " + url + ")");
                } else if (requests[i].response.empty()) {
                    trigger_error("Empty response from provider (HTTP " + std::to_string(http_code) + ", URL: " + url + ")");
                } else if (isJson(requests[i].response)) {
                    try {
                        nlohmann::json parsed = nlohmann::json::parse(requests[i].response);
                        if (debug_llm_io) {
                            DebugPrintLlmIo("raw response", i, parsed.dump(2));
                        }
                        checkResponse(parsed, request_type);

                        // Extract token usage for completions/embeddings
                        if (!is_transcription) {
                            auto [input_tokens, output_tokens] = ExtractTokenUsage(parsed);
                            batch_input_tokens += input_tokens;
                            batch_output_tokens += output_tokens;
                        }

                        // Let provider extract output based on request type
                        try {
                            results[i] = ExtractOutput(parsed, request_type);
                            if (debug_llm_io) {
                                DebugPrintLlmIo("parsed output", i, results[i].dump(2));
                            }
                        } catch (const std::exception& e) {
                            std::string msg = e.what();
                            if (msg.rfind("[ModelProvider]", 0) == 0) {
                                throw;
                            }
                            trigger_error(std::string("Output extraction error: ") + msg);
                        }
                    } catch (const std::exception& e) {
                        std::string msg = e.what();
                        if (msg.rfind("[ModelProvider]", 0) == 0) {
                            throw;
                        }
                        trigger_error(std::string("Response processing error: ") + msg);
                    }
                } else {
                    trigger_error("Invalid JSON response (HTTP " + std::to_string(http_code) + ", URL: " + url + "): " + requests[i].response);
                }
            }

            if (!is_transcription) {
                MetricsManager::UpdateTokens(batch_input_tokens, batch_output_tokens);
            }
            MetricsManager::AddApiDuration(api_duration_ms);
            for (size_t i = 0; i < jsons.size(); ++i) {
                MetricsManager::IncrementApiCalls();
            }

            write_query_trace("success", "");
            cleanup_requests();
            return results;
        } catch (const std::exception& exc) {
            write_query_trace("failed", exc.what());
            cleanup_requests();
            throw;
        } catch (...) {
            write_query_trace("failed", "unknown exception");
            cleanup_requests();
            throw;
        }
#endif
    }

    virtual void setParameters(const std::string& data, const std::string& contentType = "") = 0;
    virtual auto postRequest(const std::string& contentType) -> decltype(((Session*) nullptr)->postPrepare(contentType)) = 0;

protected:
    bool _throw_exception;
    std::vector<nlohmann::json> _request_batch;
    std::vector<RequestType> _request_types;

    virtual std::string getCompletionUrl() const = 0;
    virtual std::string getEmbedUrl() const = 0;
    virtual std::string getTranscriptionUrl() const = 0;
    virtual void prepareSessionForRequest(const std::string& url) = 0;
    virtual std::vector<std::string> getExtraHeaders() const { return {}; }
    virtual void checkProviderSpecificResponse(const nlohmann::json&, RequestType request_type) {}
    virtual nlohmann::json ExtractCompletionOutput(const nlohmann::json&) const { return {}; }
    virtual nlohmann::json ExtractEmbeddingVector(const nlohmann::json&) const { return {}; }
    virtual nlohmann::json ExtractTranscriptionOutput(const nlohmann::json&) const = 0;

    // Unified extraction method - delegates to specific Extract* methods based on request type
    nlohmann::json ExtractOutput(const nlohmann::json& parsed, RequestType request_type) const {
        if (request_type == RequestType::Completion) {
            return ExtractCompletionOutput(parsed);
        } else if (request_type == RequestType::Embedding) {
            return ExtractEmbeddingVector(parsed);
        } else {
            return ExtractTranscriptionOutput(parsed);
        }
    }
    virtual std::pair<int64_t, int64_t> ExtractTokenUsage(const nlohmann::json& response) const = 0;

    void trigger_error(const std::string& msg) {
        const std::string prefix = "[ModelProvider] ";
        std::string full_message;
        if (msg.rfind(prefix, 0) == 0) {
            full_message = msg;
        } else {
            full_message = prefix + msg;
        }

        if (_throw_exception) {
            throw std::runtime_error(full_message);
        } else {
            std::cerr << full_message << '\n';
        }
    }

    void checkResponse(const nlohmann::json& json, RequestType request_type) {
        if (json.contains("error")) {
            const auto& err = json["error"];
            std::string reason;

            if (err.is_object()) {
                if (err.contains("message") && err["message"].is_string()) {
                    reason = err["message"].get<std::string>();
                } else {
                    reason = err.dump();
                }
            } else if (err.is_string()) {
                reason = err.get<std::string>();
            } else {
                reason = err.dump();
            }

            trigger_error("Provider error: " + reason);
            std::cerr << ">> response error :\n"
                      << json.dump(2) << "\n";
        }
        checkProviderSpecificResponse(json, request_type);
    }

    bool isJson(const std::string& data) {
        try {
            (void)nlohmann::json::parse(data);
        } catch (...) {
            return false;
        }
        return true;
    }
};

}// namespace flock
