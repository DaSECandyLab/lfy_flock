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
#include <iostream>
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
        };
        std::vector<CurlRequestData> requests(jsons.size());
        if (jsons.empty()) {
            // 空 batch 直接返回，避免创建无意义的 multi handle。
            return {};
        }
        const bool debug_llm_io = DebugLlmIoEnabled();
        CURLM* multi_handle = curl_multi_init();

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

        // 限制同时在途的请求数；0 表示不额外限制，最多等于本 batch 请求数。
        // 这里把默认值提升到 2048，与上层 SemBench/Flock 的默认配置保持一致，
        // 避免未显式设置环境变量时仍退回旧的 32 并发上限。
        auto parse_max_in_flight = [request_count = jsons.size()]() {
            size_t max_in_flight = 2048;
            if (const char* env_value = std::getenv("FLOCK_VLLM_MAX_IN_FLIGHT")) {
                try {
                    max_in_flight = std::stoul(env_value);
                } catch (...) {
                    max_in_flight = 2048;
                }
            }
            if (max_in_flight == 0) {
                max_in_flight = request_count;
            }
            return std::max<size_t>(1, std::min(max_in_flight, request_count));
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

        const auto request_rate = parse_request_rate();
        const auto max_in_flight = parse_max_in_flight();
        const auto paced_requests = std::isfinite(request_rate);
        std::mt19937 rng(std::random_device{}());
        std::exponential_distribution<double> poisson_interval(paced_requests ? request_rate : 1.0);

        using Clock = std::chrono::steady_clock;
        auto api_start = Clock::now();
        auto next_request_time = api_start;
        size_t next_request_idx = 0;
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

        // 回收已经完成的 easy handle，只记录完成状态；响应内容稍后仍按 requests 下标解析。
        auto drain_completed_requests = [&]() {
            int msgs_left = 0;
            while (auto* msg = curl_multi_info_read(multi_handle, &msgs_left)) {
                if (msg->msg == CURLMSG_DONE) {
                    CurlRequestData* request = nullptr;
                    curl_easy_getinfo(msg->easy_handle, CURLINFO_PRIVATE, &request);
                    if (request != nullptr && !request->completed) {
                        request->completed = true;
                        request->curl_result = msg->data.result;
                        curl_easy_getinfo(msg->easy_handle, CURLINFO_RESPONSE_CODE, &request->http_code);
                        curl_multi_remove_handle(multi_handle, msg->easy_handle);
                        completed_requests++;
                    }
                    if (in_flight > 0) {
                        in_flight--;
                    }
                }
            }
        };

        // 调度循环：满足 request_rate 和 max_in_flight 时继续发起新请求，
        // 同时不断驱动 curl multi 并回收已完成请求。
        while (completed_requests < jsons.size()) {
            auto now = Clock::now();
            while (next_request_idx < jsons.size() && in_flight < max_in_flight && (!paced_requests || now >= next_request_time)) {
                const auto add_result = curl_multi_add_handle(multi_handle, requests[next_request_idx].easy);
                if (add_result != CURLM_OK) {
                    trigger_error(std::string("Failed to add request to curl multi handle: ") + curl_multi_strerror(add_result));
                }
                requests[next_request_idx].added = true;
                next_request_idx++;
                in_flight++;
                schedule_next_arrival();
                now = Clock::now();
            }

            curl_multi_perform(multi_handle, &still_running);
            drain_completed_requests();

            if (completed_requests >= jsons.size()) {
                break;
            }

            long timeout_ms = 1000;
            if (paced_requests && next_request_idx < jsons.size() && in_flight < max_in_flight) {
                const auto wait_ms = std::chrono::duration_cast<std::chrono::milliseconds>(next_request_time - Clock::now()).count();
                timeout_ms = static_cast<long>(std::max<long long>(0, std::min<long long>(wait_ms, 1000)));
            }

            if (in_flight > 0) {
                // 有在途请求时等待网络事件；timeout 会被下一次 paced arrival 截断。
                int numfds = 0;
                curl_multi_wait(multi_handle, NULL, 0, timeout_ms, &numfds);
            } else if (paced_requests && next_request_idx < jsons.size()) {
                // 没有在途请求但还没到下一次到达时间时，直接 sleep 到发起时间。
                std::this_thread::sleep_until(next_request_time);
            }
        }

        auto api_end = Clock::now();
        double api_duration_ms = std::chrono::duration<double, std::milli>(api_end - api_start).count();

        int64_t batch_input_tokens = 0;
        int64_t batch_output_tokens = 0;

        // results 与 jsons/requests 等长并共享下标，因此即使完成顺序不同，返回顺序仍等于请求入队顺序。
        std::vector<nlohmann::json> results(jsons.size());
        for (size_t i = 0; i < requests.size(); ++i) {
            // Clean up temp files for transcriptions
            if (is_transcription && requests[i].is_temp_file && !requests[i].temp_file_path.empty()) {
                std::remove(requests[i].temp_file_path.c_str());
            }

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

            if (requests[i].added && !requests[i].completed) {
                // 异常路径下可能还没收到完成事件，清理前确保从 multi handle 移除。
                curl_multi_remove_handle(multi_handle, requests[i].easy);
            }
            // Clean up mime form for transcriptions
            if (is_transcription && requests[i].mime_form) {
                curl_mime_free(requests[i].mime_form);
            }
            if (requests[i].headers) {
                // headers 生命周期跟随 easy handle，在请求处理完后统一释放。
                curl_slist_free_all(requests[i].headers);
            }
            curl_easy_cleanup(requests[i].easy);
        }

        if (!is_transcription) {
            MetricsManager::UpdateTokens(batch_input_tokens, batch_output_tokens);
        }
        MetricsManager::AddApiDuration(api_duration_ms);
        for (size_t i = 0; i < jsons.size(); ++i) {
            MetricsManager::IncrementApiCalls();
        }

        curl_multi_cleanup(multi_handle);
        return results;
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
