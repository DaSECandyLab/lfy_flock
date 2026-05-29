#include "flock/model_manager/providers/adapters/openai.hpp"
#include "flock/model_manager/model.hpp"
#include "flock/model_manager/providers/handlers/url_handler.hpp"
#include <cctype>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <utility>

namespace flock {

namespace {

std::string NormalizeApiBaseUrl(const std::string& api_base_url) {
    // 统一 base_url 形态，避免后续拼接 /v1/completions 或 /tokenize 时出现双斜杠、漏斜杠等问题。
    if (api_base_url.empty()) {
        return "https://api.openai.com/v1";
    }
    if (!api_base_url.empty() && api_base_url.back() == '/') {
        return api_base_url.substr(0, api_base_url.size() - 1);
    }
    return api_base_url;
}

std::string DeriveTokenizeUrl(const std::string& api_base_url) {
    // vLLM 的 OpenAI-compatible /tokenize 通常挂在根路径下，而不是 /v1/tokenize，
    // 因此这里要先去掉尾部的 /v1，再拼出真正可调用的 tokenize 地址。
    auto normalized = NormalizeApiBaseUrl(api_base_url);
    if (normalized.size() >= 3 && normalized.substr(normalized.size() - 3) == "/v1") {
        normalized.erase(normalized.size() - 3);
    }
    return normalized + "/tokenize";
}

nlohmann::json BuildItemSchema(OutputType output_type) {
    const char* bool_schema_mode = std::getenv("FLOCK_BOOL_SCHEMA_MODE");
    if (output_type == OutputType::BOOL && bool_schema_mode != nullptr &&
        std::string(bool_schema_mode) == "string_enum") {
        return {{"type", "string"}, {"enum", {"true", "false"}}};
    }
    if (output_type == OutputType::BOOL && bool_schema_mode != nullptr &&
        std::string(bool_schema_mode) == "boolean_enum") {
        return {{"type", "boolean"}, {"enum", {true, false}}};
    }
    return {{"type", IProvider::GetOutputTypeString(output_type)}};
}

nlohmann::json BuildResponseSchema(const int num_output_tuples, OutputType output_type) {
    const nlohmann::json item_schema = BuildItemSchema(output_type);
    const nlohmann::json items_schema = {
            {"type", "array"},
            {"minItems", num_output_tuples},
            {"maxItems", num_output_tuples},
            {"items", item_schema},
    };
    return {
            {"type", "object"},
            {"properties", {{"items", items_schema}}},
            {"required", {"items"}},
            {"additionalProperties", false},
    };
}

bool CacheBlendTimingDebugEnabled() {
    const char* env_value = std::getenv("FLOCK_DEBUG_CACHEBLEND_TIMING");
    if (env_value == nullptr) {
        return false;
    }
    auto value = std::string(env_value);
    for (auto& ch: value) {
        ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
    }
    return !(value.empty() || value == "0" || value == "false" || value == "off" || value == "no");
}

}// namespace

OpenAIProvider::OpenAIProvider(const ModelDetails& model_details)
    : IProvider(model_details), tokenize_session_("OpenAI", true) {
    auto base_url = std::string("");
    if (const auto it = model_details_.secret.find("base_url"); it != model_details_.secret.end()) {
        base_url = it->second;
    }
    const auto normalized_api_base_url = NormalizeApiBaseUrl(base_url);
    api_key_ = model_details_.secret.at("api_key");
    tokenize_url_ = DeriveTokenizeUrl(normalized_api_base_url);
    // 普通查询继续沿用 chat/completions 路径。
    model_handler_ = std::make_unique<OpenAIModelManager>(
            api_key_, normalized_api_base_url, true, OpenAICompletionApiMode::Chat);
    // cacheblend 需要把 prompt 以 token id 形式提交给 /v1/completions，所以额外维护一个 text completion handler。
    text_completion_handler_ = std::make_unique<OpenAIModelManager>(
            api_key_, normalized_api_base_url, true, OpenAICompletionApiMode::Text);
    tokenize_session_.setToken(api_key_, "");
}

void OpenAIProvider::ResetDebugTimingState() {
    debug_tokenize_call_count_ = 0;
    debug_tokenize_prompt_chars_total_ = 0;
    debug_tokenize_tokens_total_ = 0;
    debug_tokenize_http_ms_total_ = 0.0;
    debug_tokenize_parse_ms_total_ = 0.0;
    debug_text_prompt_tokens_total_ = 0;
    debug_text_completion_request_count_ = 0;
    debug_chat_prompt_chars_total_ = 0;
}

void OpenAIProvider::AddCompletionRequest(const std::string& prompt, const int num_output_tuples, OutputType output_type, const nlohmann::json& media_data) {
    if (pending_text_completion_requests_ > 0) {
        // 同一个 batch 里一旦混入 token-id completion，就无法再复用 chat/completions 的解析与收集逻辑。
        throw std::runtime_error("Cannot mix token-id completions with chat completions in the same OpenAIProvider batch");
    }
    auto message_content = nlohmann::json::array();

    message_content.push_back({{"type", "text"}, {"text", prompt}});

    // Process image columns
    if (media_data.contains("image") && !media_data["image"].empty() && media_data["image"].is_array()) {
        std::string detail = "low";
        auto column_index = 1u;
        for (const auto& column: media_data["image"]) {
            // Process image column as before
            if (column_index == 1) {
                detail = column.contains("detail") ? column["detail"].get<std::string>() : "low";
            }
            auto image_type = column.contains("type") ? column["type"].get<std::string>() : "image";
            auto mime_type = std::string("image/");
            if (size_t pos = image_type.find("/"); pos != std::string::npos) {
                mime_type += image_type.substr(pos + 1);
            } else {
                mime_type += std::string("png");
            }
            message_content.push_back(
                    {{"type", "text"},
                     {"text", "ATTACHMENT COLUMN"}});
            auto row_index = 1u;
            for (const auto& image: column["data"]) {
                // Skip null values
                if (image.is_null()) {
                    continue;
                }
                message_content.push_back(
                        {{"type", "text"}, {"text", "ROW " + std::to_string(row_index) + " :"}});
                auto image_url = std::string();
                std::string image_str;
                if (image.is_string()) {
                    image_str = image.get<std::string>();
                } else {
                    image_str = image.dump();
                }

                // Handle file path or URL
                if (URLHandler::IsUrl(image_str)) {
                    // URL - send directly to API
                    image_url = image_str;
                } else {
                    // File path - read and convert to base64
                    auto base64_result = URLHandler::ResolveFileToBase64(image_str);
                    image_url = duckdb_fmt::format("data:{};base64,{}", mime_type, base64_result.base64_content);
                }

                message_content.push_back(
                        {{"type", "image_url"},
                         {"image_url", {{"url", image_url}, {"detail", detail}}}});
                row_index++;
            }
            column_index++;
        }
    }


    nlohmann::json request_payload = {{"model", model_details_.model},
                                      {"messages", {{{"role", "user"}, {"content", message_content}}}}};

    if (!model_details_.model_parameters.empty()) {
        request_payload.update(model_details_.model_parameters);
    }

    if (model_details_.model_parameters.contains("response_format")) {
        auto schema = model_details_.model_parameters["response_format"]["json_schema"]["schema"];
        auto strict = model_details_.model_parameters["response_format"]["strict"];
        request_payload["response_format"] = {
                {"type", "json_schema"},
                {"json_schema",
                 {{"name", "flock_response"},
                  {"strict", strict},
                  {"schema", {{"type", "object"}, {"properties", {{"items", {{"type", "array"}, {"minItems", num_output_tuples}, {"maxItems", num_output_tuples}, {"items", schema}}}}}, {"required", {"items"}}, {"additionalProperties", false}}}}}};
    } else {
        request_payload["response_format"] = {
                {"type", "json_schema"},
                {"json_schema",
                 {{"name", "flock_response"},
                  {"strict", false},
                  {"schema", BuildResponseSchema(num_output_tuples, output_type)}}}};
    }

    model_handler_->AddRequest(request_payload);
    pending_chat_completion_requests_++;
    debug_chat_prompt_chars_total_ += prompt.size();
}

void OpenAIProvider::AddCompletionRequestTokenIds(const std::vector<int>& prompt_token_ids, const int num_output_tuples, OutputType output_type) {
    if (pending_chat_completion_requests_ > 0) {
        // cacheblend 的 token-id 请求必须单独成批发送，避免和普通 chat completion 走到不同 handler 时状态错乱。
        throw std::runtime_error("Cannot mix chat completions with token-id completions in the same OpenAIProvider batch");
    }
    // 这里显式使用 prompt token ids，避免服务端再次重新分词，保证上游分段结果能原样传到 vLLM/LMCache。
    nlohmann::json request_payload = {
            {"model", model_details_.model},
            {"prompt", prompt_token_ids},
    };

    if (!model_details_.model_parameters.empty()) {
        request_payload.update(model_details_.model_parameters);
    }

    if (model_details_.model_parameters.contains("response_format")) {
        auto schema = model_details_.model_parameters["response_format"]["json_schema"]["schema"];
        auto strict = model_details_.model_parameters["response_format"]["strict"];
        request_payload["response_format"] = {
                {"type", "json_schema"},
                {"json_schema",
                 {{"name", "flock_response"},
                  {"strict", strict},
                  {"schema", {{"type", "object"}, {"properties", {{"items", {{"type", "array"}, {"minItems", num_output_tuples}, {"maxItems", num_output_tuples}, {"items", schema}}}}}, {"required", {"items"}}, {"additionalProperties", false}}}}}};
    } else {
        request_payload["response_format"] = {
                {"type", "json_schema"},
                {"json_schema",
                 {{"name", "flock_response"},
                  {"strict", false},
                  {"schema", BuildResponseSchema(num_output_tuples, output_type)}}}};
    }

    text_completion_handler_->AddRequest(request_payload);
    pending_text_completion_requests_++;
    debug_text_prompt_tokens_total_ += prompt_token_ids.size();
    debug_text_completion_request_count_++;
}

void OpenAIProvider::AddEmbeddingRequest(const std::vector<std::string>& inputs) {
    nlohmann::json request_payload = {
            {"model", model_details_.model},
            {"input", inputs},
    };

    model_handler_->AddRequest(request_payload, IModelProviderHandler::RequestType::Embedding);
}

void OpenAIProvider::AddTranscriptionRequest(const nlohmann::json& audio_files) {
    for (const auto& audio_file: audio_files) {
        // Skip null values
        if (audio_file.is_null()) {
            continue;
        }
        std::string audio_file_str;
        if (audio_file.is_string()) {
            audio_file_str = audio_file.get<std::string>();
        } else {
            audio_file_str = audio_file.dump();
        }

        // Handle file download and validation
        auto file_result = URLHandler::ResolveFilePath(audio_file_str);

        nlohmann::json transcription_request = {
                {"file_path", file_result.file_path},
                {"model", model_details_.model},
                {"is_temp_file", file_result.is_temp_file}};
        model_handler_->AddRequest(transcription_request, IModelProviderHandler::RequestType::Transcription);
    }
}

std::vector<int> OpenAIProvider::TokenizePrompt(const std::string& prompt, bool add_special_tokens) {
    // cacheblend 先调用 /tokenize，把每个 prompt 片段映射成 token id。
    // 这样 flock 可以自行拼出完整 token 序列，而不是把整段文本重新交给服务端分词。
    tokenize_session_.setUrl(tokenize_url_);
    tokenize_session_.setBody(
            nlohmann::json{
                    {"model", model_details_.model},
                    {"prompt", prompt},
                    {"add_special_tokens", add_special_tokens},
            }.dump());
    const auto http_start = std::chrono::steady_clock::now();
    const auto response = tokenize_session_.postPrepare("application/json");
    const auto http_end = std::chrono::steady_clock::now();
    if (response.is_error) {
        throw std::runtime_error("OpenAI-compatible /tokenize request failed: " + response.error_message);
    }

    const auto parse_start = std::chrono::steady_clock::now();
    const auto parsed = nlohmann::json::parse(response.text);
    const auto parse_end = std::chrono::steady_clock::now();
    if (!parsed.contains("tokens") || !parsed["tokens"].is_array()) {
        throw std::runtime_error("OpenAI-compatible /tokenize response does not contain a tokens array");
    }
    auto token_ids = parsed["tokens"].get<std::vector<int>>();
    debug_tokenize_call_count_++;
    debug_tokenize_prompt_chars_total_ += prompt.size();
    debug_tokenize_tokens_total_ += token_ids.size();
    debug_tokenize_http_ms_total_ += std::chrono::duration<double, std::milli>(http_end - http_start).count();
    debug_tokenize_parse_ms_total_ += std::chrono::duration<double, std::milli>(parse_end - parse_start).count();
    return token_ids;
}

std::vector<nlohmann::json> OpenAIProvider::CollectCompletions(const std::string& contentType) {
    if (pending_chat_completion_requests_ > 0 && pending_text_completion_requests_ > 0) {
        throw std::runtime_error("OpenAIProvider has mixed completion batch state");
    }
    if (pending_text_completion_requests_ > 0) {
        // cacheblend 的补全结果由 text completion handler 收集。
        if (CacheBlendTimingDebugEnabled()) {
            std::cerr << duckdb_fmt::format(
                    "[FLOCK_CACHEBLEND_PROVIDER_TIMING] mode=text_completion requests={} prompt_tokens_total={} tokenize_calls={} tokenize_prompt_chars_total={} tokenize_tokens_total={} tokenize_http_ms_total={:.3f} tokenize_parse_ms_total={:.3f}\n",
                    debug_text_completion_request_count_,
                    debug_text_prompt_tokens_total_,
                    debug_tokenize_call_count_,
                    debug_tokenize_prompt_chars_total_,
                    debug_tokenize_tokens_total_,
                    debug_tokenize_http_ms_total_,
                    debug_tokenize_parse_ms_total_);
        }
        pending_text_completion_requests_ = 0;
        auto completions = text_completion_handler_->CollectCompletions(contentType);
        ResetDebugTimingState();
        return completions;
    }
    if (pending_chat_completion_requests_ > 0) {
        // 普通查询仍然完全沿用原来的 chat completion handler。
        if (CacheBlendTimingDebugEnabled()) {
            std::cerr << duckdb_fmt::format(
                    "[FLOCK_CACHEBLEND_PROVIDER_TIMING] mode=chat_completion requests={} prompt_chars_total={}\n",
                    pending_chat_completion_requests_,
                    debug_chat_prompt_chars_total_);
        }
        pending_chat_completion_requests_ = 0;
        auto completions = model_handler_->CollectCompletions(contentType);
        ResetDebugTimingState();
        return completions;
    }
    return {};
}

}// namespace flock
