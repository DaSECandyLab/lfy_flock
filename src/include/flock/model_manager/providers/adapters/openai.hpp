#pragma once

#include "flock/model_manager/providers/handlers/openai.hpp"
#include "flock/model_manager/providers/provider.hpp"

namespace flock {

class OpenAIProvider : public IProvider {
public:
    explicit OpenAIProvider(const ModelDetails& model_details);

    void AddCompletionRequest(const std::string& prompt, const int num_output_tuples, OutputType output_type, const nlohmann::json& media_data) override;
    void AddCompletionRequestTokenIds(const std::vector<int>& prompt_token_ids, const int num_output_tuples, OutputType output_type) override;
    void AddEmbeddingRequest(const std::vector<std::string>& inputs) override;
    void AddTranscriptionRequest(const nlohmann::json& audio_files) override;
    std::vector<int> TokenizePrompt(const std::string& prompt, bool add_special_tokens) override;
    std::vector<nlohmann::json> CollectCompletions(const std::string& contentType = "application/json") override;

private:
    void ResetDebugTimingState();

    // cacheblend 走的是 OpenAI-compatible 的 text completion 接口；普通文本路径仍走 model_handler_ 的 chat 接口。
    std::unique_ptr<OpenAIModelManager> text_completion_handler_;
    // 为 cacheblend 额外保留一个 /tokenize 会话，用来拿到精确的 token id 序列。
    Session tokenize_session_;
    std::string api_key_;
    std::string tokenize_url_;
    // 记录当前 batch 中普通 chat completion 的请求数量，用来避免和 token-id completion 混批。
    size_t pending_chat_completion_requests_ = 0;
    // 记录当前 batch 中 token-id completion 的请求数量；只有 cacheblend 路径会累加这里。
    size_t pending_text_completion_requests_ = 0;
    size_t debug_tokenize_call_count_ = 0;
    size_t debug_tokenize_prompt_chars_total_ = 0;
    size_t debug_tokenize_tokens_total_ = 0;
    double debug_tokenize_http_ms_total_ = 0.0;
    double debug_tokenize_parse_ms_total_ = 0.0;
    size_t debug_text_prompt_tokens_total_ = 0;
    size_t debug_text_completion_request_count_ = 0;
    size_t debug_chat_prompt_chars_total_ = 0;
};

}// namespace flock
