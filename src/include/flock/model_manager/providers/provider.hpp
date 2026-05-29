#pragma once

#include "fmt/format.h"
#include <regex>

#include "flock/core/common.hpp"
#include "flock/model_manager/providers/handlers/handler.hpp"
#include "flock/model_manager/repository.hpp"
#include <nlohmann/json.hpp>
#include <stdexcept>

namespace flock {

bool is_base64(const std::string& str);

enum class OutputType {
    STRING,
    OBJECT,
    BOOL,
    INTEGER
};

class IProvider {
public:
    ModelDetails model_details_;
    std::unique_ptr<IModelProviderHandler> model_handler_;

    explicit IProvider(const ModelDetails& model_details) : model_details_(model_details) {};
    virtual ~IProvider() = default;

    virtual void AddCompletionRequest(const std::string& prompt, const int num_output_tuples, OutputType output_type, const nlohmann::json& media_data) = 0;
    // cacheblend 需要先自行按特殊分隔符拆 prompt，再把每段精确映射成 token id。
    // 这个接口就是为“直接提交 token id”的补全路径预留的，普通 query 仍然只走文本 prompt 接口。
    virtual void AddCompletionRequestTokenIds(const std::vector<int>& prompt_token_ids, const int num_output_tuples, OutputType output_type) {
        throw std::runtime_error("Prompt token ID completions are not supported by this provider");
    }
    virtual void AddEmbeddingRequest(const std::vector<std::string>& inputs) = 0;
    virtual void AddTranscriptionRequest(const nlohmann::json& audio_files) = 0;
    // cacheblend 路径要保留各段 token 边界，因此 provider 需要暴露底层 tokenizer 能力。
    // 不支持 cacheblend 的 provider 可以维持默认实现并在调用时抛错。
    virtual std::vector<int> TokenizePrompt(const std::string& prompt, bool add_special_tokens) {
        throw std::runtime_error("Prompt tokenization is not supported by this provider");
    }

    virtual std::vector<nlohmann::json> CollectCompletions(const std::string& contentType = "application/json") {
        return model_handler_->CollectCompletions(contentType);
    }
    virtual std::vector<nlohmann::json> CollectEmbeddings(const std::string& contentType = "application/json") {
        return model_handler_->CollectEmbeddings(contentType);
    }
    virtual std::vector<nlohmann::json> CollectTranscriptions(const std::string& contentType = "multipart/form-data") {
        return model_handler_->CollectTranscriptions(contentType);
    }

    static std::string GetOutputTypeString(const OutputType output_type) {
        switch (output_type) {
            case OutputType::STRING:
                return "string";
            case OutputType::OBJECT:
                return "object";
            case OutputType::BOOL:
                return "boolean";
            case OutputType::INTEGER:
                return "integer";
            default:
                throw std::invalid_argument("Unsupported output type");
        }
    }
};

class ExceededMaxOutputTokensError : public std::exception {
public:
    const char* what() const noexcept override {
        return "The response exceeded the max_output_tokens length; increase your max_output_tokens parameter.";
    }
};

}// namespace flock
