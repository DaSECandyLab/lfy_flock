#pragma once

#include "flock/core/common.hpp"
#include "flock/model_manager/model.hpp"

namespace flock {

struct LlmFunctionBindData : public duckdb::FunctionData {
    nlohmann::json model_json;// Store model JSON to create fresh Model instances per call
    std::string prompt;
    // 是否为该查询启用 cacheblend。只有显式打开时，执行阶段才会走 prompt 分段、tokenize 和 token-id completion 路径。
    bool cacheblend = false;
    // cacheblend 用来切分 prompt 的特殊分隔符，需要从 bind 阶段保留下来，供 execute 阶段继续使用。
    std::string blend_special_str;
    // 是否沿用旧的 continuation tokenization 规则：对后续 segment / separator 删掉第一个 token。
    bool cacheblend_remove_first_token = false;

    LlmFunctionBindData() = default;

    // Create a fresh Model instance (thread-safe, each call gets its own provider)
    Model CreateModel() const {
        return Model(model_json);
    }

    duckdb::unique_ptr<duckdb::FunctionData> Copy() const override {
        auto result = duckdb::make_uniq<LlmFunctionBindData>();
        result->model_json = model_json;
        result->prompt = prompt;
        result->cacheblend = cacheblend;
        result->blend_special_str = blend_special_str;
        result->cacheblend_remove_first_token = cacheblend_remove_first_token;
        return std::move(result);
    }

    bool Equals(const duckdb::FunctionData& other) const override {
        auto& other_bind = other.Cast<LlmFunctionBindData>();
        return prompt == other_bind.prompt &&
               cacheblend == other_bind.cacheblend &&
               blend_special_str == other_bind.blend_special_str &&
               cacheblend_remove_first_token == other_bind.cacheblend_remove_first_token &&
               model_json == other_bind.model_json;
    }
};

}// namespace flock
