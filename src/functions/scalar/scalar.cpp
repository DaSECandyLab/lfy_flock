#include "flock/functions/scalar/scalar.hpp"
#include "flock/model_manager/model.hpp"
#include <duckdb/planner/expression/bound_function_expression.hpp>
#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdlib>
#include <iostream>

namespace flock {

namespace {

std::vector<std::string> SplitByDelimiter(const std::string& text, const std::string& delimiter) {
    // CacheBlend 需要把一条 prompt 拆成多个稳定片段分别分词，
    // 后续才能把“片段 token + 分隔符 token”重新拼成 prompt_token_ids。
    if (delimiter.empty()) {
        return {text};
    }

    std::vector<std::string> parts;
    size_t start = 0;
    size_t pos = 0;
    while ((pos = text.find(delimiter, start)) != std::string::npos) {
        parts.push_back(text.substr(start, pos - start));
        start = pos + delimiter.size();
    }
    parts.push_back(text.substr(start));
    return parts;
}



std::vector<int> TokenizeCacheBlendContinuation(Model& model, const std::string& text, bool remove_first_token) {
    // continuation segment / separator 先按“完整 prompt”方式 tokenize。
    // 只有显式启用兼容旧逻辑时，才去掉第一个 token。
    auto token_ids = model.TokenizePrompt(text, true);
    if (remove_first_token && !token_ids.empty()) {
        token_ids.erase(token_ids.begin());
    }
    return token_ids;
}

bool CacheBlendTimingDebugEnabled() {
    const char* env_value = std::getenv("FLOCK_DEBUG_CACHEBLEND_TIMING");
    if (env_value == nullptr) {
        return false;
    }
    auto value = std::string(env_value);
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return !(value.empty() || value == "0" || value == "false" || value == "off" || value == "no");
}

bool CacheBlendPromptDebugEnabled() {
    const char* env_value = std::getenv("FLOCK_DEBUG_CACHEBLEND_PROMPT");
    if (env_value == nullptr) {
        return false;
    }
    auto value = std::string(env_value);
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return !(value.empty() || value == "0" || value == "false" || value == "off" || value == "no");
}

std::string SanitizeDebugSnippet(std::string text, const size_t max_len = 240) {
    for (auto& ch: text) {
        if (ch == '\n' || ch == '\r' || ch == '\t') {
            ch = ' ';
        }
    }
    if (text.size() <= max_len) {
        return text;
    }
    return text.substr(0, max_len) + "...";
}

}// namespace

void ScalarFunctionBase::ValidateArgumentCount(
        const duckdb::vector<duckdb::unique_ptr<duckdb::Expression>>& arguments,
        const std::string& function_name) {
    if (arguments.size() != 2) {
        throw duckdb::BinderException(
                function_name + " requires 2 arguments: (1) model, (2) prompt with context_columns. Got " +
                std::to_string(arguments.size()));
    }
}

void ScalarFunctionBase::ValidateArgumentTypes(
        const duckdb::vector<duckdb::unique_ptr<duckdb::Expression>>& arguments,
        const std::string& function_name) {
    if (arguments[0]->return_type.id() != duckdb::LogicalTypeId::STRUCT) {
        throw duckdb::BinderException(function_name + ": First argument must be model (struct type)");
    }
    if (arguments[1]->return_type.id() != duckdb::LogicalTypeId::STRUCT) {
        throw duckdb::BinderException(
                function_name + ": Second argument must be prompt with context_columns (struct type)");
    }
}

ScalarFunctionBase::PromptStructInfo ScalarFunctionBase::ExtractPromptStructInfo(
        const duckdb::LogicalType& prompt_type) {
    PromptStructInfo info{false, std::nullopt, ""};

    for (idx_t i = 0; i < duckdb::StructType::GetChildCount(prompt_type); i++) {
        auto field_name = duckdb::StructType::GetChildName(prompt_type, i);
        if (field_name == "context_columns") {
            info.has_context_columns = true;
        } else if (field_name == "prompt" || field_name == "prompt_name") {
            if (!info.prompt_field_index.has_value()) {
                info.prompt_field_index = i;
                info.prompt_field_name = field_name;
            }
        }
    }

    return info;
}

void ScalarFunctionBase::ValidatePromptStructFields(const PromptStructInfo& info,
                                                    const std::string& function_name,
                                                    bool require_context_columns) {
    if (require_context_columns && !info.has_context_columns) {
        throw duckdb::BinderException(
                function_name + ": Second argument must contain 'context_columns' field");
    }
}

void ScalarFunctionBase::InitializeModelJson(
        duckdb::ClientContext& context,
        const duckdb::unique_ptr<duckdb::Expression>& model_expr,
        LlmFunctionBindData& bind_data) {
    if (!model_expr->IsFoldable()) {
        return;
    }

    auto model_value = duckdb::ExpressionExecutor::EvaluateScalar(context, *model_expr);
    auto user_model_json = CastValueToJson(model_value);
    bind_data.model_json = Model::ResolveModelDetailsToJson(user_model_json);
}

nlohmann::json ScalarFunctionBase::Complete(nlohmann::json& columns, const std::string& user_prompt,
                                            ScalarFunctionType function_type, Model& model) {
    const auto [prompt, media_data] = PromptManager::Render(user_prompt, columns, function_type, model.GetModelDetails().tuple_format);
    OutputType output_type = OutputType::STRING;
    if (function_type == ScalarFunctionType::FILTER) {
        output_type = OutputType::BOOL;
    }

    model.AddCompletionRequest(prompt, static_cast<int>(columns[0]["data"].size()), output_type, media_data);
    auto response = model.CollectCompletions();
    return response[0]["items"];
};

nlohmann::json ScalarFunctionBase::BatchAndComplete(const nlohmann::json& tuples,
                                                    const std::string& user_prompt,
                                                    const ScalarFunctionType function_type, Model& model,
                                                    const bool cacheblend,
                                                    const std::string& blend_special_str,
                                                    const bool cacheblend_remove_first_token) {
    const auto batch_start = std::chrono::steady_clock::now();
    const auto tuple_count = static_cast<int>(tuples[0]["data"].size());
    auto responses = nlohmann::json::array();

    // 将单元格值转成可直接放入 prompt 的文本；字符串不保留 JSON 引号。
    auto value_to_string = [](const nlohmann::json& value) {
        if (value.is_null()) {
            return std::string("");
        }
        if (value.is_string()) {
            return value.get<std::string>();
        }
        return value.dump();
    };

    // 普通列仍优先走原有 placeholder 语义：找到 {{列名}} 就做逐处替换。
    auto replace_all = [](std::string& prompt, const std::string& placeholder, const std::string& value) {
        size_t pos = 0;
        bool replaced = false;
        while ((pos = prompt.find(placeholder, pos)) != std::string::npos) {
            prompt.replace(pos, placeholder.size(), value);
            pos += value.size();
            replaced = true;
        }
        return replaced;
    };

    // 这些列不拼进文本 prompt，而是作为当前行的附件传给 provider。
    auto is_image_column = [](const nlohmann::json& column) {
        return PromptManager::IsImageColumn(column);
    };


    // provider 每个请求只对应一行，因此 media_data 中也只保留当前行的数据。
    auto build_single_row_column = [](const nlohmann::json& column, const int row_idx) {
        auto row_column = nlohmann::json::object();
        for (const auto& item: column.items()) {
            if (item.key() == "data") {
                row_column["data"] = nlohmann::json::array({item.value()[row_idx]});
            } else {
                row_column[item.key()] = item.value();
            }
        }
        return row_column;
    };

    OutputType output_type = OutputType::STRING;
    if (function_type == ScalarFunctionType::FILTER) {
        output_type = OutputType::BOOL;
    }

    std::vector<std::string> prompt_segments;
    std::vector<int> separator_token_ids;
    double split_prompt_ms = 0.0;
    double separator_tokenize_ms = 0.0;
    double placeholder_replace_ms = 0.0;
    double segment_tokenize_ms = 0.0;
    double assemble_prompt_ids_ms = 0.0;
    double enqueue_completion_ms = 0.0;
    size_t total_segment_count = 0;
    size_t total_prompt_token_ids = 0;
    size_t total_segment_token_calls = 0;
    size_t total_placeholder_columns = 0;
    if (cacheblend) {
        const auto split_start = std::chrono::steady_clock::now();
        // 开启 CacheBlend 时，仍然先按约定分隔符拆成若干文本片段，
        // 这样每一行替换 placeholder 时可以稳定地保留片段边界。
        if (blend_special_str.empty()) {
            throw std::runtime_error("cacheblend is enabled but blend_special_str is empty");
        }
        if (user_prompt.find(blend_special_str) == std::string::npos) {
            throw std::runtime_error("cacheblend is enabled but the prompt does not contain blend_special_str");
        }
        prompt_segments = SplitByDelimiter(user_prompt, blend_special_str);
        const auto split_end = std::chrono::steady_clock::now();
        split_prompt_ms += std::chrono::duration<double, std::milli>(split_end - split_start).count();
        // 与 LMCache CacheBlend 示例保持一致：
        // 第一个 segment 保留完整 encode；后续 segment 和 separator 是否去掉第一个 token
        // 由 cacheblend_remove_first_token 显式控制。
        const auto separator_start = std::chrono::steady_clock::now();
        separator_token_ids = TokenizeCacheBlendContinuation(
                model, blend_special_str, cacheblend_remove_first_token);
        const auto separator_end = std::chrono::steady_clock::now();
        separator_tokenize_ms += std::chrono::duration<double, std::milli>(separator_end - separator_start).count();
        if (separator_token_ids.empty()) {
            throw std::runtime_error(
                    "cacheblend separator becomes empty after LMCache-style tokenization");
        }
    }

    // 先把当前 DuckDB chunk 内的每一行都转成一个独立请求并入队；这里不 collect，
    // 这样底层 curl multi 可以并发发起这些请求。
    for (auto row_idx = 0; row_idx < tuple_count; row_idx++) {
        std::string prompt = user_prompt;
        auto row_segments = prompt_segments;
        auto media_data = nlohmann::json::object();
        media_data["image"] = nlohmann::json::array();

        for (auto column_idx = 0; column_idx < static_cast<int>(tuples.size()); column_idx++) {
            const auto row_column = build_single_row_column(tuples[column_idx], row_idx);

            if (is_image_column(tuples[column_idx])) {
                // 附件列只作为请求附件传递；不把文件路径/base64 当普通文本替换进 prompt。
                media_data["image"].push_back(row_column);
                continue;
            }


            const auto prompt_column = row_column;

            std::string column_name = "COLUMN " + std::to_string(column_idx + 1);
            if (prompt_column.contains("name") && prompt_column["name"].is_string()) {
                column_name = prompt_column["name"].get<std::string>();
            }

            auto cell_value = nlohmann::json(nullptr);
            if (prompt_column.contains("data") && prompt_column["data"].is_array() && !prompt_column["data"].empty()) {
                cell_value = prompt_column["data"][0];
            }
            const auto value = value_to_string(cell_value);
            const auto placeholder = "{{" + column_name + "}}";
            auto replaced = false;
            const auto replace_start = cacheblend ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
            if (cacheblend) {
                for (auto& segment: row_segments) {
                    if (replace_all(segment, placeholder, value)) {
                        replaced = true;
                    }
                }
                const auto replace_end = std::chrono::steady_clock::now();
                placeholder_replace_ms += std::chrono::duration<double, std::milli>(replace_end - replace_start).count();
                total_placeholder_columns++;
            } else {
                replaced = replace_all(prompt, placeholder, value);
            }
            if (!replaced) {
                // 用户 prompt 没写该列的 placeholder 时，按“列名:值;”追加到末尾，
                // 让“判断这条评论情感”这类无 placeholder prompt 仍能看到行上下文。
                auto fallback = std::string("\n") + column_name + ":" + value + "; ";
                if (cacheblend) {
                    if (row_segments.empty()) {
                        row_segments.push_back(fallback);
                    } else {
                        row_segments.back() += fallback;
                    }
                } else {
                    prompt += fallback;
                }
            }
        }

        if (cacheblend && row_idx == 0 && CacheBlendPromptDebugEnabled()) {
            size_t review1_template_segments = 0;
            size_t review1_unreplaced_segments = 0;
            size_t review2_template_segments = 0;
            size_t review2_unreplaced_segments = 0;
            size_t unresolved_placeholder_segments = 0;
            for (size_t segment_idx = 0; segment_idx < prompt_segments.size() && segment_idx < row_segments.size(); ++segment_idx) {
                if (prompt_segments[segment_idx].find("{{review1}}") != std::string::npos) {
                    review1_template_segments++;
                    if (row_segments[segment_idx].find("{{review1}}") != std::string::npos) {
                        review1_unreplaced_segments++;
                    }
                    std::cerr << duckdb_fmt::format(
                            "[FLOCK_CACHEBLEND_PROMPT_DEBUG] row=0 placeholder=review1 segment={} replaced={} text=\"{}\"\n",
                            segment_idx,
                            row_segments[segment_idx].find("{{review1}}") == std::string::npos ? "true" : "false",
                            SanitizeDebugSnippet(row_segments[segment_idx]));
                }
                if (prompt_segments[segment_idx].find("{{review2}}") != std::string::npos) {
                    review2_template_segments++;
                    if (row_segments[segment_idx].find("{{review2}}") != std::string::npos) {
                        review2_unreplaced_segments++;
                    }
                    std::cerr << duckdb_fmt::format(
                            "[FLOCK_CACHEBLEND_PROMPT_DEBUG] row=0 placeholder=review2 segment={} replaced={} text=\"{}\"\n",
                            segment_idx,
                            row_segments[segment_idx].find("{{review2}}") == std::string::npos ? "true" : "false",
                            SanitizeDebugSnippet(row_segments[segment_idx]));
                }
                if (row_segments[segment_idx].find("{{") != std::string::npos ||
                    row_segments[segment_idx].find("}}") != std::string::npos) {
                    unresolved_placeholder_segments++;
                    std::cerr << duckdb_fmt::format(
                            "[FLOCK_CACHEBLEND_PROMPT_DEBUG] row=0 unresolved_placeholder segment={} text=\"{}\"\n",
                            segment_idx,
                            SanitizeDebugSnippet(row_segments[segment_idx]));
                }
            }
            std::cerr << duckdb_fmt::format(
                    "[FLOCK_CACHEBLEND_PROMPT_DEBUG] row=0 review1_segments={} review1_unreplaced={} review2_segments={} review2_unreplaced={} unresolved_placeholder_segments={}\n",
                    review1_template_segments,
                    review1_unreplaced_segments,
                    review2_template_segments,
                    review2_unreplaced_segments,
                    unresolved_placeholder_segments);
            if (unresolved_placeholder_segments > 0) {
                throw std::runtime_error(
                        "cacheblend row 0 still contains unresolved placeholders after prompt replacement");
            }
        }

        if (cacheblend) {
            if (media_data.contains("image") && !media_data["image"].empty()) {
                throw std::runtime_error("cacheblend path does not support image attachments");
            }

            // CacheBlend 路径不走字符串 prompt，而是显式构造 prompt_token_ids：
            // [片段1 tokens][分隔符 tokens][片段2 tokens]...
            // 这样请求最终会落到 /v1/completions，并把分段后的 token 边界保留下来。
            total_segment_count += row_segments.size();
            std::vector<int> prompt_token_ids;
            const auto tokenize_segments_start = std::chrono::steady_clock::now();
            for (size_t segment_idx = 0; segment_idx < row_segments.size(); ++segment_idx) {
                const auto segment_token_ids = segment_idx == 0
                                                       ? model.TokenizePrompt(
                                                                 row_segments[segment_idx], true)
                                                       : TokenizeCacheBlendContinuation(
                                                                 model,
                                                                 row_segments[segment_idx],
                                                                 cacheblend_remove_first_token);
                total_segment_token_calls++;
                prompt_token_ids.insert(
                        prompt_token_ids.end(),
                        segment_token_ids.begin(),
                        segment_token_ids.end());
                if (segment_idx + 1 < row_segments.size()) {
                    prompt_token_ids.insert(
                            prompt_token_ids.end(),
                            separator_token_ids.begin(),
                            separator_token_ids.end());
                }
            }
            const auto tokenize_segments_end = std::chrono::steady_clock::now();
            segment_tokenize_ms += std::chrono::duration<double, std::milli>(tokenize_segments_end - tokenize_segments_start).count();
            total_prompt_token_ids += prompt_token_ids.size();
            const auto enqueue_start = std::chrono::steady_clock::now();
            model.AddCompletionRequestTokenIds(prompt_token_ids, 1, output_type);
            const auto enqueue_end = std::chrono::steady_clock::now();
            assemble_prompt_ids_ms += std::chrono::duration<double, std::milli>(enqueue_start - tokenize_segments_end).count();
            enqueue_completion_ms += std::chrono::duration<double, std::milli>(enqueue_end - enqueue_start).count();
        } else {
            model.AddCompletionRequest(prompt, 1, output_type, media_data);
        }
    }

    // 所有行请求都已入队后再统一 collect，保留 batch 内并发能力。
    const auto collect_start = std::chrono::steady_clock::now();
    const auto completion_responses = model.CollectCompletions();
    const auto collect_end = std::chrono::steady_clock::now();
    const auto collect_ms = std::chrono::duration<double, std::milli>(collect_end - collect_start).count();
    for (auto response_idx = 0; response_idx < tuple_count; response_idx++) {
        if (response_idx >= static_cast<int>(completion_responses.size()) ||
            !completion_responses[response_idx].contains("items") ||
            !completion_responses[response_idx]["items"].is_array() ||
            completion_responses[response_idx]["items"].empty()) {
            responses.push_back(nullptr);
            continue;
        }
        // CollectCompletions 返回顺序由底层 handler 按请求下标恢复，这里按行号取回结果。
        responses.push_back(completion_responses[response_idx]["items"][0]);
    }

    if (cacheblend && CacheBlendTimingDebugEnabled()) {
        const auto batch_end = std::chrono::steady_clock::now();
        const auto batch_ms = std::chrono::duration<double, std::milli>(batch_end - batch_start).count();
        std::cerr << duckdb_fmt::format(
                "[FLOCK_CACHEBLEND_BATCH_TIMING] tuples={} prompt_segments={} placeholder_columns={} segment_tokenize_calls={} prompt_token_ids_total={} split_prompt_ms={:.3f} separator_tokenize_ms={:.3f} placeholder_replace_ms={:.3f} segment_tokenize_ms={:.3f} assemble_prompt_ids_ms={:.3f} enqueue_completion_ms={:.3f} collect_ms={:.3f} batch_total_ms={:.3f}\n",
                tuple_count,
                total_segment_count,
                total_placeholder_columns,
                total_segment_token_calls,
                total_prompt_token_ids,
                split_prompt_ms,
                separator_tokenize_ms,
                placeholder_replace_ms,
                segment_tokenize_ms,
                assemble_prompt_ids_ms,
                enqueue_completion_ms,
                collect_ms,
                batch_ms);
    }

    return responses;
}

void ScalarFunctionBase::InitializePrompt(
        duckdb::ClientContext& context,
        const duckdb::unique_ptr<duckdb::Expression>& prompt_expr,
        LlmFunctionBindData& bind_data) {
    nlohmann::json prompt_json;

    if (prompt_expr->IsFoldable()) {
        auto prompt_value = duckdb::ExpressionExecutor::EvaluateScalar(context, *prompt_expr);
        prompt_json = CastValueToJson(prompt_value);
    } else if (prompt_expr->expression_class == duckdb::ExpressionClass::BOUND_FUNCTION) {
        auto& func_expr = prompt_expr->Cast<duckdb::BoundFunctionExpression>();
        const auto& struct_type = prompt_expr->return_type;

        for (idx_t i = 0; i < duckdb::StructType::GetChildCount(struct_type) && i < func_expr.children.size(); i++) {
            auto field_name = duckdb::StructType::GetChildName(struct_type, i);
            auto& child = func_expr.children[i];

            if (field_name != "context_columns" && child->IsFoldable()) {
                try {
                    auto field_value = duckdb::ExpressionExecutor::EvaluateScalar(context, *child);
                    if (field_value.type().id() == duckdb::LogicalTypeId::VARCHAR) {
                        prompt_json[field_name] = field_value.GetValue<std::string>();
                    } else {
                        prompt_json[field_name] = CastValueToJson(field_value);
                    }
                } catch (...) {
                    // Skip fields that can't be evaluated
                }
            }
        }
    }

    if (prompt_json.contains("context_columns")) {
        prompt_json.erase("context_columns");
    }

    // cacheblend / blend_special_str 不是 PromptManager 的通用 prompt 字段，
    // 需要在这里单独抽出来，保存到 bind_data，供执行阶段决定是否走 token-id 路径。
    bind_data.cacheblend = false;
    bind_data.blend_special_str.clear();
    bind_data.cacheblend_remove_first_token = false;
    if (prompt_json.contains("cacheblend")) {
        bind_data.cacheblend = ParseOptionalBool(prompt_json["cacheblend"], false);
        prompt_json.erase("cacheblend");
    }
    if (prompt_json.contains("blend_special_str")) {
        if (!prompt_json["blend_special_str"].is_string()) {
            throw std::runtime_error("blend_special_str must be a string");
        }
        bind_data.blend_special_str = prompt_json["blend_special_str"].get<std::string>();
        prompt_json.erase("blend_special_str");
    }
    if (prompt_json.contains("cacheblend_remove_first_token")) {
        bind_data.cacheblend_remove_first_token =
                ParseOptionalBool(prompt_json["cacheblend_remove_first_token"], false);
        prompt_json.erase("cacheblend_remove_first_token");
    }
    if (bind_data.cacheblend && bind_data.blend_special_str.empty()) {
        // 给 cacheblend 一个稳定默认分隔符，避免每个 query 都必须显式传同一常量。
        bind_data.blend_special_str = " # # ";
    }

    auto prompt_details = PromptManager::CreatePromptDetails(prompt_json);
    bind_data.prompt = prompt_details.prompt;
}

duckdb::unique_ptr<LlmFunctionBindData> ScalarFunctionBase::ValidateAndInitializeBindData(
        duckdb::ClientContext& context,
        duckdb::vector<duckdb::unique_ptr<duckdb::Expression>>& arguments,
        const std::string& function_name,
        bool require_context_columns,
        bool initialize_prompt) {

    ValidateArgumentCount(arguments, function_name);
    ValidateArgumentTypes(arguments, function_name);

    const auto& prompt_type = arguments[1]->return_type;
    auto prompt_info = ExtractPromptStructInfo(prompt_type);
    ValidatePromptStructFields(prompt_info, function_name, require_context_columns);

    auto bind_data = duckdb::make_uniq<LlmFunctionBindData>();

    InitializeModelJson(context, arguments[0], *bind_data);
    if (initialize_prompt) {
        InitializePrompt(context, arguments[1], *bind_data);
    }
    return bind_data;
}

}// namespace flock
