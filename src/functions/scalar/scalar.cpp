#include "flock/functions/scalar/scalar.hpp"
#include "flock/model_manager/model.hpp"
#include <duckdb/planner/expression/bound_function_expression.hpp>

namespace flock {

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
                                                    const ScalarFunctionType function_type, Model& model) {
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

    // 这些列不拼进文本 prompt，而是作为当前行的图片/媒体附件传给 provider。
    auto is_image_column = [](const nlohmann::json& column) {
        if (!column.contains("type") || !column["type"].is_string()) {
            return false;
        }
        const auto column_type = column["type"].get<std::string>();
        return column_type == "image" || column_type == "media" || column_type == "photo" || column_type.rfind("image/", 0) == 0;
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

    // 先把当前 DuckDB chunk 内的每一行都转成一个独立请求并入队；这里不 collect，
    // 这样底层 curl multi 可以并发发起这些请求。
    for (auto row_idx = 0; row_idx < tuple_count; row_idx++) {
        std::string prompt = user_prompt;
        auto media_data = nlohmann::json::object();
        media_data["image"] = nlohmann::json::array();
        media_data["audio"] = nlohmann::json::array();

        for (auto column_idx = 0; column_idx < static_cast<int>(tuples.size()); column_idx++) {
            const auto row_column = build_single_row_column(tuples[column_idx], row_idx);

            if (is_image_column(tuples[column_idx])) {
                // 图片/媒体列只作为附件传递；不把文件路径/base64 当普通文本替换进 prompt。
                media_data["image"].push_back(row_column);
                continue;
            }

            std::string column_name = "COLUMN " + std::to_string(column_idx + 1);
            if (tuples[column_idx].contains("name") && tuples[column_idx]["name"].is_string()) {
                column_name = tuples[column_idx]["name"].get<std::string>();
            }

            const auto value = value_to_string(tuples[column_idx]["data"][row_idx]);
            const auto placeholder = "{{" + column_name + "}}";
            const auto replaced = replace_all(prompt, placeholder, value);
            if (!replaced) {
                // 用户 prompt 没写该列的 placeholder 时，按“列名:值;”追加到末尾，
                // 让“判断这条评论情感”这类无 placeholder prompt 仍能看到行上下文。
                prompt += "\n";
                prompt += column_name;
                prompt += ":";
                prompt += value;
                prompt += "; ";
            }
        }

        model.AddCompletionRequest(prompt, 1, output_type, media_data);
    }

    // 所有行请求都已入队后再统一 collect，保留 batch 内并发能力。
    const auto completion_responses = model.CollectCompletions();
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
