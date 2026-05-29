#include "duckdb/planner/expression/bound_function_expression.hpp"
#include "flock/functions/scalar/llm_complete.hpp"
#include "flock/functions/scalar/scalar.hpp"
#include "flock/metrics/manager.hpp"
#include "flock/model_manager/model.hpp"
#include <algorithm>
#include <cctype>

namespace {

bool ParseOptionalBoolRuntime(const nlohmann::json& value, bool default_value) {
    // 运行时再解析一次布尔开关。
    // 这样即使 bind 阶段没有把 cacheblend 等字段完整带下来，
    // 执行阶段也仍然能按本次请求的真实配置选择路径。
    if (value.is_boolean()) {
        return value.get<bool>();
    }
    if (value.is_number_integer()) {
        return value.get<int64_t>() != 0;
    }
    if (value.is_string()) {
        auto lowered = value.get<std::string>();
        std::transform(lowered.begin(), lowered.end(), lowered.begin(),
                       [](unsigned char c) { return std::tolower(c); });
        if (lowered == "true" || lowered == "1" || lowered == "yes" || lowered == "on") {
            return true;
        }
        if (lowered == "false" || lowered == "0" || lowered == "no" || lowered == "off") {
            return false;
        }
    }
    return default_value;
}

}// namespace

namespace flock {

duckdb::unique_ptr<duckdb::FunctionData> LlmComplete::Bind(
        duckdb::ClientContext& context,
        duckdb::ScalarFunction& bound_function,
        duckdb::vector<duckdb::unique_ptr<duckdb::Expression>>& arguments) {
    return ScalarFunctionBase::ValidateAndInitializeBindData(context, arguments, "llm_complete", false);
}


void LlmComplete::ValidateArguments(duckdb::DataChunk& args) {
    if (args.ColumnCount() < 2 || args.ColumnCount() > 3) {
        throw std::runtime_error("Invalid number of arguments.");
    }

    if (args.data[0].GetType().id() != duckdb::LogicalTypeId::STRUCT) {
        throw std::runtime_error("Model details must be a string.");
    }
    if (args.data[1].GetType().id() != duckdb::LogicalTypeId::STRUCT) {
        throw std::runtime_error("Prompt details must be a struct.");
    }

    if (args.ColumnCount() == 3) {
        if (args.data[2].GetType().id() != duckdb::LogicalTypeId::STRUCT) {
            throw std::runtime_error("Inputs must be a struct.");
        }
    }
}

std::vector<std::string> LlmComplete::Operation(duckdb::DataChunk& args, LlmFunctionBindData* bind_data) {
    Model model = bind_data->CreateModel();

    auto model_details = model.GetModelDetails();
    MetricsManager::SetModelInfo(model_details.model_name, model_details.provider_name);

    auto prompt_context_json = CastVectorOfStructsToJson(args.data[1], args.size());
    auto context_columns = nlohmann::json::array();
    if (prompt_context_json.contains("context_columns")) {
        context_columns = prompt_context_json["context_columns"];
    }

    auto prompt = bind_data->prompt;
    auto cacheblend = bind_data->cacheblend;
    // 优先读取当前请求里实际携带的 prompt 配置；
    // 如果没有，再使用 bind_data 中已有的值，保证非 cacheblend 查询继续走原路径。
    if (prompt_context_json.contains("cacheblend")) {
        cacheblend = ParseOptionalBoolRuntime(prompt_context_json["cacheblend"], cacheblend);
    }
    auto blend_special_str = bind_data->blend_special_str;
    if (prompt_context_json.contains("blend_special_str") && prompt_context_json["blend_special_str"].is_string()) {
        blend_special_str = prompt_context_json["blend_special_str"].get<std::string>();
    }
    auto cacheblend_remove_first_token = bind_data->cacheblend_remove_first_token;
    if (prompt_context_json.contains("cacheblend_remove_first_token")) {
        cacheblend_remove_first_token = ParseOptionalBoolRuntime(
                prompt_context_json["cacheblend_remove_first_token"],
                cacheblend_remove_first_token);
    }

    std::vector<std::string> results;
    if (context_columns.empty()) {
        auto template_str = prompt;
        model.AddCompletionRequest(template_str, 1, OutputType::STRING);
        auto response = model.CollectCompletions()[0]["items"][0];
        if (response.is_string()) {
            results.push_back(response.get<std::string>());
        } else {
            results.push_back(response.dump());
        }
    } else {
        if (context_columns.empty()) {
            return results;
        }

        auto responses = BatchAndComplete(
                context_columns, prompt, ScalarFunctionType::COMPLETE, model,
                cacheblend, blend_special_str, cacheblend_remove_first_token);

        results.reserve(responses.size());
        for (const auto& response: responses) {
            if (response.is_string()) {
                results.push_back(response.get<std::string>());
            } else {
                results.push_back(response.dump());
            }
        }
    }
    return results;
}

void LlmComplete::Execute(duckdb::DataChunk& args, duckdb::ExpressionState& state, duckdb::Vector& result) {
    auto& context = state.GetContext();
    auto* db = context.db.get();
    const void* invocation_id = MetricsManager::GenerateUniqueId();

    MetricsManager::StartInvocation(db, invocation_id, FunctionType::LLM_COMPLETE);

    auto exec_start = std::chrono::high_resolution_clock::now();

    auto& func_expr = state.expr.Cast<duckdb::BoundFunctionExpression>();
    auto* bind_data = &func_expr.bind_info->Cast<LlmFunctionBindData>();

    if (const auto results = LlmComplete::Operation(args, bind_data); static_cast<int>(results.size()) == 1) {
        auto empty_vec = duckdb::Vector(std::string());
        duckdb::UnaryExecutor::Execute<duckdb::string_t, duckdb::string_t>(
                empty_vec, result, args.size(),
                [&](duckdb::string_t name) { return duckdb::StringVector::AddString(result, results[0]); });
    } else {
        auto index = 0;
        for (const auto& res: results) {
            result.SetValue(index++, duckdb::Value(res));
        }
    }

    auto exec_end = std::chrono::high_resolution_clock::now();
    double exec_duration_ms = std::chrono::duration<double, std::milli>(exec_end - exec_start).count();
    MetricsManager::AddExecutionTime(exec_duration_ms);
}

}// namespace flock
