// CPU contract: exercises the native parser, request policy, and wire serializers.
// No GPU timings or model results are synthesized here.
#include "serve/openai_schema.h"
#include "serve/translate.h"
#include <iostream>
#include <stdexcept>
using namespace ninfer::serve;
using Json = nlohmann::json;
static void require(bool value, const char* message) {
    if (!value) { throw std::runtime_error(message); }
}
static GenerationRequest parse(const Json& body) {
    return parse_completion_request(body, RequestLimits{512}, "served-model");
}
static void rejected(Json body, const char* param) {
    try { (void)to_request_options(parse(body), ServeOptions{}); } catch (const ApiException& e) {
        require(e.error().status == 400, "invalid completion must return 400");
        require(e.error().param == param, "invalid field must be identified");
        return;
    }
    throw std::runtime_error("invalid completion accepted: " + body.dump());
}
static Json event(const std::string& wire) {
    require(wire.starts_with("data: ") && wire.ends_with("\n\n"), "invalid SSE framing");
    return Json::parse(wire.substr(6, wire.size() - 8));
}
int main() {
    try {
        auto text = parse({{"prompt", "  literal <think>\n"}, {"model", "served-model"}});
        require(text.raw_completion && text.prompt_text == "  literal <think>\n", "raw text changed");
        require(text.messages.empty(), "completion inserted chat messages");
        require(text.max_tokens == 512 && text.model == "served-model", "defaults lost");
        auto ids = parse({{"prompt", Json::array({0, 248046, 42, 42})}, {"max_tokens", 7},
                          {"ignore_eos", true}, {"stop", Json::array({"STOP"})},
                          {"temperature", 0.0}, {"top_p", 0.8}, {"top_k", 3},
                          {"seed", 4}, {"cache_salt", "request-a"}, {"stream", true},
                          {"stream_options", {{"include_usage", true}}}});
        require(ids.prompt_token_ids == std::vector<std::int32_t>({0,248046,42,42}), "IDs changed");
        require(ids.prompt_text.empty() && ids.messages.empty(), "IDs converted to text/chat");
        require(ids.stream && ids.include_usage && ids.ignore_eos, "stream/EOS flags lost");
        ninfer::serve::ServeOptions server;
        server.allow_prefix_reuse = true;
        auto options = to_request_options(ids, server);
        require(options.output.raw && !options.stop.include_model_defaults, "ignore_eos not applied");
        require(options.execution.requested_output_tokens == 7, "output budget changed");
        require(!options.execution.allow_prefix_reuse, "completion prefix reuse enabled");
        require(options.stop.strings.size() == 1 && options.stop.strings[0].text == "STOP", "explicit stop lost");
        require(options.execution.sampling.temperature == 0.0F && options.execution.sampling.top_k == 3, "sampling lost");
        require(to_request_options(text, server).stop.include_model_defaults, "EOS defaults lost");
        require(!to_request_options(text, server).execution.allow_prefix_reuse, "string cache not isolated");
        require(!resolve_prompt_semantics(ids, server, {}).enable_thinking, "raw inherits chat reasoning policy");
        for (const Json& prompt : {Json(), Json(""), Json::array(), Json(true), Json(42),
                                  Json::array({true}), Json::array({1.0}), Json::array({"hi"}),
                                  Json::array({Json::array({1})}), Json::array({-1}),
                                  Json::array({2147483648ULL})}) {
            rejected({{"prompt",prompt}}, "prompt");
        }
        rejected(Json::object(), "prompt");
        rejected({{"prompt","x"},{"model","wrong"}}, "model");
        rejected({{"prompt","x"},{"model",3}}, "model");
        for (const char* key : {"echo","logprobs","suffix","best_of","tools","messages","response_format","logit_bias"}) {
            rejected({{"prompt","x"},{key,true}}, key);
        }
        for (const Json& n : {Json(2), Json(0), Json(true), Json("1"), Json(nullptr)}) {
            rejected({{"prompt","x"},{"n",n}}, "n");
        }
        require(parse({{"prompt","x"},{"n",1}}).sampling.n == 1, "n=1 rejected");
        for (const char* key : {"ignore_eos","stream"}) {
            rejected({{"prompt","x"},{key,1}}, key);
            rejected({{"prompt","x"},{key,nullptr}}, key);
        }
        rejected({{"prompt","x"},{"cache_salt",1}}, "cache_salt");
        rejected({{"prompt","x"},{"cache_salt",nullptr}}, "cache_salt");
        rejected({{"prompt","x"},{"stream_options",true}}, "stream_options");
        rejected({{"prompt","x"},{"stream_options",{{"include_usage",1}}}}, "include_usage");
        for (const Json& budget : {Json(0), Json(-1), Json(true), Json("5"), Json(2147483648ULL), Json(nullptr)}) {
            rejected({{"prompt","x"},{"max_tokens",budget}}, "max_tokens");
        }
        rejected({{"prompt","x"},{"temperature",true}}, "temperature");
        rejected({{"prompt","x"},{"temperature",3}}, "temperature");
        rejected({{"prompt","x"},{"stop",Json::array({1})}}, "stop");
        auto chat = parse_chat_completion_request({{"messages", Json::array({{{"role","user"},{"content","hi"}}})}}, RequestLimits{8});
        const auto chat_options = to_request_options(chat, server);
        require(!chat.raw_completion && !chat_options.output.raw && chat_options.stop.include_model_defaults && chat_options.execution.allow_prefix_reuse, "chat policy changed");
        require(resolve_prompt_semantics(chat, server, {}).enable_thinking, "chat reasoning changed");
        const CompletionUsage usage{8192, 7, 0};
        auto response = Json::parse(make_completion_response("cmpl-test", "served-model", 1, "<think>x", "length", usage));
        require(response["object"] == "text_completion" && response["choices"][0]["text"] == "<think>x", "wrong completion shape");
        require(response["choices"][0]["finish_reason"] == "length" && !response["choices"][0].contains("message"), "chat serialization leaked");
        require(response["usage"]["prompt_tokens"] == 8192 && response["usage"]["completion_tokens"] == 7 && response["usage"]["total_tokens"] == 8199, "usage rewritten");
        auto empty = event(make_completion_chunk("cmpl-test", "served-model", 1, "", nullptr, true));
        require(empty["choices"][0]["text"] == "" && empty["choices"][0]["finish_reason"].is_null() && empty["usage"].is_null(), "empty token event lost");
        auto chunk = event(make_completion_chunk("cmpl-test", "served-model", 1, "abc", nullptr, false));
        require(chunk["object"] == "text_completion" && chunk["choices"][0]["text"] == "abc" && !chunk.contains("usage"), "stream shape wrong");
        auto final = event(make_completion_chunk("cmpl-test", "served-model", 1, "", "length", true));
        require(final["choices"][0]["finish_reason"] == "length", "final reason lost");
        auto stats = event(make_completion_usage_chunk("cmpl-test", "served-model", 1, usage));
        require(stats["choices"].empty() && stats["usage"]["completion_tokens"] == 7, "usage derived from chunks");
        auto models = Json::parse(make_models_list("served-model",1,8704,false));
        require(models["data"][0]["context_window"] == 8704 && models["data"][0]["max_model_len"] == 8704 && models["data"][0]["context_length"] == 8704, "model context aliases wrong");
        std::cout << "native completions schema/policy/serializer contracts passed\n";
        return 0;
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
