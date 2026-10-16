// Real embedded tokenizer/Frontend CPU checks; no engine, model, GPU or timings.
#include <ninfer/targets/qwen3_6/frontend.h>
#include <ninfer/targets/qwen3_6/frontend_resources.h>
#include "targets/qwen3_6/impl/frontend/test_access.h"
#include "media/decode/decode.h"
#include <nlohmann/json.hpp>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
using Json = nlohmann::json;
using namespace ninfer;
using namespace ninfer::targets::qwen3_6;
static void require(bool value, const char* message) {
    if (!value) { throw std::runtime_error(message); }
}
// Text-only tests link real Processor, tokenizer, frontend and xgrammar. FFmpeg
// development headers are unavailable locally; fail loudly if media is reached.
namespace ninfer::media::decode {
Image decode_image(std::span<const std::uint8_t>, const Policy&) {
    throw std::logic_error("media decoding is outside this text-only CPU contract");
}
Video decode_video(std::span<const std::uint8_t>, const Policy&, double, int, int) {
    throw std::logic_error("media decoding is outside this text-only CPU contract");
}
}
static FrontendResources resources(const char* template_path) {
    FrontendResources r;
    std::ifstream stream(template_path);
    r.chat_template_jinja = std::string(std::istreambuf_iterator<char>(stream), {});
    if (!r.chat_template_jinja.empty() && r.chat_template_jinja.back() == '\n') {
        r.chat_template_jinja.pop_back();
    }
    Json added = Json::array();
    for (const auto& [id, text] : std::vector<std::pair<int,std::string>>{
            {1,"helloST"},{2,"OPtail"},{3,"thought</thi"},{4,"nk>\n\nanswer"},
            {6,"<eos>"},{7,"<|im_start|>"},{8,"<|im_end|>"},{9,"user\n"},
            {13,"assistant\n"},{14,"\n"},{15,"<think>"},{16,"</think>"}}) {
        added.push_back({{"id",id},{"content",text},{"single_word",false},
                         {"lstrip",false},{"rstrip",false},{"normalized",false},
                         {"special",id==6 || id==7 || id==8}});
    }
    r.tokenizer_json = Json{{"model",{{"type","BPE"},{"vocab",{{"x",0},{"ä",10},{"¸",11},{"Ń",12}}},
                                                    {"merges",Json::array()}}},
                            {"added_tokens",added}}.dump();
    r.tokenizer_config_json = Json{{"add_bos_token",false},{"add_prefix_space",false},
        {"pad_token","<|endoftext|>"},{"chat_template",r.chat_template_jinja},
        {"added_tokens_decoder",Json::object()}}.dump();
    r.generation_config_json = R"({"eos_token_id":[6]})";
    r.preprocessor_config_json = R"({"patch_size":16,"temporal_patch_size":2,"merge_size":2,"image_mean":[0.5,0.5,0.5],"image_std":[0.5,0.5,0.5],"size":{"shortest_edge":4096,"longest_edge":16777216}})";
    r.video_preprocessor_config_json = r.preprocessor_config_json;
    return r;
}
int main(int argc, char** argv) {
    try {
        require(argc == 2, "template fixture required");
        auto frontend = FrontendTestAccess::create_component(resources(argv[1]), false);
        auto tokens = frontend.tokenize("xx<eos>");
        require(tokens == std::vector<TokenId>({0,0,6}), "raw tokenizer added a template/BOS or changed special tokens");
        auto prompt = frontend.prepare_tokens(tokens, false);
        const auto& data = FrontendTestAccess::inspect(prompt);
        require(data.token_ids == tokens && prompt.summary().prompt_tokens == 3, "raw prompt IDs changed");
        require(!data.identity.reusable && !data.starts_in_reasoning, "raw prompt identity/reasoning changed");
        try {
            (void)frontend.prepare_tokens({5}, false); // sparse vocabulary hole
            throw std::runtime_error("vocabulary hole accepted");
        } catch (const std::out_of_range&) {}
        StopPolicy stop;
        stop.include_model_defaults = false;
        OutputOptions raw;
        raw.raw = true;
        auto session = frontend.make_output_session(prompt, stop, raw);
        auto decision = session.preview(std::vector<TokenId>{6}, 3, FinishReason::OutputLimit);
        require(decision.accepted_tokens == 1 && decision.finish_reason == FinishReason::None, "ignore_eos stopped at EOS");
        auto output = session.commit_preview();
        require(output.size() == 1 && output[0].text == "<eos>" && output[0].tokens == 1, "raw EOS text/count lost");
        decision = session.preview(std::vector<TokenId>{3,4}, 2, FinishReason::OutputLimit);
        output = session.commit_preview();
        require(decision.accepted_tokens == 2 && decision.finish_reason == FinishReason::OutputLimit, "requested generation budget not honored");
        require(output.size() == 1 && output[0].text == "thought</think>\n\nanswer" && output[0].tokens == 2 && output[0].channel == OutputChannel::Content, "raw reasoning parsed or token count changed");
        auto defaults = frontend.make_output_session(prompt, StopPolicy{}, raw);
        decision = defaults.preview(std::vector<TokenId>{6,0}, 2, FinishReason::OutputLimit);
        require(decision.accepted_tokens == 1 && decision.finish_reason == FinishReason::StopToken, "normal EOS policy disabled");
        (void)defaults.commit_preview();
        stop.strings = {{.text="STOP",.channel=OutputChannel::Content,.include_in_output=false}};
        auto explicit_stop = frontend.make_output_session(prompt, stop, raw);
        decision = explicit_stop.preview(std::vector<TokenId>{1,2}, 4, FinishReason::OutputLimit);
        output = explicit_stop.commit_preview();
        require(decision.accepted_tokens == 2 && decision.finish_reason == FinishReason::StopString && output[0].text == "hello", "ignore_eos erased explicit stop strings");
        auto utf8 = frontend.make_output_session(prompt, StopPolicy{.include_model_defaults=false}, raw);
        decision = utf8.preview(std::vector<TokenId>{10}, 3, FinishReason::OutputLimit);
        output = utf8.commit_preview();
        require(output.size() == 1 && output[0].text.empty() && output[0].tokens == 1, "first empty-content UTF8 token event dropped");
        decision = utf8.preview(std::vector<TokenId>{11,12}, 2, FinishReason::OutputLimit);
        output = utf8.commit_preview();
        require(output.size() == 1 && output[0].text == "中" && output[0].tokens == 2, "UTF8 token accounting changed");
        PromptInput chat;
        chat.messages = {{.role="user",.parts={{.text="x"}}}};
        auto chat_prompt = frontend.prepare(std::move(chat));
        require(chat_prompt.summary().prompt_tokens > 1 && FrontendTestAccess::inspect(chat_prompt).identity.reusable && FrontendTestAccess::inspect(chat_prompt).starts_in_reasoning, "normal chat template/cache policy changed");
        auto thinking = frontend.make_output_session(chat_prompt, StopPolicy{});
        decision = thinking.preview(std::vector<TokenId>{3,4}, 3, FinishReason::OutputLimit);
        output = thinking.commit_preview();
        require(output.size() == 2 && output[0].channel == OutputChannel::Reasoning && output[1].text == "answer", "normal chat reasoning parsing changed");
        std::cout << "native embedded-tokenizer/frontend/EOS/empty-event/chat contracts passed\n";
        return 0;
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
