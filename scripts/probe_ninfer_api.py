#!/usr/bin/env python3
"""Live native completions contract probe. This performs real model requests.

Example: python scripts/probe_ninfer_api.py --base-url http://127.0.0.1:8000/v1 \
  --model MODEL --tokenizer /path/to/extracted-tokenizer \
  --prompt-tokens 8192 --max-tokens 512 --output results/ninfer-performance/probe.json

Requires the tokenizers package only for external exact prompt-count comparison.
A passing probe establishes API behavior, not benchmark qualification or speed.
"""
import argparse
import json
from pathlib import Path
import time
import urllib.error
import urllib.request


def request(base, path, payload=None, *, api_key="", timeout=1800):
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(base + path,
        data=None if payload is None else json.dumps(payload).encode(), headers=headers)
    return urllib.request.urlopen(req, timeout=timeout)


def check_completion(body, *, prompt_tokens, max_tokens):
    assert body["object"] == "text_completion", body
    assert len(body["choices"]) == 1, body
    choice = body["choices"][0]
    assert isinstance(choice["text"], str) and "message" not in choice, body
    assert choice["finish_reason"] == "length", body
    usage = body["usage"]
    assert usage["prompt_tokens"] == prompt_tokens, usage
    assert usage["completion_tokens"] == max_tokens, usage
    assert usage["total_tokens"] == prompt_tokens + max_tokens, usage
    assert usage.get("prompt_tokens_details", {}).get("cached_tokens", 0) == 0, usage


def probe(args):
    from tokenizers import Tokenizer
    base = args.base_url.rstrip("/")
    if not base.endswith("/v1"):
        base += "/v1"
    tokenizer_path = Path(args.tokenizer)
    if tokenizer_path.is_dir():
        tokenizer_path /= "tokenizer.json"
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    text_prompt = "Hello, world.\n"
    encoded = tokenizer.encode(text_prompt, add_special_tokens=False).ids
    assert encoded, "external tokenizer returned no IDs"
    raw_ids = (encoded * ((args.prompt_tokens + len(encoded) - 1) // len(encoded)))[:args.prompt_tokens]
    with request(base, "/models", api_key=args.api_key) as response:
        models = json.load(response)
    model = next(item for item in models["data"] if item["id"] == args.model)
    context = model.get("context_length", model.get("max_model_len", model.get("context_window")))
    assert context >= args.prompt_tokens + args.max_tokens, model
    common = {"model": args.model, "max_tokens": args.max_tokens, "temperature": 0,
              "seed": 1, "ignore_eos": True, "cache_salt": "probe-isolated"}
    token_request = {**common, "prompt": raw_ids}
    started = time.monotonic()
    with request(base, "/completions", token_request, api_key=args.api_key) as response:
        body = json.load(response)
    check_completion(body, prompt_tokens=len(raw_ids), max_tokens=args.max_tokens)
    nonstream_seconds = time.monotonic() - started
    # The string path must use the embedded artifact tokenizer without a template.
    with request(base, "/completions", {**common, "prompt": text_prompt}, api_key=args.api_key) as response:
        text_result = json.load(response)
    check_completion(text_result, prompt_tokens=len(encoded), max_tokens=args.max_tokens)
    pieces = []
    events = []
    final = None
    usage = None
    saw_done = False
    first_token_seconds = None
    started = time.monotonic()
    stream_request = {**token_request, "stream": True, "stream_options": {"include_usage": True}}
    with request(base, "/completions", stream_request, api_key=args.api_key) as response:
        assert response.headers.get_content_type() == "text/event-stream", response.headers
        for line in response:
            if not line.startswith(b"data: "):
                continue
            data = line[6:].strip()
            if data == b"[DONE]":
                saw_done = True
                break
            event = json.loads(data)
            assert "error" not in event, event
            assert event["object"] == "text_completion", event
            events.append(event)
            if not event["choices"]:
                assert usage is None, "multiple usage chunks"
                usage = event["usage"]
                continue
            choice = event["choices"][0]
            assert isinstance(choice["text"], str) and "delta" not in choice, event
            assert event["usage"] is None, event
            if choice["finish_reason"] is None:
                if first_token_seconds is None:
                    first_token_seconds = time.monotonic() - started
                pieces.append(choice["text"])
            else:
                assert final is None, "multiple terminal chunks"
                final = choice
    assert saw_done and final is not None and usage is not None and first_token_seconds is not None, events
    streamed = {"object":"text_completion", "choices":[{**final,"text":"".join(pieces)}], "usage":usage}
    check_completion(streamed, prompt_tokens=len(raw_ids), max_tokens=args.max_tokens)
    assert streamed["choices"][0]["text"] == body["choices"][0]["text"], "greedy stream/nonstream content differs"
    # Invalid vocabulary IDs must be rejected by the engine, with no usage invented.
    invalid_payloads = [({"prompt":[True]},400), ({"prompt":[[0]]},400),
                        ({"prompt":[2147483647]},400), ({"model":"__wrong_model__","prompt":[0]},400)]
    for invalid, wanted in invalid_payloads:
        try:
            with request(base,"/completions",{**common,**invalid},api_key=args.api_key) as response:
                raise AssertionError(f"invalid request accepted: {response.status}")
        except urllib.error.HTTPError as error:
            assert error.code == wanted, (error.code,error.read().decode())
    return {"status":"PASS", "base_url":base, "model":args.model,
            "prompt_tokens":len(raw_ids), "completion_tokens":args.max_tokens,
            "string_prompt_tokens":len(encoded), "ignore_eos":True,
            "nonstream_seconds":nonstream_seconds, "stream_first_token_seconds":first_token_seconds,
            "stream_token_events":len(pieces), "stream_total_events":len(events),
            "usage":usage, "context_length":context,
            "note":"Usage comes from the service; SSE chunks may contain multiple engine tokens. CPU frontend tests independently exercise EOS tokens."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--prompt-tokens", type=int, default=32)
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--api-key", default="")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.prompt_tokens <= 0 or args.max_tokens <= 0:
        parser.error("prompt-tokens and max-tokens must be positive")
    report = probe(args)
    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
