"""Comparable streaming observations; engine counters remain the token authority."""
import asyncio
import hashlib
import json
import math
import time

import aiohttp


def summarize_response(events: list[dict], wall_s: float, ttft_s: float | None,
                       cache_mode: str) -> dict:
    if cache_mode not in ('cold', 'warm', 'natural'):
        raise ValueError('cache_mode must be cold, warm or natural')
    text, reasoning, tool_calls, errors = [], [], [], []
    usage, timings, finish = {}, {}, None
    for event in events:
        if event.get('error'):
            errors.append(event['error'])
        if event.get('usage'):
            usage = event['usage']
        if event.get('timings'):
            timings = event['timings']
        for choice in event.get('choices', []):
            delta = choice.get('delta') or choice.get('message') or {}
            text.append(delta.get('content') or '')
            reasoning.append(delta.get('reasoning_content') or '')
            tool_calls.extend(delta.get('tool_calls') or [])
            finish = choice.get('finish_reason') or finish
    tokens = usage.get('completion_tokens')
    cached = (usage.get('prompt_tokens_details') or {}).get('cached_tokens')
    valid_tokens = isinstance(tokens, int) and not isinstance(tokens, bool) and tokens > 0
    valid_time = math.isfinite(wall_s) and wall_s > 0
    if not finish:
        errors.append('missing terminal completion')
    if not valid_tokens:
        errors.append('missing or non-positive completion_tokens')
    if not valid_time:
        errors.append('invalid wall time')
    if cache_mode == 'cold' and cached and cached > 0:
        errors.append('cold request reports cached tokens')
    if ttft_s is not None and (not math.isfinite(ttft_s) or ttft_s < 0 or ttft_s > wall_s):
        errors.append('invalid TTFT')
    if not (''.join(text) or ''.join(reasoning) or tool_calls):
        errors.append('empty output')
    ok = not errors
    decode = None
    if ok and ttft_s is not None and wall_s > ttft_s and tokens > 1:
        # An observational estimate: the first SSE chunk can contain multiple tokens.
        decode = (tokens - 1) / (wall_s - ttft_s)
    return dict(ok=ok, errors=errors, wall_s=wall_s if valid_time else None,
                ttft_s=ttft_s, completion_tokens=tokens,
                prompt_tokens=usage.get('prompt_tokens'), cached_tokens=cached,
                cache_mode=cache_mode, e2e_tps=tokens / wall_s if ok else None,
                observed_decode_tps=decode, usage=usage, engine_timings=timings,
                text=''.join(text), reasoning=''.join(reasoning), tool_calls=tool_calls,
                finish_reason=finish, truncated=finish == 'length')


async def stream_request(session, endpoint: str, payload: dict, cache_mode: str) -> dict:
    started = time.perf_counter()
    ttft = None
    events = []
    try:
        async with session.post(endpoint.rstrip('/') + '/v1/chat/completions',
                                json=payload) as response:
            if response.status != 200:
                events.append({'error': f'HTTP {response.status}: {(await response.text())[:2000]}'})
            else:
                async for line in response.content:
                    line = line.strip()
                    if not line.startswith(b'data:'):
                        continue
                    data = line[5:].strip()
                    if data == b'[DONE]':
                        break
                    event = json.loads(data)
                    events.append(event)
                    if ttft is None:
                        for choice in event.get('choices', []):
                            delta = choice.get('delta') or {}
                            if any(delta.get(k) for k in ('content', 'reasoning_content', 'tool_calls')):
                                ttft = time.perf_counter() - started
                                break
    except (aiohttp.ClientError, asyncio.TimeoutError, ValueError) as exc:
        events.append({'error': f'{type(exc).__name__}: {exc}'})
    wall = time.perf_counter() - started
    return dict(summary=summarize_response(events, wall, ttft, cache_mode), events=events,
                request=payload, request_sha256=hashlib.sha256(
                    json.dumps(payload, sort_keys=True).encode()).hexdigest())


async def benchmark(endpoint: str, model: str, corpus: list[dict], output,
                    config: str, cache_mode: str, repeat: int, concurrency: int) -> list[dict]:
    records = []
    timeout = aiohttp.ClientTimeout(total=1800, sock_connect=10, sock_read=900)
    async with aiohttp.ClientSession(timeout=timeout,
                                   connector=aiohttp.TCPConnector(force_close=True)) as session:
        for repetition in range(repeat):
            for prompt in corpus:
                payload = dict(model=model, messages=prompt['messages'], stream=True,
                               stream_options={'include_usage': True}, temperature=0,
                               top_p=1, seed=42, max_tokens=prompt.get('max_tokens', 256),
                               presence_penalty=0, frequency_penalty=0,
                               cache_prompt=cache_mode != 'cold')
                payload.update(prompt.get('parameters', {}))
                if cache_mode == 'warm':
                    warmup = await stream_request(session, endpoint, payload, 'natural')
                    if not warmup['summary']['ok']:
                        raise RuntimeError(f'warmup failed: {warmup["summary"]["errors"]}')
                    output.write(json.dumps(dict(kind='warmup', config=config,
                                                 prompt_id=prompt['id'], **warmup)) + '\n')
                    output.flush()
                start = time.perf_counter()
                wave = await asyncio.gather(*[
                    stream_request(session, endpoint, payload, cache_mode) for _ in range(concurrency)])
                wave_s = time.perf_counter() - start
                for index, result in enumerate(wave):
                    record = dict(kind='trial', config=config, prompt_id=prompt['id'],
                                  repetition=repetition, concurrency=concurrency, index=index,
                                  wave_s=wave_s, **result)
                    records.append(record)
                    output.write(json.dumps(record, ensure_ascii=False) + '\n')
                    output.flush()
                    summary = result['summary']
                    print(json.dumps(dict(config=config, prompt=prompt['id'], rep=repetition,
                                          index=index, **{k: summary[k] for k in (
                                              'ok', 'errors', 'completion_tokens', 'ttft_s',
                                              'e2e_tps', 'observed_decode_tps', 'truncated')})), flush=True)
    return records
