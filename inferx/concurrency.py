"""Measured arrivals and protocol observations, independent of serving-engine policy."""
import asyncio
import collections
import json
import math
import statistics
import time

import aiohttp

from inferx.tuning import stream_request


def reconstruct_tool_calls(events: list[dict]) -> list[dict]:
    calls = {}
    for event in events:
        for choice in event.get('choices', []):
            delta = choice.get('delta') or choice.get('message') or {}
            for position, part in enumerate(delta.get('tool_calls') or []):
                index = part.get('index', position)
                call = calls.setdefault(index, dict(id=None, name='', raw_arguments=''))
                if part.get('id'):
                    call['id'] = part['id']
                function = part.get('function', {})
                call['name'] += function.get('name') or ''
                call['raw_arguments'] += function.get('arguments') or ''
    result = []
    for index in sorted(calls):
        call = dict(calls[index])
        try:
            call['arguments'] = json.loads(call['raw_arguments'])
        except (ValueError, TypeError):
            call['arguments'] = None
        result.append(call)
    return result


def bandwidth_bytes_per_s(size: int, copies: int, elapsed_ms: float) -> float:
    if size <= 0 or copies <= 0 or not math.isfinite(elapsed_ms) or elapsed_ms <= 0:
        raise ValueError('positive size, copies and elapsed time required')
    return 2 * size * copies / (elapsed_ms / 1000)


def distribution(values: list[float]) -> dict:
    values = sorted(v for v in values if v is not None and math.isfinite(v))
    def quantile(q, minimum):
        return values[math.ceil(q * len(values)) - 1] if len(values) >= minimum else None
    return dict(n=len(values), median=statistics.median(values) if values else None,
                minimum=values[0] if values else None, maximum=values[-1] if values else None,
                p95=quantile(.95, 20), p99=quantile(.99, 100))


def summarize_load(records: list[dict], span_s: float) -> dict:
    if span_s <= 0 or not math.isfinite(span_s):
        raise ValueError('positive measured span required')
    if len({r['id'] for r in records}) != len(records):
        raise ValueError('duplicate request identity')
    good = [r for r in records if r['summary']['ok']]
    valid_tools = [r for r in good if r.get('functional_ok') is True]
    tokens = sum(r['summary']['completion_tokens'] for r in good)
    slo = {}
    for threshold in [.5, 1, 2, 5, 10]:
        protocol_count = sum(r['latency_s'] <= threshold for r in good)
        tool_count = sum(r['latency_s'] <= threshold for r in valid_tools)
        slo[str(threshold)] = dict(protocol_met=protocol_count, tool_met=tool_count,
            protocol_goodput_per_s=protocol_count/span_s, tool_goodput_per_s=tool_count/span_s,
            offered_fraction=protocol_count/max(1, len(records)))
    return dict(offered=len(records), protocol_ok=len(good), errors=len(records)-len(good),
        tool_valid=len(valid_tools), tool_checked=sum(r.get('functional_ok') is not None for r in records),
        truncated=sum(bool(r['summary'].get('truncated')) for r in good),
        completion_tokens=tokens, measured_span_s=span_s, aggregate_tokens_per_s=tokens/span_s,
        latency_s=distribution([r['latency_s'] for r in good]),
        attempted_latency_s=distribution([r['latency_s'] for r in records]),
        ttft_s=distribution([r.get('arrival_ttft_s') for r in good]),
        dispatch_lag_s=distribution([r.get('dispatch_lag_s', 0) for r in records]), slo=slo)


async def run_load(endpoint: str, requests: list[dict], output, mode: str,
                   workers: int) -> list[dict]:
    if mode not in {'open', 'closed'} or workers < 1:
        raise ValueError('open/closed mode and positive workers required')
    if len({r['id'] for r in requests}) != len(requests):
        raise ValueError('duplicate request identity')
    if mode == 'open' and any(r.get('offset_s', -1) < 0 for r in requests):
        raise ValueError('open-loop offsets must be nonnegative')
    origin = time.perf_counter()
    records = []
    timeout = aiohttp.ClientTimeout(total=180, sock_connect=10, sock_read=150)
    async with aiohttp.ClientSession(timeout=timeout,
            connector=aiohttp.TCPConnector(limit=0, force_close=True)) as session:
        async def execute(specification):
            if mode == 'open':
                await asyncio.sleep(max(0, origin + specification['offset_s'] - time.perf_counter()))
            dispatched = time.perf_counter() - origin
            scheduled = specification['offset_s'] if mode == 'open' else dispatched
            result = await stream_request(session, endpoint, specification['payload'], 'natural')
            finished = time.perf_counter() - origin
            lag = max(0, dispatched - scheduled)
            expected = specification.get('expected_tool')
            calls = reconstruct_tool_calls(result['events']) if expected is not None else []
            valid = (len(calls) == 1 and calls[0]['name'] == expected['name']
                     and calls[0]['arguments'] == expected['arguments']) if expected is not None else None
            record = dict(id=specification['id'], prompt_id=specification.get('prompt_id'),
                scheduled_s=scheduled, dispatched_s=dispatched, finished_s=finished,
                dispatch_lag_s=lag, latency_s=lag+result['summary']['wall_s'],
                arrival_ttft_s=lag+result['summary']['ttft_s']
                    if result['summary']['ttft_s'] is not None else None,
                functional_ok=valid, expected_tool=expected, reconstructed_tools=calls, **result)
            records.append(record)
            output.write(json.dumps(record, ensure_ascii=False) + '\n')
            output.flush()

        if mode == 'open':
            await asyncio.gather(*(execute(specification) for specification in requests))
        else:
            queue = collections.deque(requests)
            async def worker():
                while queue:
                    specification = queue.popleft()
                    await execute(specification)
            await asyncio.gather(*(worker() for _ in range(min(workers, len(requests)))))
    return records


def summarize_native(events: list[dict]) -> dict:
    done = [e for e in events if e.get('event') == 'request_done']
    starts = [e for e in events if e.get('event') == 'request_start']
    errors = [e for e in events if e.get('event') == 'request_error']
    accepted = sum(e.get('speculative', {}).get('accepted_tokens', 0) for e in done)
    drafted = sum(e.get('speculative', {}).get('drafted_tokens', 0) for e in done)
    rounds = sum(e.get('speculative', {}).get('rounds', 0) for e in done)
    batch = [e['decode_batch'] for e in events if e.get('event') == 'throughput' and 'decode_batch' in e]
    batch_rounds = sum(x['rounds'] for x in batch)
    return dict(request_starts=len(starts), request_done=len(done), request_errors=len(errors),
        completion_tokens=sum(e['result']['completion_tokens'] for e in done),
        prompt_tokens=sum(e['result']['prompt_tokens'] for e in done),
        cached_tokens=sum(e['result']['prefix_cache_hit_tokens'] for e in done),
        computed_prefill_tokens=sum(e['result'].get('computed_prefill_tokens', 0) for e in done),
        accepted_tokens=accepted, drafted_tokens=drafted, speculative_rounds=rounds,
        acceptance=accepted/drafted if drafted else None,
        accepted_per_round=accepted/rounds if rounds else None,
        realized_batch_size=sum(x['row_rounds'] for x in batch)/batch_rounds if batch_rounds else None,
        sampled_batch_rounds=batch_rounds,
        prefix_paths=dict(collections.Counter(e['result']['prefix_reuse_path'] for e in done)),
        request_timing_distributions_s={k: distribution([e['timings_seconds'][k] for e in done])
            for k in ['prepare', 'prefill', 'decode', 'ttft', 'total']},
        ttft_residual_s=distribution([e['timings_seconds']['ttft']-e['timings_seconds']['prepare']
                                     -e['timings_seconds']['prefill'] for e in done]),
        timer_warning='Request wall timers can overlap; their sums are not GPU execution time.')
