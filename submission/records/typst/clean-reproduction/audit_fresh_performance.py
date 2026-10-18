#!/usr/bin/env python3
"""CPU-only independent audit of the completed clean-build C14 run."""
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / 'results/three-node-optimization/ucloud2/full-quantization-reproduction-116'
RAW = BASE / 'fresh-C14-raw-001'
OUT = Path(__file__).resolve().parent
BINARY = 'e919f18084f3cc09718158cfae071cdf8c58d5660715648db375643016c878f5'
SOURCE = 'c976692d9657408308193b50772c700382a6e243d9da353165205dc5252f76a8'
RECIPE = '4b7d1b7542291948bbec1af23fe99973189d48d6e171bcabc4a40e0b6c123a59'
INPUT = '6a22077fc53f81d60176e844adc01dd2dfeb7af19adebe9069f1d9e52969c7d1'


def audit():
    sources = {}

    def read(path, lines=False):
        raw = path.read_bytes()
        sources[str(path.relative_to(ROOT))] = {
            'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw)}
        return [json.loads(line) for line in raw.splitlines() if line] if lines else json.loads(raw)

    def close(a, b):
        assert math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-8), (a, b)

    def p95(values):
        values = sorted(values)
        rank = .95 * (len(values) - 1)
        lo, hi = math.floor(rank), math.ceil(rank)
        return values[lo] + (rank-lo) * (values[hi]-values[lo])

    summary = read(RAW / 'deployment116/performance-C14/summary.json')
    identity = read(RAW / 'deployment116/performance-C14/evaluation-identity.json')
    launch = read(RAW / 'deployment116/service-perf-c14/launch.json')
    phase = read(RAW / 'deployment116/performance-C14/c14/phases/profiling/profile_export_aiperf.json')
    windows = read(RAW / 'deployment116/performance-C14/c14/phase_manifest.json')
    records = read(RAW / 'deployment116/performance-C14/c14/profile_export.jsonl', lines=True)
    native = read(RAW / 'deployment116/service-perf-c14/requests.jsonl', lines=True)
    payload = read(RAW / 'deployment116/performance-C14/c14/inputs.json')
    terminal = read(RAW / 'final-evaluation-job.json')
    assert launch['binary_sha256'] == BINARY
    assert launch['source_sha256'] == SOURCE
    assert launch['recipe_sha256'] == identity['recipe_sha256'] == RECIPE
    assert identity['status'] == 'complete'
    assert summary['status'] == '完整' and not summary['issues']
    assert summary['concurrency'] == 14 and summary['records'] == summary['valid_records'] == 128
    assert summary['error_requests'] == 0 and summary['lengths_verified'] and summary['osl_verified']
    assert Counter(r['metadata']['benchmark_phase'] for r in records) == {'profiling':128,'warmup':8}
    formal = [r for r in records if r['metadata']['benchmark_phase'] == 'profiling']
    assert len({r['metadata']['x_request_id'] for r in formal}) == 128
    for record in records:
        metrics = record['metrics']
        assert metrics['input_sequence_length']['value'] == 8192
        assert metrics['output_sequence_length']['value'] == 512
        assert metrics['usage_prompt_cache_read_tokens']['value'] == 0
        assert not record['metadata']['was_cancelled'] and not record['metadata']['context_overflow_skip']
    starts = [r for r in native if r['event'] == 'server_start']
    done = [r for r in native if r['event'] == 'request_done']
    assert len(starts) == 1 and len(done) == 137
    engine = starts[0]['engine']
    for key, value in {'max_concurrency':13,'kv_capacity':121856,'max_context':65536,
                       'speculative_draft_window':3,'speculative_backend':'mtp',
                       'prefill_chunk':2048,'prefix_reuse':False}.items():
        assert engine[key] == value, (key, engine[key])
    assert Counter(r['result']['completion_tokens'] for r in done) == {512:136,1:1}
    assert all(r['result']['prompt_tokens'] == r['result']['computed_prefill_tokens'] == 8192
               and r['result']['prefix_cache_hit_tokens'] == 0 for r in done)
    assert len(payload['data']) == 137
    canonical = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',',':'), ensure_ascii=False).encode()).hexdigest()
    assert canonical == identity['actual_inputs_canonical_sha256'] == INPUT
    ttft = p95([r['metrics']['time_to_first_token']['value'] for r in formal])
    tpot = p95([(r['metrics']['request_latency']['value'] - r['metrics']['time_to_first_token']['value']) / 511 for r in formal])
    for name, value in [('time_to_first_token',ttft),('inter_token_latency',tpot)]:
        close(value, summary['metrics'][name]['p95'])
        close(value, phase[name]['p95'])
    window = next(p for p in windows['phases'] if p['phase_kind'] == 'profiling')
    assert window['total_request_count'] == window['successful_request_count'] == 128
    assert window['error_request_count'] == 0 and not window['was_cancelled']
    duration = (window['end_ns']-window['start_ns'])/1e9
    tokens = sum(r['metrics']['output_token_count']['value'] for r in formal)
    assert tokens == 65536
    throughput = tokens/duration
    close(throughput, summary['output_token_throughput'])
    close(throughput, phase['output_token_throughput']['avg'])
    perf_run = next(r for r in terminal['runs'] if r['mode']=='performance')
    assert perf_run['evaluation_identity']['status']=='complete'
    assert perf_run['service_terminal']['returncode']==0 and perf_run['service_terminal']['reaped']
    # Keep exact source fields and numbers. SLO is derived separately because the
    # frozen parser has no meets_slo field and the entrypoint receipt says null.
    excerpt = {k: summary[k] for k in ['concurrency','records','error_requests']}
    excerpt['metrics'] = {k:{'p95':summary['metrics'][k]['p95']} for k in ['time_to_first_token','inter_token_latency']}
    excerpt['output_token_throughput'] = summary['output_token_throughput']
    raw_excerpt = '{\n' + ''.join(f'  "{k}": {excerpt[k]},\n' for k in ['concurrency','records','error_requests'])
    raw_excerpt += '  "metrics": {\n' + ',\n'.join(f'    "{k}": '+json.dumps(v) for k,v in excerpt['metrics'].items()) + '\n  },\n'
    raw_excerpt += '  "output_token_throughput": '+str(excerpt['output_token_throughput'])+'\n}'
    assert json.loads(raw_excerpt) == excerpt
    result = dict(status='PASS', binary_sha256=BINARY,source_sha256=SOURCE,recipe_sha256=RECIPE,
        client_concurrency=14,native_concurrency=13,kv=121856,measured=128,warmup=8,probe=1,
        ttft_p95_ms=ttft,tpot_p95_ms=tpot,output_tokens=tokens,phase_seconds=duration,
        observation_duration_seconds=duration,
        observation_start_ns=window['start_ns'],observation_end_ns=window['end_ns'],
        observation_duration_source='c14/phase_manifest.json profiling.end_ns minus profiling.start_ns, integer ns before conversion',
        benchmark_duration_seconds=phase['benchmark_duration']['avg'],
        throughput_denominator='observation_duration_seconds; exported benchmark_duration is a distinct metric',
        output_token_throughput=throughput,derived_meets_slo=ttft<=10000 and tpot<=40,
        parser_has_meets_slo='meets_slo' in summary,entrypoint_meets_slo=identity['meets_slo'],
        original_payload_verified=True,service_reaped=True,performance_json=raw_excerpt,sources=sources,
        scope='Independent e919 normal-source rebuild using newly fitted478 caches. Actual original137 payloads,128 measurements, phase throughput and normal service reap verified from frozen raw records. Installed dependencies and public base remain reused. No quality result inferred.')
    (OUT/'fresh-performance-audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    return result


if __name__ == '__main__':
    result = audit()
    print(json.dumps({k:result[k] for k in ['status','ttft_p95_ms','tpot_p95_ms','output_token_throughput','derived_meets_slo']},ensure_ascii=False))
