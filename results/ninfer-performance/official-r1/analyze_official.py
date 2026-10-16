"""Offline audit of the original judge and one-second native GPU sampling.

Place beside extracted runs/official-r1 and runs/official-r1-service, then run.
Only reads frozen evidence and writes analysis.json. No inference or retries.
"""
from pathlib import Path
from collections import Counter
import csv
import datetime as dt
import gzip
import hashlib
import importlib.util
import io
import json
from statistics import median
import sys


ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[2]
HASHES = {}


def read(path):
    stored = path if path.exists() else path.with_name(path.name + '.gz')
    raw = stored.read_bytes()
    HASHES[str(stored.relative_to(ROOT))] = hashlib.sha256(raw).hexdigest()
    return (gzip.decompress(raw) if stored.suffix == '.gz' else raw).decode()


def jload(path):
    return json.loads(read(path))


def jsonlines(path):
    return [json.loads(line) for line in read(path).splitlines() if line.strip()]


def samples_summary(samples):
    result = {'samples': len(samples), 'selection': 'All recorded samples in the specified interval, no utilization filter. Means weight recorded points equally; coverage is reported separately.'}
    for key in ('utilization.gpu', 'utilization.memory', 'memory.used', 'power.draw', 'temperature.gpu', 'clocks.sm', 'clocks.mem'):
        values = []
        for s in samples:
            try:
                values.append(float(s[key]))
            except ValueError:
                pass
        result[key] = {'available_samples': len(values), 'missing_samples': len(samples) - len(values),
                       **({'min': min(values), 'mean': sum(values) / len(values), 'max': max(values)} if values else {})}
    for key in ('clocks_event_reasons.sw_power_cap', 'clocks_event_reasons.sw_thermal_slowdown', 'clocks_event_reasons.hw_thermal_slowdown'):
        result[key] = dict(Counter(s[key] for s in samples))
    return result


def coverage(samples, begin_ms, end_ms):
    """Reject silently truncated/sparse telemetry; preserve observed cadence."""
    points = [s['unix_ms'] for s in samples if begin_ms <= s['unix_ms'] <= end_ms]
    assert len(points) >= 2, 'Missing GPU sampling coverage'
    gaps = [b - a for a, b in zip(points, points[1:])]
    leading, trailing = points[0] - begin_ms, end_ms - points[-1]
    ratio = len(points) / ((end_ms - begin_ms) / 1000)
    result = {'samples': len(points), 'span_s': (points[-1] - points[0]) / 1000,
              'leading_gap_s': leading / 1000, 'trailing_gap_s': trailing / 1000,
              'period_min_s': min(gaps) / 1000, 'period_mean_s': sum(gaps) / len(gaps) / 1000,
              'period_max_s': max(gaps) / 1000, 'count_over_target_1hz': ratio,
              'gaps_over_5s_unix_ms': [[a, b] for a, b in zip(points, points[1:]) if b - a > 5000],
              'validation_limits': 'Strictly increasing; max gap5s; leading/trailing<=2.5s; count/target1Hz between0.85 and1.15.'}
    assert min(gaps) > 0 and max(gaps) <= 5000, result
    assert leading <= 2500 and trailing <= 2500 and .85 <= ratio <= 1.15, result
    return result


def coverage_record(samples, begin_ms, end_ms):
    # Keep the strict acceptance criteria. A coverage failure must remain visible
    # without discarding independently valid request/score data or rerunning for
    # a better sample. These aggregates describe observed points, not full time.
    try:
        return {'status': 'complete', **coverage(samples, begin_ms, end_ms)}
    except AssertionError as error:
        if not error.args or not isinstance(error.args[0], dict):
            raise
        return {'status': 'partial', **error.args[0],
                'limitation': 'Failed strict target1Hz coverage validation; interpret GPU values only as statistics of recorded points.'}


if __name__ == '__main__':
    official = ROOT / 'runs/official-r1'
    service = ROOT / 'runs/official-r1-service'
    status = jload(service / 'operator-status.json')
    assert status['status'] == 'judge-exited-zero' and status['operator_exit_code'] == 0
    runner_path = PROJECT / 'submission/organizer/runner.py'
    runner_sha = hashlib.sha256(runner_path.read_bytes()).hexdigest()
    identity = jload(service / 'operator-process.json')
    assert runner_sha == identity['runner_sha256']
    spec = importlib.util.spec_from_file_location('inferx_original_judge_analysis', runner_path)
    runner = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = runner
    spec.loader.exec_module(runner)

    schema = jload(ROOT / 'gpu-schema.json')
    gpu = []
    for fields in csv.reader(io.StringIO(read(service / 'gpu.csv'))):
        assert len(fields) == len(schema['fields']), fields
        sample = dict(zip(schema['fields'], (s.strip() for s in fields)))
        sample['unix_ms'] = dt.datetime.strptime(sample['timestamp'], '%Y/%m/%d %H:%M:%S.%f').replace(tzinfo=dt.timezone.utc).timestamp() * 1000
        gpu.append(sample)
    assert gpu and abs(gpu[0]['unix_ms'] / 1000 - dt.datetime.fromisoformat(status['started_at']).timestamp()) < 5
    assert all(s['uuid'] == schema['gpu_uuid'] for s in gpu)
    # Validate original ordering too: interval filtering must not hide a clock
    # reversal or duplicate timestamps outside the selected phase windows.
    assert all(a['unix_ms'] < b['unix_ms'] for a, b in zip(gpu, gpu[1:]))
    whole_coverage = coverage_record(gpu, dt.datetime.fromisoformat(status['started_at']).timestamp() * 1000,
                              dt.datetime.fromisoformat(status['ended_at']).timestamp() * 1000)
    native = jsonlines(service / 'requests.jsonl')
    done = [r for r in native if r.get('event') == 'request_done']
    output = {'scope': 'One original full run, no selection among repetitions; warmups excluded from performance metrics.',
              'runner_sha256': runner_sha, 'operator': status,
              'gpu_sampling_limit': 'Target1Hz, actual coverage reported and may be partial. Observed point means/peaks, not continuous-time averages/peaks. Memory utilization is not achieved bandwidth. Clock flags are observations, not causal experiments.',
              'native_timing_limit': 'Decode counter includes complete shared-batch duration per request plus MTP tails. Residual is elapsed post-first wall outside these blocks; never sum decode counters as exclusive GPU time.',
              'performance': [], 'gpu_recorded_overall_samples': samples_summary(gpu),
              'gpu_whole_run_coverage': whole_coverage}
    windows = []
    for concurrency in (1, 2, 4, 8):
        folder = official / f'performance/c{concurrency}'
        parsed = runner.parse_performance(folder, 100, concurrency, warmup_expected=8, strict_isl=True)
        assert parsed['status'] == '完整' and not parsed['issues'], parsed
        # Also hash the authoritative inputs consumed directly by the parser.
        for name in ('profile_export_aiperf.json', 'profile_export.jsonl'):
            read(folder / name)
        clients = [r for r in jsonlines(folder / 'profile_export.jsonl') if r['metadata']['benchmark_phase'] == 'profiling']
        begin = min(r['metadata']['request_start_ns'] for r in clients) / 1e6
        end = max(r['metadata']['request_end_ns'] for r in clients) / 1e6
        windows.append((begin, end))
        server = [r for r in done if r['result']['completion_tokens'] == 512 and begin <= r['timestamp_unix_ms'] <= end + 100]
        assert len(server) == 100
        assert all(r['result']['prompt_tokens'] == 8192 and r['result']['prefix_cache_hit_tokens'] == 0 for r in server)
        times = [r['timings_seconds'] for r in server]
        metrics = parsed['metrics']
        prefill_sum = sum(t['prefill'] for t in times)
        row = {'concurrency': concurrency, 'valid_requests': parsed['valid_records'], 'warmup_requests': 8,
               'output_tok_s': parsed['output_token_throughput'],
               'benchmark_duration_s': parsed['benchmark_duration'],
               'ttft_p95_s': metrics['time_to_first_token']['p95'] / 1000,
               'tpot_p95_ms': metrics['inter_token_latency']['p95'],
               'e2e_p95_s': metrics['request_latency']['p95'] / 1000,
               'meets_slo': metrics['time_to_first_token']['p95'] <= 10000 and metrics['inter_token_latency']['p95'] <= 40,
               'profiling_window_unix_ms': [begin, end],
               'median_native_prefill_s': median(t['prefill'] for t in times),
               'median_native_decode_counter_s': median(t['decode'] for t in times),
               'median_native_residual_s': median(t['total'] - t['ttft'] - t['decode'] for t in times),
               'sum_own_prefill_s': prefill_sum,
               'sum_own_prefill_over_benchmark_duration': prefill_sum / parsed['benchmark_duration'],
               'prefill_fraction_limit': 'Host-timed synchronized prefill scheduling units on the serial worker, not kernel-only GPU occupancy or a bandwidth measurement.',
               'gpu_coverage': coverage_record(gpu, begin, end),
               'gpu': samples_summary([s for s in gpu if begin <= s['unix_ms'] <= end])}
        drafted = sum(r.get('speculative', {}).get('drafted_tokens', 0) for r in server)
        row['weighted_draft_acceptance'] = sum(r['speculative']['accepted_tokens'] for r in server) / drafted if drafted else None
        output['performance'].append(row)
    output['gpu_performance_profiling_union'] = samples_summary([s for s in gpu if any(a <= s['unix_ms'] <= b for a, b in windows)])
    accuracy = runner.parse_accuracy(official / 'accuracy')
    assert accuracy['status'] == '完整' and not accuracy['issues'], accuracy
    for path in sorted((official / 'accuracy').rglob('*')):
        if path.is_file() and path.suffix in {'.json', '.jsonl'}:
            read(path)
    output['accuracy'] = {**accuracy,
                           'native_report': str(Path(accuracy['native_report']).relative_to(ROOT)),
                           'artifacts': str(Path(accuracy['artifacts']).relative_to(ROOT))}
    chats = [r for r in done if r['request']['protocol'] == 'openai_chat_completions']
    assert len(chats) == 100
    output['accuracy_native'] = {'requests': len(chats),
                                 'finish_reasons': dict(Counter(r['result']['finish_reason'] for r in chats)),
                                 'completion_token_min': min(r['result']['completion_tokens'] for r in chats),
                                 'completion_token_max': max(r['result']['completion_tokens'] for r in chats),
                                 'completion_token_sum': sum(r['result']['completion_tokens'] for r in chats)}
    output['validation'] = {'original_benchmark_complete': True,
                             'gpu_coverage_complete': whole_coverage['status'] == 'complete' and all(r['gpu_coverage']['status'] == 'complete' for r in output['performance']),
                             'scope': 'Benchmark integrity and GPU sampling coverage are separate findings; analysis exit0 does not mean coverage is complete.'}
    output['source_sha256'] = dict(sorted(HASHES.items()))
    (ROOT / 'analysis.json').write_text(json.dumps(output, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'performance': [{k:v for k,v in row.items() if k in {'concurrency','output_tok_s','ttft_p95_s','tpot_p95_ms','meets_slo'}} for row in output['performance']], 'accuracy': accuracy, 'accuracy_native': output['accuracy_native'], 'validation': output['validation']}, ensure_ascii=False, indent=2))
