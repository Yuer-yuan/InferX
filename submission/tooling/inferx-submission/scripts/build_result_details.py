#!/usr/bin/env python3
"""CPU-only export of concurrency curves and original IFBench case evidence.

Source selection: submission/track-a/data/result-details-sources.json.
Recomputes request P95s, checks identical payload/configuration, and binds final
answers to the original prediction, review, and per-sample score hashes.
"""
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
DATA = ROOT / 'submission/track-a/data'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def write_changed(path, value):
    text = json.dumps(value, ensure_ascii=False, indent=2) + '\n'
    if not path.exists() or path.read_text() != text:
        path.write_text(text)


def p95(values):
    values = sorted(values)
    x = .95 * (len(values) - 1)
    lo, hi = math.floor(x), math.ceil(x)
    return values[lo] + (values[hi] - values[lo]) * (x - lo)


def build_result_details():
    sources = {}

    def read(rel, jsonl=False, digest_only=False):
        path = ROOT / rel
        raw = path.read_bytes()
        sources[str(path.relative_to(ROOT))] = {'sha256': sha(raw), 'bytes': len(raw)}
        if digest_only:
            return sha(raw)
        return [json.loads(line) for line in raw.splitlines() if line] if jsonl else json.loads(raw)

    registry = read('submission/track-a/data/result-details-sources.json')
    rows, comparable = [], []
    for entry in registry['performance']:
        c = entry['c']
        base = entry['raw_dir']
        run = f'{base}/candidate/c{c}'
        raw = read(f'{run}/summary.json')
        analysis = read(entry['analysis'])
        identity = read(f'{base}/candidate-identity.json')
        config = read(f'{base}/config.json')
        environment = read(entry['environment'])
        command = read(f'{run}/command.json')
        payload_sha = read(f'{run}/inputs.json', digest_only=True)
        dataset_sha = read(f'{base}/dataset.jsonl', digest_only=True)
        records = read(f'{run}/profile_export.jsonl', jsonl=True)
        formal = [r for r in records if r['metadata']['benchmark_phase'] == 'profiling']
        assert len(formal) == raw['valid_records'] == raw['records'] == config['requests'] == 128
        assert raw['status'] == 'complete' and raw['error_requests'] == analysis['errors'] == 0
        assert raw['lengths_verified'] and raw['osl_verified']
        assert analysis['binary'] == identity['binary_sha256']
        assert analysis['source'] == identity['source_sha256']
        assert analysis['C'] == analysis['native_cap'] == raw['concurrency'] == c
        assert analysis['all137cold'] and analysis['payload_same_as_parent_including_salt']
        native = config['cases'][0]['command']
        flag = lambda name: native[native.index(name) + 1]
        assert int(flag('--max-concurrency')) == c
        kv = int(flag('--kv-capacity'))
        assert kv == c * 8704, 'KV budget must be disclosed as scaling with capacity'
        assert config['warmup'] == analysis['warmup'] == 8
        for r in records:
            m = r['metrics']
            assert m['input_sequence_length']['value'] == 8192
            assert m['output_sequence_length']['value'] == 512
            assert m['usage_prompt_cache_read_tokens']['value'] == 0
        ttft = p95([r['metrics']['time_to_first_token']['value'] for r in formal])
        tpot = p95([(r['metrics']['request_latency']['value'] - r['metrics']['time_to_first_token']['value']) / 511 for r in formal])
        assert math.isclose(ttft, raw['metrics']['time_to_first_token']['p95'], abs_tol=1e-8)
        assert math.isclose(tpot, raw['metrics']['inter_token_latency']['p95'], abs_tol=1e-8)
        assert math.isclose(ttft, analysis['metrics']['TTFT_p95_ms'], abs_tol=1e-8)
        assert math.isclose(tpot, analysis['metrics']['TPOT_p95_ms'], abs_tol=1e-8)
        tps = raw['output_token_throughput']  # Original phase denominator, not wall-clock duration.
        assert tps == analysis['metrics']['output_tokens_per_second']
        joint = ttft <= 10000 and tpot <= 40
        assert joint == raw['meets_slo']
        # Normalize only capacity, KV size and run-specific log/port paths.
        normalized = native.copy()
        for name in ['--max-concurrency', '--kv-capacity', '--port', '--request-log-jsonl']:
            normalized[normalized.index(name) + 1] = '<varies>'
        extra = json.loads(command[command.index('--extra-inputs') + 1])
        comparable.append((entry['node'], identity, environment, normalized, payload_sha,
                           dataset_sha, extra, config['seed'], config['warmup']))
        rows.append({'id': entry['id'], 'node': entry['node'], 'c': c, 'kv': kv,
                     'binary_sha256': identity['binary_sha256'], 'source_sha256': identity['source_sha256'],
                     'n': len(formal), 'warmup': config['warmup'], 'k': int(flag('--draft-tokens')),
                     'ttft_s': ttft / 1000, 'tpot_ms': tpot, 'tps': tps, 'joint_slo': joint,
                     'payload_sha256': payload_sha, 'dataset_sha256': dataset_sha})
    assert all(item == comparable[0] for item in comparable), 'Curve points are not comparable'

    q = registry['quality']
    base = q['content_dir']
    cohort = read(f'{base}/first100-frozen-cohort.jsonl', jsonl=True)
    predictions = read(f'{base}/fixed-checkpoints/prefix100/predictions-original-lines.jsonl', jsonl=True)
    reviews = read(f'{base}/fixed-checkpoints/prefix100/reviews-original-lines.jsonl', jsonl=True)
    scores = read(f'{base}/CPU-score-001/per-sample.jsonl', jsonl=True)
    summary = read(f'{base}/CPU-score-001/summary.json')
    runtime = read(f'{base}/expected-runtime.json')
    assert len(cohort) == len(predictions) == len(reviews) == len(scores) == summary['samples'] == 100
    assert summary['original_verifier_recomputed_all100'] and summary['all_native_score_values_match']
    assert sum(s['scores']['prompt_level_strict'] for s in scores) == 84
    cases = []
    for key in q['selected_keys']:
        sample = next(s for s in scores if s['key'] == key)
        original = next(s for s in cohort if s['corpus']['key'] == key)['corpus']
        pred = next(s for s in predictions if s['index'] == sample['ordinal'])
        review = next(s for s in reviews if s['index'] == sample['ordinal'])['sample_score']
        content = pred['model_output']['choices'][0]['message']['content']
        final = ''.join(s['text'] for s in content if s['type'] == 'text')
        assert sha(final.encode()) == sample['final_sha256']
        assert final == review['score']['prediction'] == review['score']['extracted_prediction']
        assert sample['scores'] == review['score']['value']
        assert original['prompt_sha256'] == sample['prompt_sha256'] == sha(original['prompt'].encode())
        assert not sample['truncated'] and sample['finish_reason'] == 'stop'
        cases.append({'key': key, 'prompt': original['prompt'], 'final': final,
                      'instruction_ids': original['instruction_id_list'], 'kwargs': original['kwargs'],
                      'scores': sample['scores'], 'final_sha256': sample['final_sha256'],
                      'prompt_sha256': sample['prompt_sha256'], 'finish_reason': sample['finish_reason'],
                      'truncated': sample['truncated']})
    # Additional visible checks explain these selected examples; original verifier scores remain authoritative.
    success = next(s for s in cases if s['key'] == '80')
    lines = success['final'].splitlines()
    assert [len(s) - len(s.lstrip()) for s in lines] == [0, 2, 4]
    assert [len(s.strip()) for s in lines] == [32, 32, 32]
    words = [w.lower().strip('.,!?') for s in lines for w in s.split()]
    assert len(words) == len(set(words)) == 18
    failed = next(s for s in cases if s['key'] == '83')
    assert failed['kwargs'][0]['sep'] == '!?!?'
    assert all(s.startswith('!?? ') for s in failed['final'].splitlines() if s)
    assert success['scores']['prompt_level_strict'] == 1 and failed['scores']['prompt_level_strict'] == 0
    # Exact first line, with no inserted ellipsis or rewritten model text.
    failed['excerpt'] = failed['final'].splitlines()[0]
    result = {'schema_version': 1, 'performance': rows,
              'comparison_scope': 'Same ucloud1 node, binary, environment, model and actual payload; one run per point. Native concurrency and KV budget increase together. Not an isolated concurrency effect or stable-capacity estimate.',
              'quality': {'binary_sha256': runtime['binary_sha256'], 'samples': 100,
                          'metrics': summary['metrics'], 'cases': cases}, 'sources': sources}
    write_changed(DATA / 'result-details.json', result)
    return result


if __name__ == '__main__':
    result = build_result_details()
    print(f"Verified {len(result['performance'])} concurrency points and {len(result['quality']['cases'])} original quality cases; {len(result['sources'])} sources.")
