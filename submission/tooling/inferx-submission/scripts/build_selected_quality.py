#!/usr/bin/env python3
"""CPU-only import of complete original-100 quality evaluation 105.

Pin the worker's entire raw archive manifest, recompute score aggregates from
all original rows, and bind displayed answers to raw HTTP/prediction/review.
This imports the original verifier's results; it does not launch inference.
"""
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
REGISTRY = ROOT / 'submission/track-a/data/selected-quality-sources.json'
OUTPUT = ROOT / 'submission/track-a/data/selected-quality.json'


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def audit():
    reg = json.loads(REGISTRY.read_text())
    base = ROOT / reg['base']
    sources = {str(REGISTRY.relative_to(ROOT)): {'sha256': sha(REGISTRY.read_bytes()), 'bytes': REGISTRY.stat().st_size}}
    manifest_path = ROOT / reg['manifest']['path']
    manifest_raw = manifest_path.read_bytes()
    assert sha(manifest_raw) == reg['manifest']['sha256']
    manifest = json.loads(manifest_raw)
    sources[reg['manifest']['path']] = {'sha256': sha(manifest_raw), 'bytes': len(manifest_raw)}
    for rel, item in manifest['files'].items():
        path = base / rel
        assert path.resolve().is_relative_to(base.resolve())
        raw = path.read_bytes()
        assert len(raw) == item['bytes'] and sha(raw) == item['sha256'], rel
        sources[str(path.relative_to(ROOT))] = item
    assert len(manifest['files']) == manifest['file_count'] == 583

    def read(rel, lines=False):
        assert rel in manifest['files'], rel
        raw = (base / rel).read_text()
        return [json.loads(l) for l in raw.splitlines() if l] if lines else json.loads(raw)

    summary = read('CPU-score-001/summary.json')
    scores = read('CPU-score-001/per-sample.jsonl', True)
    cohort = read('first100-frozen-cohort.jsonl', True)
    predictions = read('accuracy/native/predictions/qwen38-27b/ifbench_default.jsonl', True)
    reviews = read('accuracy/native/reviews/qwen38-27b/ifbench_default.jsonl', True)
    attempts = read('CPU-score-001/all-http-attempts.jsonl', True)
    joined = read('CPU-http-native-join-001/per-completion.jsonl', True)
    startup = read('service/actual-startup.json')
    environment = read('service/environment.json')
    expected = read('expected-runtime.json')
    qualification = read('qualification-evidence.json')
    config = read('resolved_config.json')
    native = read('service/native.jsonl', True)
    assert expected['binary_sha256'] == qualification['binary']['sha256'] == reg['binary_sha256']
    assert expected['source_aggregate_sha256'] == qualification['source_aggregate'] == reg['source_sha256']
    assert config['eval_batch_size'] == config['generation_config']['batch_size'] == 13
    assert config['limit'] == 100 and config['generation_config']['max_tokens'] == 32768
    assert startup['engine']['max_concurrency'] == 13 and startup['engine']['kv_capacity'] == 532480
    assert startup['engine']['speculative_draft_window'] == 3 and not startup['engine']['prefix_reuse']
    assert environment['NINFER_SCHED_BOUNDED_DECODE_MEMBERSHIP'] == '1'
    assert environment['NINFER_PROMPT_LOOKUP_N5_PRIORITY'] == environment['NINFER_PROMPT_LOOKUP_N5_HYBRID'] == '0'
    assert summary['original_verifier_recomputed_all100'] and summary['all_native_score_values_match']
    for name, path in [('predictions', 'accuracy/native/predictions/qwen38-27b/ifbench_default.jsonl'),
                       ('reviews', 'accuracy/native/reviews/qwen38-27b/ifbench_default.jsonl')]:
        assert summary[name]['sha256'] == manifest['files'][path]['sha256']
    assert len(scores) == len(cohort) == len(predictions) == len(reviews) == len(attempts) == len(joined) == 100
    assert {s['ordinal'] for s in scores} == {p['index'] for p in predictions} == set(range(100))
    assert len({a['ordinal'] for a in attempts}) == len({j['key'] for j in joined}) == 100
    by_pred = {p['index']: p for p in predictions}
    by_review = {r['index']: r['sample_score']['score'] for r in reviews}
    by_cohort = {c['ordinal']: c['corpus'] for c in cohort}
    by_attempt = {a['ordinal']: a for a in attempts}
    by_join = {j['ordinal']: j for j in joined}
    by_native = {r['request']['request_id']: r for r in native if r['event'] == 'request_done'}
    cases = []
    counts = {name: 0 for name in ['prompt_strict', 'prompt_loose', 'instruction_strict', 'instruction_loose']}
    for sample in scores:
        i = sample['ordinal']
        original, pred, review = by_cohort[i], by_pred[i], by_review[i]
        attempt, join = by_attempt[i], by_join[i]
        assert sample['key'] == original['key'] == join['key']
        assert sha(original['prompt'].encode()) == original['prompt_sha256'] == sample['prompt_sha256']
        assert sample['instructions'] == len(original['instruction_id_list'])
        assert sample['status'] == 'complete' and sample['error'] is None
        assert sample['attempts'] == 1 and attempt['status_code'] == 200
        attempt_base = f"http-attempts/{attempt['attempt']:05d}"
        payload = read(attempt_base + '/payload.json')
        response = read(attempt_base + '/response.json')
        assert set(payload) == {'messages', 'model', 'max_tokens'}
        assert payload['messages'] == [{'role': 'user', 'content': original['prompt']}]
        assert payload['max_tokens'] == 32768
        assert sha((base / (attempt_base + '/request.raw.bin')).read_bytes()) == attempt['request_body_sha256']
        assert sha((base / (attempt_base + '/response.raw.bin')).read_bytes()) == attempt['response_sha256']
        assert response['id'] == sample['native_response_id'] == join['response_id']
        final = ''.join(c['text'] for c in pred['model_output']['choices'][0]['message']['content'] if c['type'] == 'text')
        assert final == review['prediction'] == review['extracted_prediction'] == response['choices'][0]['message']['content']
        assert sha(final.encode()) == sample['final_sha256'] == join['content_sha256']
        assert sample['empty_final'] == (not final.strip())
        assert sample['scores'] == review['value']
        assert sample['finish_reason'] == response['choices'][0]['finish_reason'] == 'stop'
        assert not sample['truncated']
        assert join['native'] == by_native[join['native_request_id']]
        request, result = join['native']['request'], join['native']['result']
        assert request['enable_thinking'] and request['requested_output_tokens'] == 32768
        assert request['sampling']['temperature'] == 1 and request['sampling']['top_k'] == 20
        assert math.isclose(request['sampling']['top_p'], .95, abs_tol=1e-7)
        assert result['prefix_cache_hit_tokens'] == 0
        assert join['native']['speculative']['draft_window'] == 3
        for suffix in ['strict', 'loose']:
            counts['prompt_' + suffix] += sample['scores']['prompt_level_' + suffix]
            counts['instruction_' + suffix] += sample['instructions'] * sample['scores']['inst_level_' + suffix]
        if sample['key'] in reg['selected_keys']:
            cases.append({'key': sample['key'], 'prompt': original['prompt'], 'final': final,
                          'instruction_ids': original['instruction_id_list'], 'kwargs': original['kwargs'],
                          'scores': sample['scores'], 'final_sha256': sample['final_sha256'],
                          'prompt_sha256': sample['prompt_sha256'], 'finish_reason': sample['finish_reason'],
                          'truncated': sample['truncated']})
    assert sum(s['instructions'] for s in scores) == summary['instructions'] == 111
    for key, value in counts.items():
        assert value == reg['expected'][key]
        metric = key.replace('instruction_', 'inst_level_').replace('prompt_', 'prompt_level_')
        assert math.isclose(value / (111 if key.startswith('instruction') else 100), summary['metrics'][metric]['score'])
    for key in ['samples', 'instructions', 'attempts', 'failures', 'empty_final', 'length']:
        assert summary[key] == reg['expected'][key]
    assert sum(s['empty_final'] for s in scores) == 1
    assert summary['prompts_with_multiple_attempts'] == 0
    assert Counter(s['finish_reason'] for s in scores) == summary['finish_reason_counts'] == {'stop': 100}
    success, failure = [next(c for c in cases if c['key'] == key) for key in ['80', '88']]
    lines = success['final'].splitlines()
    assert [len(s) - len(s.lstrip()) for s in lines] == [0, 4, 8]
    assert [len(s.strip()) for s in lines] == [28, 28, 28]
    words = [w.lower().strip('.,!?') for s in lines for w in s.split()]
    assert len(words) == len(set(words)) == 15
    assert failure['kwargs'][0]['sep'] == '!?!?'
    assert all(s.startswith('!?? ') for s in failure['final'].splitlines())
    assert failure['scores']['prompt_level_strict'] == failure['scores']['prompt_level_loose'] == 0
    assert failure['scores']['inst_level_strict'] == failure['scores']['inst_level_loose'] == .5
    result = {'schema_version': 1, 'campaign': reg['campaign'],
              'binary_sha256': reg['binary_sha256'], 'source_sha256': reg['source_sha256'],
              'samples': 100, 'instructions': 111, 'counts': counts, 'metrics': summary['metrics'],
              'failures': 0, 'empty_final': 1, 'length': 0, 'attempts': 100,
              'finish_reason_counts': summary['finish_reason_counts'], 'cases': cases,
              'configuration': {'client_c': 13, 'native_c': 13, 'kv': 532480, 'k': 3, 'max_tokens': 32768,
                                'actual_temperature': 1, 'actual_top_k': 20, 'actual_top_p': .95,
                                'request_generation_override': 'max_tokens only; server seed42'},
              'scope': summary['qualification'], 'bootstrap': summary['bootstrap'],
              'summary_source': str((base / 'CPU-score-001/summary.json').relative_to(ROOT)), 'sources': sources,
              'validation': 'All 583 archive members hashed; all 100 original prompt/prediction/review/HTTP records cross-checked; original verifier scores aggregated; one empty final retained. Native joins match worker-provided exact usage/timing associations, not a shared exported request ID. No new GPU run.'}
    text = json.dumps(result, ensure_ascii=False, indent=2) + '\n'
    if not OUTPUT.exists() or OUTPUT.read_text() != text:
        OUTPUT.write_text(text)
    return result


if __name__ == '__main__':
    result = audit()
    print(f"Verified original {result['samples']} prompts / {result['instructions']} instructions; {len(result['sources'])} source hashes; counts={result['counts']}; empty={result['empty_final']}.")
