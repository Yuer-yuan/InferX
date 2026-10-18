#!/usr/bin/env python3
"""Audit fresh 116 quality raw archive, cohort, scores and termination without inference."""
import argparse
import hashlib
import json
import math
import tarfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
BINARY = 'e919f18084f3cc09718158cfae071cdf8c58d5660715648db375643016c878f5'
RECIPE = '4b7d1b7542291948bbec1af23fe99973189d48d6e171bcabc4a40e0b6c123a59'
SOURCE = 'c976692d9657408308193b50772c700382a6e243d9da353165205dc5252f76a8'
METRICS = ['prompt_level_strict','inst_level_strict','prompt_level_loose','inst_level_loose']


def audit(base):
    sources = {}

    def read(path, lines=False):
        raw = path.read_bytes()
        sources[str(path.relative_to(ROOT))] = {
            'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw)}
        return [json.loads(line) for line in raw.splitlines() if line] if lines else json.loads(raw)

    def compact(value):
        if isinstance(value,dict):
            return {k:compact(v) for k,v in value.items() if v is not None}
        if isinstance(value,list):
            return [compact(v) for v in value]
        return value

    archive = Path(str(base)+'.tar.gz')
    payload = archive.read_bytes()
    assert len(payload) == 1480012
    assert hashlib.sha256(payload).hexdigest() == '005a5d4f00454bcb758ad28487e089772e19b5dfbd943720074fd16b7aba4d41'
    sources[str(archive.relative_to(ROOT))] = dict(sha256=hashlib.sha256(payload).hexdigest(),bytes=len(payload))
    archive_files = []
    with tarfile.open(archive) as handle:
        for member in handle.getmembers():
            assert member.isfile(), member.name
            path = base/member.name
            assert path.resolve().is_relative_to(base.resolve())
            raw = path.read_bytes()
            assert raw == handle.extractfile(member).read(), member.name
            sources[str(path.relative_to(ROOT))] = dict(sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw))
            archive_files.append(member.name)
    assert len(archive_files) == len(set(archive_files)) == 47
    terminal = read(base/'actual-terminal-and-GPU-116.json')
    assert not terminal['gpu_compute_processes'].strip()
    assert not terminal['service_port_listeners']
    assert all(not row['exists'] for row in terminal['owned_processes'])
    for receipt in [terminal['controller_terminal'],terminal['CPU_score_terminal']]+[r['terminal'] for r in terminal['service_terminals']]:
        assert receipt['reaped'] and receipt['returncode'] == 0
    original_verifier = read(base/'quality-independent-CPU-001/summary.json')
    assert original_verifier['original_verifier_recomputed_all100']
    assert original_verifier['all_native_scores_exact_match']
    assert original_verifier['original_first100_metadata_all_match']
    assert original_verifier['binary_sha256'] == BINARY
    quality = base/'deployment116/quality100'
    summary = read(quality/'evaluation/summary.json')
    native_summary = read(quality/'evaluation/accuracy/summary.json')
    raw_score = summary['accuracy']
    assert raw_score == native_summary
    assert raw_score['status'] == '完整' and not raw_score['issues']
    launch = read(base/'deployment116/service-quality100/launch.json')
    identity = read(quality/'evaluation-identity.json')
    config = read(quality/'evaluation/accuracy/task_config.json')
    readiness = read(quality/'model-readiness.json')
    assert launch['binary_sha256'] == BINARY and launch['source_sha256'] == SOURCE
    assert launch['profile'] == dict(client_concurrency=13,native_concurrency=13,
                                     kv_capacity=532480,mode='accuracy',name='quality100')
    assert identity['status'] == 'complete' and identity['exit_code'] == 0
    assert identity['recipe_sha256'] == launch['recipe_sha256'] == RECIPE
    assert original_verifier['recipe_sha256'] == RECIPE
    assert config['limit'] == 100 and config['generation_config'] == {'max_tokens':32768}
    assert config['eval_batch_size'] == 13
    assert readiness['ready'] and readiness['generation_requests_sent'] == 0
    paths = {}
    for kind in ['predictions','reviews']:
        matches = list((quality/'evaluation/accuracy/native'/kind).rglob('*.jsonl'))
        assert len(matches) == 1, matches
        paths[kind] = matches[0]
    predictions = read(paths['predictions'],True)
    reviews = read(paths['reviews'],True)
    assert len(predictions) == len(reviews) == 100
    by_pred = {r['index']:r for r in predictions}
    by_review = {r['index']:r for r in reviews}
    assert set(by_pred) == set(by_review) == set(range(100))
    historical = read(ROOT/'submission/track-a/data/selected-quality-sources.json')
    cohort = read(ROOT/historical['base']/'first100-frozen-cohort.jsonl',True)
    # Verify the exact first-100 corpus independently of the new run's outputs.
    assert len(cohort) == 100
    totals = {k:0 for k in METRICS}
    empty = errors = instructions = 0
    samples = []
    independent_rows = read(base/'quality-independent-CPU-001/all100.jsonl',True)
    assert len(independent_rows) == 100
    independent_by_index = {r['ordinal']:r for r in independent_rows}
    empty_indices = []
    length_indices = []
    for ordinal in range(100):
        prediction, review = by_pred[ordinal], by_review[ordinal]
        original = cohort[ordinal]['corpus']
        for key in ['key','prompt','instruction_id_list','kwargs']:
            assert compact(prediction['metadata'][key]) == compact(original[key])
            assert compact(review['sample_score']['sample_metadata'][key]) == compact(original[key])
        output = prediction['model_output']
        errors += bool(prediction.get('error') or not output or output.get('error'))
        assert len(output['choices']) == 1
        content = output['choices'][0]['message']['content']
        final = content if isinstance(content,str) else '\n'.join(c.get('text','') for c in content if c.get('type') == 'text')
        score = review['sample_score']['score']
        assert score['prediction'] == final
        n = len(original['instruction_id_list'])
        instructions += n
        empty += not final.strip()
        if not final.strip(): empty_indices.append(ordinal)
        if output['choices'][0]['stop_reason'] == 'max_tokens': length_indices.append(ordinal)
        independent = independent_by_index[ordinal]
        assert independent['key'] == str(original['key'])
        assert independent['scores'] == score['value']
        assert independent['final_sha256'] == hashlib.sha256(final.encode()).hexdigest()
        assert independent['stop_reason'] == output['choices'][0]['stop_reason']
        for metric in METRICS:
            totals[metric] += float(score['value'][metric]) * (n if metric.startswith('inst_') else 1)
        samples.append(dict(index=ordinal,key=original['key'],instructions=n,
                            empty_final=not final.strip(),scores=score['value']))
    assert instructions == 111 and errors == 0
    aggregates = {}
    for metric in METRICS:
        denominator = instructions if metric.startswith('inst_') else 100
        numerator = round(totals[metric])
        assert math.isclose(numerator,totals[metric],abs_tol=1e-9)
        assert raw_score['metrics'][metric]['num'] == denominator
        # EvalScope's original report rounds these aggregate values to four
        # decimal places. Derive exact counts from all per-prompt reviews.
        assert round(numerator/denominator,4) == raw_score['metrics'][metric]['score']
        aggregates[metric] = dict(numerator=numerator,denominator=denominator,
                                  score=numerator/denominator,
                                  reported_score=raw_score['metrics'][metric]['score'])
        assert original_verifier['metrics'][metric]['numerator'] == numerator
        assert original_verifier['metrics'][metric]['denominator'] == denominator
    native_path = base/'deployment116/service-quality100/requests.jsonl'
    native = read(native_path,True)
    for name,path in [('predictions',paths['predictions']),('reviews',paths['reviews']),('native',native_path)]:
        assert hashlib.sha256(path.read_bytes()).hexdigest() == original_verifier['raw_identity'][name]['sha256']
    starts = [r for r in native if r['event'] == 'request_start']
    done = [r for r in native if r['event'] == 'request_done']
    assert len(starts) == len(done) == raw_score['generated_samples'] == 100
    assert not any(r['event'] == 'request_error' for r in native)
    assert len({r['request']['request_id'] for r in done}) == 100
    assert all(r['request']['enable_thinking'] and r['request']['requested_output_tokens'] == 32768 for r in starts)
    assert all(r['result']['computed_prefill_tokens'] == r['result']['prompt_tokens']
               and r['result']['prefix_cache_hit_tokens'] == 0 for r in done)
    finish = dict(Counter(r['result']['finish_reason'] for r in done))
    assert original_verifier['empty_final'] == empty
    assert original_verifier['native_finish_reason_counts'] == finish == {'stop_token':99,'output_limit':1}
    assert empty_indices == length_indices == [50]
    assert independent_by_index[50]['reasoning_nonempty']
    assert independent_by_index[50]['usage']['output_tokens'] == 32768
    expected = {'prompt_level_strict':85,'prompt_level_loose':88,'inst_level_strict':95,'inst_level_loose':99}
    assert {k:v['numerator'] for k,v in aggregates.items()} == expected
    excerpt = {'accuracy':{k:raw_score[k] for k in ['status','generated_samples']}}
    excerpt['accuracy']['metrics'] = {k:raw_score['metrics'][k] for k in METRICS}
    raw = '{\n  "accuracy": {\n'
    raw += '    "status": '+json.dumps(raw_score['status'],ensure_ascii=False)+',\n'
    raw += '    "generated_samples": '+str(raw_score['generated_samples'])+',\n'
    raw += '    "metrics": {\n'+',\n'.join('      '+json.dumps(k)+': '+json.dumps(excerpt['accuracy']['metrics'][k]) for k in METRICS)+'\n    }\n  }\n}'
    assert json.loads(raw) == excerpt
    result = dict(status='PASS',scope='CPU aggregation and input/output cross-check. Original verifier rerun is a separate receipt; no inference executed here.',
        binary_sha256=BINARY,source_sha256=SOURCE,recipe_sha256=RECIPE,samples=100,instructions=instructions,
        metrics=aggregates,empty_final=empty,errors=errors,length_stops=finish.get('output_limit',0),
        native_finish_reasons=finish,original_first100_verified=True,service_reaped=True,
        archive_files_verified=len(archive_files),empty_final_ordinals=empty_indices,length_stop_ordinals=length_indices,quality_json=raw,sources=sources)
    (OUT/'fresh-quality-audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    return result


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('delivery',type=Path)
    result=audit(parser.parse_args().delivery.resolve())
    print(json.dumps({k:result[k] for k in ['status','samples','metrics','empty_final','length_stops']},ensure_ascii=False))
