#!/usr/bin/env python3
"""Verify fresh239 fitting receipts and compare all478 recorded payload hashes."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4]
STAGE=Path(__file__).resolve().parent
BASE=ROOT/'results/three-node-optimization/ucloud2/full-quantization-reproduction-116'
LIGHT=BASE/'fresh239-fit-light-evidence-001'
SOURCES={}
def read(path):
    raw=path.read_bytes();SOURCES[str(path.relative_to(ROOT))]=dict(sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw));return json.loads(raw)
def main():
    compare=read(BASE/'fresh-vs-original478-SHA.json');reference=BASE/'entrypoint-template/candidate_recipes/assets-u3-47.json';old=read(reference)
    assert SOURCES[str(reference.relative_to(ROOT))]['sha256']==compare['reference_sha256']=='7f1f5dbd1b6b288f165852772ed0c8090cc8f5abc0a73265a14f3b604d0b911a'
    summary=read(LIGHT/'full-fresh-fit-evidence.json');assert summary['newly_fitted_serving_matrices']==239 and summary['newly_fitted_payload_files']==478 and not summary['task_accuracy_inferred']
    rows=[];matrix_total=0
    for cache,stage,expected in [('weights025','true-int4-all-layer-GPTQ-025',128),('weights041','true-int4-all-input-GPTQ-041',64),('weights047','true-int4-all-output-GPTQ-047',47)]:
        path=LIGHT/stage/'fresh-serving-manifest.json';fresh=read(path)
        assert fresh['newly_fitted'] and fresh['matrices']==expected
        frozen=next(x for x in summary['manifests'] if f'/{stage}/' in x['path']);assert SOURCES[str(path.relative_to(ROOT))]['sha256']==frozen['sha256']
        entries={x['relative_path']:{'sha256':x['sha256'],'bytes':x['bytes']} for x in fresh['files']}
        assert len(entries)==expected*2 and entries==old['caches'][cache]['files']
        assert len(fresh['numerical'])==expected
        for item in fresh['numerical']:
            assert item['CPU_sample_code_differences']==0
            folder=LIGHT/stage/f"weights-L{item['layer']}-{item['kind']}"
            fit=read(folder/'fit-frozen-before-heldout.json')
            assert SOURCES[str((folder/'fit-frozen-before-heldout.json').relative_to(ROOT))]['sha256']==item['fit_manifest_sha256']
            expected_fit_kind=item['kind'] if cache=='weights025' else ('attention' if item['layer']%4==3 else 'gdn')
            assert fit['layer']==item['layer'] and fit['kind']==expected_fit_kind and not fit['heldout_values_read']
            equiv=read(folder/'fit-equivalence.json');assert equiv['CPU_sample_code_differences']==0
        rows.append(dict(cache=cache,matrices=expected,files=len(entries),identical_sha_and_bytes=len(entries),differences=0));matrix_total+=expected
    recipe_path=BASE/'deployment116/entrypoint/candidate_recipes/116-fresh47.json';recipe=read(recipe_path)
    assert recipe['binary_sha256']=='e919f18084f3cc09718158cfae071cdf8c58d5660715648db375643016c878f5'
    assert recipe['source_sha256']=='c976692d9657408308193b50772c700382a6e243d9da353165205dc5252f76a8'
    assert recipe['assets']['contract']['sha256']==compare['reference_sha256']
    assert compare['complete'] and not compare['pending'] and compare['identical_files']==478 and compare['different_files']==0
    result=dict(status='PASS',scope='CPU audit of frozen fresh fitting/serving manifests and all239 per-matrix receipts. Compares all478 recorded SHA/size pairs with the old selected contract; does not reread large payloads or run GPU checks. No performance or quality result inferred.',matrices=matrix_total,files=478,rows=rows,binary_sha256=recipe['binary_sha256'],source_sha256=recipe['source_sha256'],scores_inherited=False,sources=SOURCES)
    (STAGE/'fresh478-completion-audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print('PASS:239 fresh matrices,478 manifest SHA/size matches,0 differences; per-matrix fitting receipts verified; newe919 recipe; no inferred scores.')
if __name__=='__main__':main()
