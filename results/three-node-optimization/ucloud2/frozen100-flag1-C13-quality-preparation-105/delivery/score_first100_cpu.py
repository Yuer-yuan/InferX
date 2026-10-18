"""CPU-only original IFBench verifier plus raw request/content/attempt audit."""
import collections
import copy
import hashlib
import importlib.metadata
import json
import os
import pathlib

import numpy as np
from evalscope.benchmarks.ifbench import evaluation_lib

R=pathlib.Path(__file__).resolve().parent
OUT=R/'CPU-score-001'
OUT.mkdir(exist_ok=False)
assert importlib.metadata.version('evalscope')=='1.13.0'
def load(p):return json.loads(p.read_text())
def rows(p):return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
def compact(v):
    if isinstance(v,dict):return {k:compact(a) for k,a in v.items() if a is not None}
    if isinstance(v,list):return [compact(a) for a in v]
    return v

cohort=rows(R/'first100-frozen-cohort.jsonl');assert len(cohort)==100
predfiles=list((R/'accuracy/native/predictions').rglob('*.jsonl'))
reviewfiles=list((R/'accuracy/native/reviews').rglob('*.jsonl*'))
assert len(predfiles)==len(reviewfiles)==1,(predfiles,reviewfiles)
preds=rows(predfiles[0]);reviews=rows(reviewfiles[0])
assert len(preds)==100 and {p['index'] for p in preds}==set(range(100)),'Incomplete original100 preserved; no full score'
preds={p['index']:p for p in preds};reviews={p['index']:p for p in reviews}
assert set(reviews)==set(range(100))
vr=pathlib.Path(evaluation_lib.__file__).parent
actual_verifiers={p.name:sha(p) for p in sorted(vr.glob('*.py'))}
expected=load(R.parent/'quality-reference-preparation/cpu-verifier-validation.json')['verifier_files']
assert actual_verifiers==expected
attempts=[];rawbyid={};by_prompt=collections.defaultdict(list)
lookup={json.dumps(r['messages'],ensure_ascii=False,sort_keys=True,separators=(',',':')):r['ordinal'] for r in cohort}
for p in sorted((R/'http-attempts').glob('*/attempt.json')):
    a=load(p);payloadp=p.parent/'payload.json'
    if not payloadp.exists():continue
    payload=load(payloadp)
    assert sorted(payload)==['max_tokens','messages','model'],payload.keys()
    assert payload['max_tokens']==32768 and payload['model']=='qwen38-27b'
    ordinal=lookup[json.dumps(payload['messages'],ensure_ascii=False,sort_keys=True,separators=(',',':'))]
    a.update(ordinal=ordinal,request_payload_sha256=sha(payloadp),path=str(p.parent))
    by_prompt[ordinal].append(a)
    responsep=p.parent/'response.json'
    if responsep.exists():
        resp=load(responsep)
        if resp.get('id') is not None:rawbyid[resp['id']]=(resp,a)
    attempts.append(a)
metrics=['prompt_level_strict','inst_level_strict','prompt_level_loose','inst_level_loose']
details=[]
for index,frozen in enumerate(cohort):
    p=preds[index];review=reviews[index];metadata=review['sample_score']['sample_metadata'];original=frozen['corpus']
    for field in ['key','prompt','instruction_id_list','kwargs']:assert compact(metadata[field])==compact(original[field]),(index,field)
    assert by_prompt[index],index
    model=p.get('model_output')
    if p.get('error') or model is None or not model.get('choices'):
        scores={m:0.0 for m in metrics};final='';reasoning='';finish=None;nativeid=None;usage={};response=None
        error=p.get('error') or 'missing native output';status='failed'
    else:
        choice=model['choices'][0];content=choice['message']['content']
        if isinstance(content,str):final,reasoning=content,''
        else:
            final='\n'.join(c.get('text','') for c in content if c.get('type')=='text')
            reasoning='\n'.join(c.get('reasoning','') for c in content if c.get('type')=='reasoning')
        assert review['sample_score']['score']['prediction']==final
        scores={k:float(v) for k,v in evaluation_lib.process_results(copy.deepcopy(original),[final]).items() if k in metrics}
        assert scores==review['sample_score']['score']['value'],(index,scores,review['sample_score']['score']['value'])
        nativeid=model['id'];response,attempt=rawbyid[nativeid]
        message=response['choices'][0]['message'];assert (message.get('content') or '')==final
        assert (message.get('reasoning_content') or message.get('reasoning') or '')==reasoning
        usage=response['usage']
        cached=(usage.get('prompt_tokens_details') or {}).get('cached_tokens')
        if cached is not None:assert cached==0
        assert usage['completion_tokens']<=32768
        finish=response['choices'][0].get('finish_reason');error=None;status='complete'
    details.append(dict(ordinal=index,key=original['key'],prompt_sha256=frozen['prompt_sha256'],record_sha256=frozen['record_sha256'],
        instructions=len(original['instruction_id_list']),status=status,error=error,scores=scores,
        final_sha256=hashlib.sha256(final.encode()).hexdigest(),reasoning_sha256=hashlib.sha256(reasoning.encode()).hexdigest(),
        final_chars=len(final),reasoning_chars=len(reasoning),empty_final=not final.strip(),
        native_response_id=nativeid,finish_reason=finish,truncated=finish=='length',usage=usage,
        attempts=len(by_prompt[index]),CPU_expected_rendered_input_ids_sha256=None,
        actual_native_input_ID_hash=None,native_ID_limitation='Original chat API exports usage/responseID; No native token-ID hash claimed; exact original raw messages and native usage retained.'))
indices=np.random.default_rng(42).integers(0,100,size=(10000,100))
weights=np.array([d['instructions'] for d in details],dtype=float)
result_metrics={}
for name in metrics:
    v=np.array([d['scores'][name] for d in details]);w=weights if name.startswith('inst_') else np.ones(100)
    boot=(v[indices]*w[indices]).sum(axis=1)/w[indices].sum(axis=1)
    result_metrics[name]=dict(score=float(np.average(v,weights=w)),prompt_cluster_95ci=np.quantile(boot,[.025,.975]).tolist())
(OUT/'per-sample.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in details))
(OUT/'all-http-attempts.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in attempts))
(OUT/'failures-empty-length-strictfail.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in details if x['status']!='complete' or x['empty_final'] or x['truncated'] or x['scores']['prompt_level_strict']!=1))
summary=dict(scope='Original first100 frozen100 e4ed/c976 flag1 FFN+input+outputVerify S4 target Prefill/Verify/MTP K3, generic input route matching performance, default thinking/model sampling, batch13/max32768/naturalEOS',
    samples=100,instructions=int(weights.sum()),metrics=result_metrics,
    failures=sum(x['status']!='complete' for x in details),empty_final=sum(x['empty_final'] for x in details),
    length=sum(x['truncated'] for x in details),finish_reason_counts=dict(collections.Counter(str(x['finish_reason']) for x in details)),
    reasoning_nonempty=sum(x['reasoning_chars']>0 for x in details),attempts=len(attempts),
    prompts_with_multiple_attempts=sum(len(v)>1 for v in by_prompt.values()),
    response_status_counts=dict(collections.Counter(str(x.get('status_code',x.get('error'))) for x in attempts)),
    raw_input_generation_fields='Exactly messages/model/max_tokens in every actual request; no thinking/temperature/seed override',
    bootstrap=dict(replicates=10000,seed=42,unit='prompt cluster',percentile='linear 2.5/97.5'),
    original_verifier_recomputed_all100=True,all_native_score_values_match=True,
    verifier_files=actual_verifiers,predictions=dict(path=str(predfiles[0]),sha256=sha(predfiles[0])),
    reviews=dict(path=str(reviewfiles[0]),sha256=sha(reviewfiles[0])),
    qualification='Candidate quality observation; no paired OFF arm and no pure quantization attribution. Same frozen100 e4ed/c976 flag1 binary and generic input routing as the corresponding native13 performance candidate (qualityKV532480,performanceKV113152; client14/native13 performance uses121856); no quality-only switch. Four original verifier metrics and all failures/empty/length retained. Not the published79.5 recipe reproduction or first100 representativeness claim.')
dump(OUT/'summary.json',summary)
print(json.dumps(summary,ensure_ascii=False),flush=True)
