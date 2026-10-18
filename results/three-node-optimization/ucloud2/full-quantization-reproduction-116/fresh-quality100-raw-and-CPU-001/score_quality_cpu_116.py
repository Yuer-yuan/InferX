from pathlib import Path
import json,hashlib,collections,copy,importlib.metadata
from evalscope.benchmarks.ifbench import evaluation_lib
R=Path(__file__).resolve().parent;Q=R/'deployment116/quality100/evaluation/accuracy';OUT=R/'quality-independent-CPU-001';OUT.mkdir(exist_ok=False)
load=lambda p:json.loads(p.read_text());rows=lambda p:[json.loads(l)for l in p.read_text().splitlines()if l.strip()];sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def save(n,d):(OUT/n).write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
def compact(d):
 if isinstance(d,dict):return{k:compact(v)for k,v in d.items()if v is not None}
 if isinstance(d,list):return[compact(v)for v in d]
 return d
assert importlib.metadata.version('evalscope')=='1.13.0'
verifiers={p.name:sha(p)for p in Path(evaluation_lib.__file__).parent.glob('*.py')};assert verifiers==load(R/'expected-verifier-files.json')
cohort=rows(R/'first100-frozen-cohort.jsonl');assert len(cohort)==100
pf=list((Q/'native/predictions').rglob('*.jsonl'));rf=list((Q/'native/reviews').rglob('*.jsonl'));assert len(pf)==len(rf)==1
pp=rows(pf[0]);rr=rows(rf[0]);assert len(pp)==len(rr)==100;preds={x['index']:x for x in pp};reviews={x['index']:x for x in rr};assert set(preds)==set(reviews)==set(range(100))
metrics=['prompt_level_strict','inst_level_strict','prompt_level_loose','inst_level_loose'];details=[]
for i,frozen in enumerate(cohort):
 p=preds[i];r=reviews[i];original=frozen['corpus']
 for key in ['key','prompt','instruction_id_list','kwargs']:
  assert compact(p['metadata'][key])==compact(original[key]),(i,key,'prediction')
  assert compact(r['sample_score']['sample_metadata'][key])==compact(original[key]),(i,key,'review')
 mo=p['model_output'];assert not p.get('error')and mo and not mo.get('error')and len(mo['choices'])==1
 c=mo['choices'][0];content=c['message']['content'];final=content if isinstance(content,str)else'\n'.join(x.get('text','')for x in content if x.get('type')=='text');reasoning=''if isinstance(content,str)else'\n'.join(x.get('reasoning','')for x in content if x.get('type')=='reasoning')
 assert r['sample_score']['score']['prediction']==final
 scores={k:float(v)for k,v in evaluation_lib.process_results(copy.deepcopy(original),[final]).items()if k in metrics};assert scores==r['sample_score']['score']['value'],(i,scores,r['sample_score']['score']['value'])
 details.append(dict(ordinal=i,key=original['key'],prompt_sha256=hashlib.sha256(original['prompt'].encode()).hexdigest(),instructions=len(original['instruction_id_list']),scores=scores,empty_final=not final.strip(),final_sha256=hashlib.sha256(final.encode()).hexdigest(),reasoning_sha256=hashlib.sha256(reasoning.encode()).hexdigest(),reasoning_nonempty=bool(reasoning.strip()),stop_reason=c.get('stop_reason'),usage=mo['usage'],native_response_id=mo['id']))
summary=load(Q/'summary.json');assert summary['issues']==[]and summary['status']=='完整'
aggregates={}
for m in metrics:
 den=sum(x['instructions']if m.startswith('inst_')else 1 for x in details);num=sum(x['scores'][m]*(x['instructions']if m.startswith('inst_')else 1)for x in details);score=num/den;assert round(score,4)==summary['metrics'][m]['score'];assert summary['metrics'][m]['num']==den
 aggregates[m]=dict(score=score,numerator=round(num),denominator=den)
native=rows(R/'deployment116/service-quality100/requests.jsonl');starts=[x for x in native if x['event']=='request_start'];done=[x for x in native if x['event']=='request_done'];errors=[x for x in native if x['event']=='request_error'];assert len(starts)==len(done)==100 and not errors;assert set(x['request']['request_id']for x in done)==set(range(1,101));assert all(x['request']['enable_thinking']is True and x['request']['requested_output_tokens']==32768 and x['request']['requested_output_tokens_source']=='client'for x in starts)
assert all(x['result']['computed_prefill_tokens']==x['result']['prompt_tokens']and x['result']['prefix_cache_hit_tokens']==0 and x['result']['prefix_reuse_path']=='full_reset'for x in done)
launch=load(R/'deployment116/service-quality100/launch.json');assert launch['profile']==dict(client_concurrency=13,kv_capacity=532480,mode='accuracy',name='quality100',native_concurrency=13)
assert launch['binary_sha256']=='e919f18084f3cc09718158cfae071cdf8c58d5660715648db375643016c878f5'
out=dict(status='COMPLETE_INDEPENDENTLY_RECOMPUTED_100',samples=100,instructions=sum(x['instructions']for x in details),binary_sha256=launch['binary_sha256'],source_sha256=launch['source_sha256'],recipe_sha256=launch['recipe_sha256'],profile=launch['profile'],metrics=aggregates,errors=0,native_started=100,native_done=100,all100cold=True,original_first100_metadata_all_match=True,original_verifier_recomputed_all100=True,all_native_scores_exact_match=True,empty_final=sum(x['empty_final']for x in details),reasoning_nonempty=sum(x['reasoning_nonempty']for x in details),stop_reason_counts=dict(collections.Counter(str(x['stop_reason'])for x in details)),native_finish_reason_counts=dict(collections.Counter(x['result']['finish_reason']for x in done)),length_stops=sum(x['result']['finish_reason']=='output_limit'for x in done),sampling_values=[json.loads(v)for v in sorted({json.dumps(x['request']['sampling'],sort_keys=True)for x in starts})],verifier_files=verifiers,raw_identity=dict(predictions=dict(path=str(pf[0]),sha256=sha(pf[0])),reviews=dict(path=str(rf[0]),sha256=sha(rf[0])),native=dict(path=str(R/'deployment116/service-quality100/requests.jsonl'),sha256=sha(R/'deployment116/service-quality100/requests.jsonl'))),scope='Actual116 fresh source100/c976 build e919 with239 freshly fitted matrices/478 payloads. No inherited quality; no claim of model-wide quality guarantee or paired quantization attribution. Full100 includes all length/empty cases.')
save('actual-first100-metadata.json',[dict(index=i,**preds[i]['metadata'])for i in range(100)]);save('summary.json',out);(OUT/'all100.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n'for x in details));print(json.dumps(out,ensure_ascii=False,indent=2))
