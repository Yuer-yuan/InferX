"""Read-only actual payload/output/native association; no model request, no score."""
import pathlib,json,hashlib,collections,sys
R=pathlib.Path(__file__).resolve().parent;OUT=R/(sys.argv[1] if len(sys.argv)>1 else 'CPU-http-native-join-001');OUT.mkdir(exist_ok=False)
def load(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')
cohort=[json.loads(l) for l in (R/'first100-frozen-cohort.jsonl').read_text().splitlines()]
def canon(x):return json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(',',':'))
lookup={canon(x['messages']):x for x in cohort}
native=[]
for l in (R/'service/native.jsonl').read_text().splitlines():
 try:x=json.loads(l)
 except ValueError:continue
 if x['event']=='request_done':native.append(x)
rows=[];used=set()
for p in sorted((R/'http-attempts').glob('*/response.json')):
 response=load(p)
 if not response.get('choices'):continue
 payload=load(p.parent/'payload.json');a=load(p.parent/'attempt.json');frozen=lookup[canon(payload['messages'])]
 assert sorted(payload)==['max_tokens','messages','model'] and payload['max_tokens']==32768 and payload['model']=='qwen38-27b'
 t=response['timings'];u=response['usage']
 candidates=[v for v in native if v['result']['prompt_tokens']==u['prompt_tokens'] and v['result']['completion_tokens']==u['completion_tokens'] and abs(v['timings_seconds']['prefill']*1000-t['prompt_ms'])<1e-8 and abs(v['timings_seconds']['decode']*1000-t['predicted_ms'])<1e-8]
 assert len(candidates)==1,(p,len(candidates));n=candidates[0];nid=n['request']['request_id'];assert nid not in used;used.add(nid)
 assert n['result']['prefix_cache_hit_tokens']==t['cache_n']==0 and t['prompt_n']==u['prompt_tokens'] and t['predicted_n']==u['completion_tokens']
 assert n['request']['requested_output_tokens']==32768 and n['request']['enable_thinking'] is True and n['speculative']['backend']=='mtp' and n['speculative']['draft_window']==3
 m=response['choices'][0]['message'];content=m.get('content') or '';reasoning=m.get('reasoning_content') or m.get('reasoning') or ''
 rows.append(dict(ordinal=frozen['ordinal'],key=frozen['corpus']['key'],prompt_sha256=frozen['prompt_sha256'],record_sha256=frozen['record_sha256'],attempt=a['attempt'],response_id=response['id'],native_request_id=nid,join_method='Unique exact native usage + FP64-reconstructed prefill/decode times from the same GenerationOutcome, not a shared exported ID',request_sha256=sha(p.parent/'request.raw.bin'),response_sha256=sha(p.parent/'response.raw.bin'),native=n,content_sha256=hashlib.sha256(content.encode()).hexdigest(),reasoning_sha256=hashlib.sha256(reasoning.encode()).hexdigest(),content_chars=len(content),reasoning_chars=len(reasoning),empty=not content.strip(),finish_reason=response['choices'][0]['finish_reason']))
(OUT/'per-completion.jsonl').write_text(''.join(json.dumps(v,ensure_ascii=False)+'\n' for v in rows))
summary=dict(completions=len(rows),distinct_prompt_keys=len({x['key'] for x in rows}),all_actual_payloads_exact_original_messages=True,all_used_generation_max32768=True,all_actual_native_cache0=True,all_actual_MTP_K3=True,actual_sampling_configs=list({canon(x['native']['request']['sampling']):x['native']['request']['sampling'] for x in rows}.values()),finish_reason_counts=dict(collections.Counter(x['finish_reason'] for x in rows)),empty=sum(x['empty'] for x in rows),reasoning_present=sum(x['reasoning_chars']>0 for x in rows),native_token_ID_hash_captured=False,scope='Raw input/usage/native/response identity audit; not task scores or expanded quality PASS')
write(OUT/'summary.json',summary);print(json.dumps(summary),flush=True)
