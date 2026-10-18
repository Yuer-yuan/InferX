"""CPU-only fixed original-prefix snapshots, using existing original review scores."""
import collections, hashlib, json, pathlib, time

def records(folder):
 out={}
 for p in sorted(folder.rglob('*.jsonl*')):
  if not p.is_file():continue
  for line in p.read_bytes().splitlines(keepends=True):
   try:x=json.loads(line)
   except ValueError:continue
   if isinstance(x,dict) and isinstance(x.get('index'),int):
    assert x['index'] not in out,(p,x['index']);out[x['index']]=(x,line)
 return out

def snapshot(r):
 out=r/'fixed-checkpoints';out.mkdir(exist_ok=True)
 pending=[n for n in (30,60,100) if not (out/f'prefix{n:03d}'/'snapshot.json').exists()]
 if not pending:return
 preds=records(r/'accuracy/native/predictions');reviews=records(r/'accuracy/native/reviews')
 cohort=[json.loads(l) for l in (r/'first100-frozen-cohort.jsonl').read_text().splitlines() if l.strip()]
 for n in pending:
  if not set(range(n)).issubset(preds) or not set(range(n)).issubset(reviews):continue
  d=out/f'prefix{n:03d}';d.mkdir(exist_ok=False)
  rawpred=b''.join(preds[i][1].rstrip(b'\n')+b'\n' for i in range(n));rawrev=b''.join(reviews[i][1].rstrip(b'\n')+b'\n' for i in range(n))
  (d/'predictions-original-lines.jsonl').write_bytes(rawpred);(d/'reviews-original-lines.jsonl').write_bytes(rawrev)
  responses={}
  for p in (r/'http-attempts').glob('*/response.json'):
   x=json.loads(p.read_text())
   if x.get('id'):responses[x['id']]=x
  metrics={}
  for name in ('prompt_level_strict','inst_level_strict','prompt_level_loose','inst_level_loose'):
   w=[len(cohort[i]['corpus']['instruction_id_list']) if name.startswith('inst_') else 1 for i in range(n)]
   v=[float(reviews[i][0]['sample_score']['score']['value'][name]) for i in range(n)]
   metrics[name]=sum(a*b for a,b in zip(w,v))/sum(w)
  finish=collections.Counter();errors=empty=0;raw=[]
  for i in range(n):
   p=preds[i][0];model=p.get('model_output');bad=bool(p.get('error') or not model or not model.get('choices'));errors+=bad
   response=responses.get(model.get('id')) if model else None
   choice=(response.get('choices') or [{}])[0] if response else {}
   reason=choice.get('finish_reason');finish[str(reason)]+=1
   text=choice.get('message',{}).get('content') or '';empty+=not text.strip()
   raw.append(dict(ordinal=i,key=cohort[i]['corpus']['key'],native_response_id=model.get('id') if model else None,raw_response_found=response is not None,finish_reason=reason,error=p.get('error'),empty_content=not text.strip()))
  x=dict(at_ns=time.time_ns(),fixed_prefix=n,ordinals=list(range(n)),predictions_completed_total=len(preds),original_review_metrics=metrics,instructions=sum(len(cohort[i]['corpus']['instruction_id_list']) for i in range(n)),errors=errors,empty_final=empty,finish_reason_counts=dict(finish),raw=raw,prediction_sha256=hashlib.sha256(rawpred).hexdigest(),review_sha256=hashlib.sha256(rawrev).hexdigest(),scope='Exploratory fixed-prefix original cache scores; full100 and independent original verifier rescoring remain required',no_new_GPU_or_HTTP=True,no_extra_verifier=True)
  (d/'snapshot.json').write_text(json.dumps(x,indent=2)+'\n')
