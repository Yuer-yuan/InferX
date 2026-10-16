"""Reproduce the independent calibration fit; no formal outcome data allowed."""
import hashlib,json,math
from collections import Counter
from pathlib import Path
from inferx.qoserve_profile import fit_profile,validate_step_row

root=Path(__file__).resolve().parent;run=root/'raw/independent-profile-v5'
plan=json.loads((run/'plan.json').read_text());state=json.loads((run/'profile-result.json').read_text());campaign=json.loads((run/'campaign.json').read_text())
assert campaign['status']=='restored' and campaign['restored'] is True and state['status']=='complete'
assert state['identity']==plan['identity'] and len(state['rollouts'])==39
assert hashlib.sha256((run/'plan.json').read_bytes()).hexdigest()==state['plan_sha256']
assert state['worker_sha256']==hashlib.sha256((run/'harness/scripts/calibrate_qoserve_model.py').read_bytes()).hexdigest()
specs={r['spec']['rollout_id']:r for r in state['rollouts']};assert len(specs)==39
rows=[json.loads(line) for line in (run/'steps.jsonl').read_text().splitlines()]
counts=Counter();train=[];held=[];pps=[];hist={}
for row in rows:
 validate_step_row(row,plan['identity']);entry=specs[row['rollout_id']];spec=entry['spec']
 assert row['engine_id']==state['engine_id'] and row['cache_salt']==spec['long_salt'] and row['phase']==spec['phase']
 counts[row['rollout_id']]+=1
 assert row['warmup']==(counts[row['rollout_id']]<=spec['warmup_steps'])
 if spec['kind']=='initial-pps':
  assert row['features']['batch_num_prefill_tokens']==512
  if not row['warmup']:pps.append(512/row['timing']['batch_execution_time'])
 elif spec['kind']=='train':train.append(row)
 else:held.append(row)
 key=spec['kind']+':'+str(spec['prefill_target'])+':'+str(spec['decoders'])
 hist.setdefault(key,Counter())[row['features']['batch_num_tokens']]+=1
assert all(counts[rid]==r['rows'] and r['status']=='collected' for rid,r in specs.items())
assert len(pps)==20
import statistics
assert math.isclose(statistics.median(pps),state['initial_pps'],rel_tol=1e-12)
result=fit_profile(train,held,plan)
result['input_evidence']=dict(archive_sha256=json.loads((root/'independent-profile-v5-inventory.json').read_text())['archive_sha256'],
 raw_steps_sha256=hashlib.sha256((run/'steps.jsonl').read_bytes()).hexdigest(),
 fitter_sha256=hashlib.sha256(Path(__import__('inferx.qoserve_profile',fromlist=['x']).__file__).read_bytes()).hexdigest(),
 actual_rows=len(rows),train_rows=len(train),holdout_rows=len(held),initial_pps=state['initial_pps'],
 actual_total_token_histograms={k:{str(t):n for t,n in sorted(v.items())} for k,v in hist.items()})
(root/'fitted-profile.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
print(json.dumps(dict(qualified=result['qualified'],reasons=result['qualification_reasons'],cutoff_s=result['cutoff_s'],
 rows=result['input_evidence']['actual_rows'],initial_pps=state['initial_pps'],
 bins={k:{a:v[a] for a in ('eligible_training_rows','rank','training_context_bands','holdout_context_bands','holdout')} for k,v in result['bins'].items()},
 candidates={k:{a:v[a] for a in ('count','guard_coverage','mean_absolute_relative_error','p95_prediction_ratio')} for k,v in result['candidates'].items()})))
