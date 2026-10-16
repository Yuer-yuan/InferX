"""Keep original training; qualify only new original/extended-cycle holdouts."""
from pathlib import Path
import sys,json,hashlib,statistics,math
from collections import Counter
root=Path(__file__).resolve().parent;run=root/'raw/independent-profile-repair-v9';old=root.parent/'independent-profile-v5/raw/independent-profile-v5'
sys.path.insert(0,str(run/'harness'))
from inferx.qoserve_profile import fit_profile,validate_step_row
plan=json.loads((run/'plan.json').read_text());state=json.loads((run/'profile-result.json').read_text());campaign=json.loads((run/'campaign.json').read_text());definition=json.loads((run/'repair-definition.json').read_text())
assert state['status']=='complete' and campaign['status']=='restored' and campaign['restored'] and len(state['rollouts'])==37
assert definition['retain_initial_training'] and definition['replace_initial_holdout'] and definition['separate_protocol_guard_required']
assert state['identity']==plan['identity']==definition['identity']
for file,key in (('plan.json','plan_sha256'),('repair-definition.json','repair_definition_sha256')):assert hashlib.sha256((run/file).read_bytes()).hexdigest()==state[key]
assert state['worker_sha256']==hashlib.sha256((run/'harness/scripts/calibrate_qoserve_model.py').read_bytes()).hexdigest()
assert hashlib.sha256((run/'basis-fit.json').read_bytes()).hexdigest()==definition['basis_fit_sha256']
initial=[json.loads(l) for l in (old/'steps.jsonl').read_text().splitlines()]
old_spec={r['spec']['rollout_id']:r['spec'] for r in json.loads((old/'profile-result.json').read_text())['rollouts']}
training=[r for r in initial if old_spec[r['rollout_id']]['kind']=='train']
assert len(training)==4129
rows=[json.loads(l) for l in (run/'steps.jsonl').read_text().splitlines()];specs={r['spec']['rollout_id']:r for r in state['rollouts']};counts=Counter();pps=[];held=[];diagnostic=[]
for row in rows:
 validate_step_row(row,plan['identity']);entry=specs[row['rollout_id']];spec=entry['spec'];counts[row['rollout_id']]+=1
 assert row['engine_id']==state['engine_id'] and row['cache_salt']==spec['long_salt'] and row['phase']==spec['phase'] and row['calibration_kind']==spec['kind']
 assert row['warmup']==(counts[row['rollout_id']]<=spec['warmup_steps'])
 assert math.isfinite(row['host_process_cpu_time_s']) and row['host_process_cpu_time_s']>=0
 if spec['kind']=='initial-pps':
  assert row['features']['batch_num_prefill_tokens']==512
  if not row['warmup']:pps.append(512/row['timing']['batch_execution_time'])
 elif spec['kind']=='train':training.append(row)
 elif spec['kind']=='holdout':
  assert row['holdout_protocol']==spec['holdout_protocol'];held.append(row)
 elif spec['kind']=='diagnostic':diagnostic.append(row)
 else:raise AssertionError('unknown rollout role')
assert all(counts[rid]==entry['rows'] and entry['status']=='collected' for rid,entry in specs.items())
assert len(pps)==20 and math.isclose(statistics.median(pps),state['initial_pps'],rel_tol=1e-12)
assert {r['holdout_protocol'] for r in held}=={'original-cycle-v1','remainder-cycle-v1'}
plan['required_holdout_protocols']=['original-cycle-v1','remainder-cycle-v1']
result=fit_profile(training,held,plan)
result['input_evidence']=dict(initial_archive_sha256=definition['basis_archive_sha256'],
 repair_archive_sha256=json.loads((root/'independent-profile-repair-v9-inventory.json').read_text())['archive_sha256'],
 initial_fit_sha256=definition['basis_fit_sha256'],repair_definition_sha256=state['repair_definition_sha256'],
 initial_training_rows=4129,supplemental_training_rows=len(training)-4129,fresh_holdout_rows=len(held),
 diagnostic_rows_excluded=len(diagnostic),initial_holdout_rows_excluded=339,initial_pps=state['initial_pps'],
 fitter_sha256=hashlib.sha256(Path(sys.modules['inferx.qoserve_profile'].__file__).read_bytes()).hexdigest())
(root/'fitted-profile.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
summary=dict(qualified=result['qualified'],reasons=result['qualification_reasons'],cutoff_s=result['cutoff_s'],
 training={k:dict(rows=v['eligible_training_rows'],rank=v.get('rank')) for k,v in result['bins'].items()},
 protocols={p:dict(bins={k:(v['count'],v['guard_coverage']) for k,v in group['bins'].items()},
 candidates={k:(v['count'],v['guard_coverage']) for k,v in group['candidates'].items()}) for p,group in result['holdout_protocols'].items()},
 negative_observed=len(result['prediction_failures']),negative_deployment=len(result['deployment_prediction_failures']),coefficients=result['coefficients'])
(root/'fit-summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary))
