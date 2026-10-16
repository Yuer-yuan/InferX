"""Reconstruct independent proxy budgets from archived clients and engine clocks."""
from pathlib import Path
import sys,json,hashlib,copy,statistics
root=Path(__file__).resolve().parent;run=root/'raw/independent-native-budgets-v6'
sys.path.insert(0,str(run/'harness'))
from inferx.pd_baselines import load_frozen_workloads
from inferx.ebplus_calibration import qualify_tool_rows
from inferx.qoserve_profile import freeze_common_budgets
from inferx.qoserve_validation import parse_contract
from scripts.calibrate_qoserve_budgets import join_solo_records

templates,_=load_frozen_workloads(root/'frozen-workloads')
state=json.loads((run/'budget-calibration.json').read_text());campaign=json.loads((run/'campaign.json').read_text())
assert state['status']=='complete' and campaign['status']=='restored' and campaign['restored'] is True
assert len(state['cases'])==7 and len(campaign['cases'])==7
for n,digest in campaign['harness_sha256'].items():assert hashlib.sha256((run/'harness'/n).read_bytes()).hexdigest()==digest
calibration=dict(identity=state['identity'],phase='independent-native-solo',strategy='plain-native',short=[],cold=[],resume=[])
strict=0;cases=[];tokens={}
for case in state['cases']:
 directory=run/case['case']['id'];kind=case['group'];prompt='tool-24576' if kind=='short' else kind+'-131072'
 runtime=json.loads((directory/'native-runtime.json').read_text())
 for key,value in dict(context_window=131328,concurrency=4,max_active=4,chunk=4096,
   long_prefill_token_threshold=0,scheduler_reserve_full_isl=True,num_gpu_blocks=98,
   enforce_eager=True,async_scheduling=False,speculative=False,kv_dtype='fp8',mamba_cache_mode='align',
   scheduler_class='ObservedNativeScheduler',qoserve_policy_enabled=False,profiling_mode='off').items():assert runtime[key]==value,(key,runtime[key])
 assert json.loads((directory/'quality-result.json').read_text())['status']=='passed'
 def read(name):return [json.loads(l) for l in (directory/name).read_text().splitlines()]
 for name,count in (('warmup.jsonl',4),('policy-prime.jsonl',100)):
  rows=read(name);assert len(rows)==count;qualify_tool_rows(rows,templates,['tool-24576']*count);strict+=count
 rows=read('measured.jsonl');assert len(rows)==(24 if kind=='short' else 1)
 checked=copy.deepcopy(rows)
 for row in checked:
  extra=row['request']['vllm_xargs']
  tags={k:extra.pop(k) for k in ('qoserve_tier','qoserve_deadline_s','qoserve_tbt_s','qoserve_trace_id')}
  parsed=parse_contract(tags);assert parsed==dict(tier=0 if kind=='short' else 1,deadline_s=3600.,tbt_s=1.,trace_id=row['id'])
  if 'vllm_xargs' not in templates[prompt]['payload']:assert not extra;row['request'].pop('vllm_xargs')
 proof=qualify_tool_rows(checked,templates,[prompt]*len(rows));strict+=len(rows)
 observer=json.loads((directory/'model-observer.json').read_text());joined=join_solo_records(rows,observer)
 assert joined==case['joined']
 for client,model in zip(rows,joined):assert len(model['model_token_times_s'])==client['summary']['completion_tokens']
 calibration[kind].extend(joined)
 tokens[kind]=proof['actual_prompt_tokens']
 cases.append(dict(case=case['case']['id'],group=kind,measured=len(rows),prompt_tokens=proof['actual_prompt_tokens'],
   cached_tokens=[r['summary']['cached_tokens'] for r in rows],actual_completion_tokens=[r['summary']['completion_tokens'] for r in rows]))
assert calibration==state['calibration']
budgets=freeze_common_budgets(calibration,state['identity']);assert budgets==state['budgets']==json.loads((run/'common-budgets.json').read_text())
report=dict(status='verified',source_archive_sha256=json.loads((root/'independent-native-budgets-v6-inventory.json').read_text())['archive_sha256'],
 strict_tool_requests=strict,passed_code_cases=7,measured_requests=sum(len(calibration[k]) for k in ('short','cold','resume')),
 trace_joins_unique=True,model_clock_count_matches_usage=True,source_harness_hashes_verified=True,
 common_budgets_recomputed_exactly=True,canonical_payloads_preserved_after_only_four_tags_removed=True,cases=cases,budgets=budgets)
(root/'raw-audit.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
print(json.dumps(dict(status=report['status'],strict_tool_requests=strict,passed_code_cases=7,measured_requests=report['measured_requests'],cases=cases,independent_native=budgets['independent_native'])))
