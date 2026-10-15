"""Audit full model gates after restoration; never turn failed gates into scores."""
import hashlib,json,os,re
from pathlib import Path
from inferx.concurrency import reconstruct_tool_calls
from inferx.pd_baselines import load_frozen_workloads,validate_sglang_runtime
from inferx.sota_results import verify_prompt_count_contract
r=Path(os.environ['INFERX_ROOT']);p=r/'morphx-control/raw/models-local-v3';metadata=json.loads((p/'campaign.json').read_text());assert metadata['status']=='restored' and metadata['restored']
assert len(metadata['cases'])==4
verify_prompt_count_contract(r/'models/qwen38-int4-redhat');templates,_=load_frozen_workloads(r/'concurrency-roofline');sources={};output=[]
def data(path):
 raw=path.read_bytes();sources[str(path.relative_to(r))]=hashlib.sha256(raw).hexdigest();return raw
for entry in metadata['cases']:
 row=dict(variant=entry['variant'],status=entry['status'],library_sha256=entry['library_sha256'])
 if entry['status']!='passed':row['error']=entry.get('error');output.append(row);continue
 folder=p/entry['variant'];case=entry['case'];startup=json.loads(data(folder/'startup.json'));validate_sglang_runtime(case,startup['server_info'])
 state=startup['server_info']['internal_states'][0]
 assert state['memory_usage']['token_capacity']==131328 and state['effective_max_running_requests_per_dp']==4
 assert state['max_mamba_cache_size']==16 and state['mamba_ssm_dtype']=='bfloat16' and state['kv_cache_dtype']=='fp8_e4m3'
 assert state.get('speculative_algorithm') is None and state['disable_overlap_schedule']
 assert all(x==0 for x in state['memory_usage']['graph'].values())
 assert json.loads(data(folder/'quality-result.json'))['status']=='passed'
 for filename,prompt,count,expected_number in [('requests.jsonl','cold-131072',131060,1),('warmup.jsonl','tool-24576',24568,4)]:
  records=[json.loads(x) for x in data(folder/'full128k'/filename).splitlines()]
  assert len(records)==expected_number
  for record in records:
   expected=templates[prompt];calls=reconstruct_tool_calls(record['events'])
   assert record['request']==expected['payload'] and record['functional_ok'] and record['summary']['ok'] and not record['summary']['truncated']
   assert record['summary']['prompt_tokens']==count and len(calls)==1 and calls[0]['name']==expected['expected_tool']['name'] and calls[0]['arguments']==expected['expected_tool']['arguments']
 coverage=json.loads(data(folder/'full128k/morphx-coverage.json'));snapshot=data(folder/'full128k/morphx-coverage.log')
 assert coverage['status']=='observed' and coverage['snapshot_sha256']==hashlib.sha256(snapshot).hexdigest() and coverage['patched_launches_observed']>0
 log=data(folder/'server.log').decode();assert '[MorphX-affinity] requested=36 observed=36' in log;data(folder/'full128k/result.json');data(folder/'full128k/specifications.json')
 row.update(strict_tools=5,actual_long_tokens=131060,patched_launches=coverage['patched_launches_observed'],patched_kernel_names=coverage['patched_kernel_names'],agent_nonzero_min_writes=len(re.findall(r'\[MorphX-control\].*new=[1-9][0-9]*:',log)),
   runtime=dict(kv_capacity=131328,concurrency=4,ssm_slots=16,ssm_dtype='bfloat16',kv_dtype='fp8_e4m3',mtp=0,eager=True))
 output.append(row)
for filename in ['campaign.json','controller.py','restored-service.json']:data(p/filename)
report=dict(scope='Actual code+four24K+full128K gate qualification, not SOTA efficacy or universal numerical/SM occupancy guarantee.',rows=output,source_sha256=sources)
(r/'morphx-control/local-model-audit.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(output))
