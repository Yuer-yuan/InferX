"""Use the tested strict exporter, plus actual placement/build evidence."""
import argparse,hashlib,json,os,re
from pathlib import Path
from inferx.pd_baselines import load_frozen_workloads
from inferx.sota_results import verify_prompt_count_contract
from scripts.analyze_sota_validation import export_campaign
p=argparse.ArgumentParser();p.add_argument('--run-id',default='transport-local-v1');p.add_argument('--output',type=Path,required=True);a=p.parse_args()
r=Path(os.environ['INFERX_ROOT']);root=r/'morphx-control/raw'/a.run_id
assert Path(a.run_id).name==a.run_id and a.run_id not in {'.','..'}
metadata=json.loads((root/'campaign.json').read_text());assert metadata['client_cpu_affinity']==[40,41,42,43,96,97,98,99]
templates,_=load_frozen_workloads(r/'concurrency-roofline');contract=verify_prompt_count_contract(r/'models/qwen38-int4-redhat')
result=export_campaign(root,templates,model_root=r/'models/qwen38-int4-redhat');configuration={x['id']:x for x in json.loads((root/'config.json').read_text())};placement=[]
for group in result['groups']:
 case=configuration[group['case']];folder=root/group['identity'];startup=json.loads((folder/'startup.json').read_text());assert startup['args'][:3]==['taskset','-c','28-35']
 if not case['morphx']['enabled']:continue
 log=(folder/'server.log').read_text();observed=re.findall(r'\[MorphX-affinity\] requested=(\d+) observed=(\d+)',log)
 assert observed and set(observed)=={('36','36')} and case['environment']['SMSCHED_AGENT_CPU']=='36'
 library=Path(case['environment']['NEUTRINO_HOOK_DRIVER']);assert hashlib.sha256(library.read_bytes()).hexdigest()==case['morphx']['library_sha256']
 placement.append(dict(case=case['id'],group=group['group'],repeat=group['repeat'],requested_cpu=36,observed_cpu=36,library_sha256=case['morphx']['library_sha256']))
code=r/'code-pd-baselines';sources={name:hashlib.sha256((code/name).read_bytes()).hexdigest() for name in ['scripts/analyze_sota_validation.py','inferx/sota_results.py','inferx/pd_baselines.py','inferx/concurrency.py','inferx/tuning.py','scripts/analyze_pd_baselines.py']}
report=dict(scope='One full diagnostic per case; not3independent repetitions or a matched-solo slowdown. Originalauthorceilingunknown.',campaign=result,prompt_count_contract=contract,placement=placement,analysis_source_sha256=sources,helper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
a.output.write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+'\n');print(json.dumps([dict(case=x['case'],short_p95_s=x['short_p95_s'],long_s=x['long_s'],output_tokens_per_s=x['output_tokens_per_s'],strict=x['strict_requests']) for x in result['groups']]))
