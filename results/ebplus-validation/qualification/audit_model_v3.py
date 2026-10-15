import os,json,hashlib,time,tarfile
from pathlib import Path
from inferx.pd_baselines import load_frozen_workloads
from inferx.ebplus_calibration import qualify_tool_rows,validate_policy_evidence
from inferx.sota_results import verify_prompt_count_contract
from inferx.service_ownership import process_identity
from scripts import run_concurrency_campaign as previous
root=Path(os.environ['INFERX_ROOT']);run=root/'ebplus-validation/raw/model-gates-v3'
templates,_=load_frozen_workloads(root/'concurrency-roofline')
contract=verify_prompt_count_contract(root/'models/qwen38-int4-redhat',backend='vllm031')
campaign=json.loads((run/'campaign.json').read_text());result=json.loads((run/'model-qualification.json').read_text())
assert campaign['status']=='restored' and campaign['restored'] and result['status']=='passed'
assert len(campaign['cases'])==3 and all(x['state']=='measured' for x in campaign['cases'])
sources={}
def data(path):
 raw=path.read_bytes();sources[str(path.relative_to(run))]=hashlib.sha256(raw).hexdigest();return raw
for name,digest in campaign['harness_sha256'].items():
 assert hashlib.sha256(data(run/'harness'/name)).hexdigest()==digest
report=dict(status='passed_numeric_and_policy_execution_gates',cases=[],prompt_count_contract=contract,
            scope='Actual model/code/tools and policy qualification; not mixed-load efficacy.',unix=time.time())
reference=None
for mode in ['native','eb','auto']:
 folder=run/('qualify-'+mode);runtime=json.loads(data(folder/'native-runtime.json'))
 assert runtime['concurrency']==runtime['max_active']==4 and runtime['context_window']==131328
 assert runtime['num_gpu_blocks']==98 and runtime['block_size']==1568
 assert runtime['async_scheduling'] is False and runtime['speculative'] is False
 assert runtime['kv_dtype']=='fp8' and runtime['mamba_cache_mode']=='align' and runtime['enforce_eager'] is True
 if reference is None:reference=runtime
 else:assert runtime==reference
 assert json.loads(data(folder/'quality-result.json'))['status']=='passed'
 item=dict(mode=mode,native_runtime=runtime,checks={})
 checks=[('warmup','tool-24576',4),('full-cold','cold-131072',1),('full-resume','resume-131072',1)]
 if mode=='auto':checks.append(('policy-prime','tool-24576',100))
 for name,prompt,count in checks:
  rows=[json.loads(line) for line in data(folder/(name+'.jsonl')).splitlines() if line.strip()]
  item['checks'][name]=qualify_tool_rows(rows,templates,[prompt]*count)
 if mode!='native':
  state=json.loads(data(folder/'policy-evidence.json'));validate_policy_evidence(state)
  assert state['completed']==(107 if mode=='auto' else 7)
  item['policy']={k:v for k,v in state.items() if k!='kv_groups'}
 report['cases'].append(item)
saved=json.loads((root/'tuning/best-service.json').read_text());assert saved==json.loads((root/'context-agents/runtime.json').read_text())
restored=json.loads(data(run/'restored-service.json'));assert saved==restored
identity=process_identity(saved['pid'],previous.BINARY);assert all(saved.get(k)==v for k,v in identity.items())
argv=Path('/proc/'+str(saved['pid'])+'/cmdline').read_bytes().decode().rstrip(chr(0)).split(chr(0))
assert argv==json.loads(data(run/'original-service.json'))['original_argv']
previous.final_shutdown_check(saved,gpu0_state='stopped')
assert previous.get('/health','http://127.0.0.1:18081')['status']=='ok'
assert previous.get('/v1/models','http://127.0.0.1:18081')['data'][0]['context_window']==262144
report.update(strict_tools=sum(x['strict_requests'] for c in report['cases'] for x in c['checks'].values()),
              code_gates=3,source_sha256=sources,restoration=dict(identity=identity,argv=argv,gpu0_stopped=True,dual_manifests_equal=True))
out=root/'ebplus-validation/artifacts';path=out/'model-audit-v3.json';assert not path.exists();path.write_text(json.dumps(report,indent=2)+'\n')
files=[p for p in sorted(run.rglob('*')) if p.is_file()]
archive=out/'model-gates-v3.tar.gz';assert not archive.exists()
with tarfile.open(archive,'w:gz') as t:
 for p in files:t.add(p,arcname=str(Path(run.name)/p.relative_to(run)))
manifest=dict(archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),files={str(p.relative_to(run)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},audit_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
(out/'model-gates-v3-inventory.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps(dict(status=report['status'],strict_tools=report['strict_tools'],code_gates=3,restored_pid=saved['pid'],files=len(files),archive_bytes=archive.stat().st_size)))
