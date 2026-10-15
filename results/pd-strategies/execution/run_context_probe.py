"""Owned GPU1 compatibility campaign using the existing tested restoration contract."""
import hashlib,json,os,signal,time
from pathlib import Path
from inferx.pd_baselines import GPU1,run_owned_command,snapshot_harness
from inferx.service_ownership import process_identity
from scripts.run_pd_baselines import run_with_restoration
from scripts import run_concurrency_campaign as previous
root=Path(os.environ['INFERX_ROOT']);assert root==Path('/home/guokc/mypro/gnn-mount-sda1/InferX')
prior=root/'pd-baselines/raw/pd-strategies-credit-diagnostic-v1/campaign.json'
while True:
 try:d=json.loads(prior.read_text())
 except (FileNotFoundError,json.JSONDecodeError):d={'status':'running'}
 if d['status']!='running':
  assert d['status']=='restored' and d.get('restored');break
 time.sleep(1)
output=root/'pd-baselines/raw/pd-strategies-green-context-numerical-v1';output.mkdir(exist_ok=False)
previous.DIRECTORY=output
saved=json.loads((root/'tuning/best-service.json').read_text());assert saved==json.loads((root/'context-agents/runtime.json').read_text())
identity=process_identity(saved['pid'],previous.BINARY);assert all(saved.get(k)==v for k,v in identity.items())
saved['original_argv']=Path(f'/proc/{saved["pid"]}/cmdline').read_bytes().decode().rstrip('\0').split('\0')
(output/'original-service.json').write_text(json.dumps(saved,indent=2));(output/'config.json').write_text('[]\n')
state=dict(status='running',started_unix=time.time(),cases=[],controller_pid=os.getpid(),workload_definitions=[],
 harness_sha256=snapshot_harness(root/'code-pd-baselines',output/'harness',[str(p.relative_to(root/'code-pd-baselines')) for folder in ['scripts','inferx'] for p in sorted((root/'code-pd-baselines'/folder).rglob('*.py'))]))
def save():(output/'campaign.json').write_text(json.dumps(state,indent=2)+'\n')
def measure():
 previous.gpu_free();env=dict(os.environ,CUDA_VISIBLE_DEVICES=GPU1,PYTHONPATH=str(root/'code-pd-baselines'))
 command=[str(root/'pd-baselines/sglang0519-venv/bin/python'),'-m','scripts.probe_pd_green_context','--output',str(output/'probe-result.json')]
 with (output/'probe.log').open('x') as log:code=run_owned_command(command,env,log,timeout=240)
 state['cases'].append(dict(case='green-context-numerical',returncode=code))
 if (output/'probe-result.json').exists():state['probe_result']=json.loads((output/'probe-result.json').read_text())
 save()
save()
try:
 run_with_restoration(saved,previous.restore,[root/'context-agents/runtime.json',root/'tuning/best-service.json'],lambda:previous.final_shutdown_check(saved),measure)
 state['status']='restored'
except BaseException as exc:
 state.update(status='interrupted_or_failed',error=f'{type(exc).__name__}: {exc}');raise
finally:
 state.update(finished_unix=time.time(),restored=(output/'restored-service.json').exists());save()
print(json.dumps(dict(event='probe_controller_done',status=state['status'],probe=state.get('cases'))),flush=True)
