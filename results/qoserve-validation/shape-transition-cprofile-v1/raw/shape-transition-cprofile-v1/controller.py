"""Owned unscored CPU diagnostic with the existing exact-restoration lifecycle."""
import hashlib,json,os,time
from pathlib import Path
from inferx.ebplus_calibration import require_terminal_restoration
from inferx.pd_baselines import run_owned_command,snapshot_harness
from inferx.service_ownership import process_identity
from scripts import run_concurrency_campaign as previous
from scripts.run_pd_baselines import run_with_restoration

root=Path(os.environ['INFERX_ROOT']);q=root/'qoserve-validation';code=q/'code-task3-v9'
prior=q/'raw/independent-profile-repair-v9';overlay=q/'overlays/adapter-profile-v9-reference'
plan=json.loads((code/'configs/qoserve-validation.json').read_text())
saved=json.loads((root/'tuning/best-service.json').read_text());assert saved==json.loads((root/'context-agents/runtime.json').read_text())
identity=process_identity(saved['pid'],previous.BINARY);assert all(saved.get(k)==v for k,v in identity.items())
require_terminal_restoration(json.loads((prior/'campaign.json').read_text()),json.loads((prior/'restored-service.json').read_text()),saved)
saved['original_argv']=Path('/proc/'+str(saved['pid'])+'/cmdline').read_bytes().decode().rstrip('\0').split('\0')
assert saved['original_argv']==json.loads((prior/'original-service.json').read_text())['original_argv']
out=q/'raw/shape-transition-cprofile-v1';out.mkdir(parents=True,exist_ok=False);previous.DIRECTORY=out
(out/'original-service.json').write_text(json.dumps(saved,indent=2)+'\n');(out/'plan.json').write_bytes((code/'configs/qoserve-validation.json').read_bytes())
worker=Path(__file__).with_name('worker.py');(out/'worker.py').write_bytes(worker.read_bytes());(out/'controller.py').write_bytes(Path(__file__).read_bytes())
metadata=dict(status='running',controller_pid=os.getpid(),started_unix=time.time(),restored=False,cases=[],scope='Unscored CPU diagnostic; not SOTA efficacy or cost training.',
 harness_sha256=snapshot_harness(code,out/'harness',[str(p.relative_to(code)) for d in ('inferx','scripts') for p in sorted((code/d).glob('*.py'))]))
def flush():(out/'campaign.json').write_text(json.dumps(metadata,indent=2)+'\n')
def measure():
 previous.gpu_free();entry=dict(case='matched-prefix-cpu-profile',state='starting');metadata['cases'].append(entry);flush()
 command=['taskset','-c','28-35',str(root/plan['base_case']['binary']),'-u',str(worker),'--plan',str(out/'plan.json'),'--output',str(out/'profile-result.json')]
 environment={**os.environ,**plan['base_case']['environment'],'PYTHONPATH':str(overlay)+':'+str(code),
  'CUDA_VISIBLE_DEVICES':plan['identity']['gpu_uuid'],'HF_ENDPOINT':'https://hf-mirror.com','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1',
  'TMPDIR':str(root/'tmp'),'INFERX_VLLM_RUNTIME':str(out/'runtime.json'),'INFERX_QOSERVE_EVIDENCE':str(out/'model-observer.json')}
 entry['command']=command;flush()
 with (out/'worker.log').open('x') as log:status=run_owned_command(command,environment,log,timeout=900)
 if status!=0:entry.update(state='failed',exit_code=status);flush();raise RuntimeError('CPU diagnostic failed, raw preserved')
 audit=json.loads((out/'cpu-profile-audit.json').read_text());assert audit['matched_features_verified'] and len(audit['intentional_aborts'])==5
 entry['state']='diagnosed';flush()
flush()
try:
 run_with_restoration(saved,previous.restore,[root/'context-agents/runtime.json',root/'tuning/best-service.json'],lambda:previous.final_shutdown_check(saved,gpu0_state='stopped'),measure)
 metadata['status']='restored'
except BaseException as exc:metadata.update(status='failed',error=repr(exc));raise
finally:metadata.update(finished_unix=time.time(),restored=(out/'restored-service.json').exists());flush()
