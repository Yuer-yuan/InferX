"""Final read-only identity/health check; never start or stop a service."""
import hashlib,json,os,time
from pathlib import Path
from inferx.service_ownership import process_identity
from scripts import run_concurrency_campaign as previous
from scripts.run_pd_baselines import get
r=Path('/home/guokc/mypro/gnn-mount-sda1/InferX')
manifest=r/'tuning/best-service.json';other=r/'context-agents/runtime.json'
a=json.loads(manifest.read_text());assert a==json.loads(other.read_text())
identity=process_identity(a['pid'],previous.BINARY)
assert all(a.get(k)==v for k,v in identity.items())
argv=Path(f'/proc/{a["pid"]}/cmdline').read_bytes().decode().rstrip('\0').split('\0')
original=json.loads((r/'pd-baselines/raw/pd-strategies-pdmux-profile-v1/original-service.json').read_text())['original_argv']
assert argv==original
pairs=sorted(previous.gpu_processes());assert [(pid,uuid) for pid,uuid in pairs if uuid==previous.GPU]==[(a['pid'],previous.GPU)]
health=get('/health','http://127.0.0.1:18081');assert health['status']=='ok'
models=get('/v1/models','http://127.0.0.1:18081');assert models['data'][0]['context_window']==262144
controllers={'wait_then_exec.py','wait_then_debug_cold.py','queue_simple_diagnostic.py','queue_simple_diagnostic_v2.py','run_context_probe.py','run_context_probe_v2.py','profile_pdmux.py','scripts.run_pd_baselines','scripts.run_pd_debug'}
active=[]
for p in Path('/proc').glob('[0-9]*'):
 try:
  if p.stat().st_uid!=os.getuid():continue
  args=(p/'cmdline').read_bytes().decode().rstrip('\0').split('\0')
 except (OSError,UnicodeDecodeError):continue
 if any(Path(x).name in controllers for x in args[1:]):active.append(dict(pid=int(p.name),argv=args))
assert not active,active
report=dict(verified_unix=time.time(),read_only=True,manifest_equal=True,manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),identity=identity,original_argv_exact=True,gpu_processes=pairs,gpu0_user_intentionally_stopped=True,gpu0_current_processes=[x for x in pairs if x[1]!=previous.GPU],health=health,models=models,active_experiment_controllers=active)
(r/'pd-strategies/final-service-readonly.json').write_text(json.dumps(report,indent=2)+'\n')
print('read-only exact restoration/health/no queued or running controllers verified',a['pid'])
