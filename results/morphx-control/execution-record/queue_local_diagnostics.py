"""Wait without GPU, then sequence audited qualification and one owned diagnostic."""
import hashlib,json,os,signal,subprocess,time
from pathlib import Path
r=Path(os.environ['INFERX_ROOT']);root=r/'morphx-control';path=root/'followups-local-v1.json';assert not path.exists()
child=None;stopping=False
metadata=dict(status='waiting_for_models',pid=os.getpid(),started_unix=time.time(),actions=[])
def flush():path.write_text(json.dumps(metadata,indent=2)+'\n')
def stop(number,frame):
 global stopping
 stopping=True
 if child is not None and child.poll() is None:child.send_signal(signal.SIGTERM)
for number in [signal.SIGINT,signal.SIGTERM,signal.SIGHUP]:signal.signal(number,stop)
def run(label,script,args=()):
 global child
 if stopping:raise InterruptedError('queue stopped before next action')
 entry=dict(action=label,started_unix=time.time(),state='running');metadata['actions'].append(entry);flush()
 command=[str(r/'repo-venv/bin/python'),'-u',str(root/script),*args]
 child=subprocess.Popen(command,env={**os.environ,'PYTHONPATH':str(r/'code-pd-baselines')},stdin=subprocess.DEVNULL)
 entry['child_pid']=child.pid;flush()
 while True:
  try:code=child.wait(timeout=1);break
  except subprocess.TimeoutExpired:pass
 entry.update(exit_code=code,state='complete' if code==0 else 'failed',finished_unix=time.time());flush();child=None
 if stopping:raise InterruptedError('queue stopped after child cleanup')
 if code!=0:raise RuntimeError(label+' failed')
flush()
try:
 prerequisite=root/'raw/models-local-v3/campaign.json'
 while True:
  if stopping:raise InterruptedError('queue stopped without touching the prerequisite controller')
  data=json.loads(prerequisite.read_text())
  if data['status']=='running':time.sleep(5);continue
  assert data['status']=='restored' and data['restored'] and len(data['cases'])==4 and all(x['status']=='passed' for x in data['cases'])
  break
 metadata.update(status='running',prerequisite_sha256=hashlib.sha256(prerequisite.read_bytes()).hexdigest());flush()
 run('audit','audit_local_cpu_models.py')
 run('freeze-configuration','prepare_local_cases.py')
 cases=['compiled-primary-control',*[f'morphx-control-{n}-96-ablation' for n in ['abi-only','fill','event','both']]]
 run('transport-diagnostics','run_local_diagnostics.py',['--run-id','transport-local-v1','--cases',*cases,'--groups','cold-high'])
 terminal=root/'raw/transport-local-v1/campaign.json';data=json.loads(terminal.read_text())
 assert data['status']=='restored' and data['restored'] and len(data['cases'])==5 and all(x['state']=='measured' for x in data['cases'])
 metadata.update(status='complete',diagnostic_campaign_sha256=hashlib.sha256(terminal.read_bytes()).hexdigest())
except BaseException as exc:metadata.update(status='stopped' if stopping else 'failed',error=repr(exc));raise
finally:metadata['finished_unix']=time.time();flush()
