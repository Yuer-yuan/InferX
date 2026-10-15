"""Proceed only if the predeclared policy diagnostic jointly selects the ablation."""
import hashlib,json,os,signal,subprocess,time
from pathlib import Path
from inferx.pd_baselines import load_frozen_workloads
from scripts.analyze_sota_validation import export_campaign
r=Path(os.environ['INFERX_ROOT']);root=r/'morphx-control';path=root/'formal-followups-v1.json';assert not path.exists()
metadata=dict(status='waiting_for_policy',pid=os.getpid(),started_unix=time.time(),actions=[],rule='Selected both96mode16 must weakly dominate both author-cap settings on shortP95/long/throughput before automatic formal launch; otherwise stop for joint assessment.')
child=None;stopping=False
def flush():path.write_text(json.dumps(metadata,indent=2)+'\n')
def stop(number,frame):
 global stopping
 stopping=True
 if child is not None and child.poll() is None:child.send_signal(signal.SIGTERM)
for number in [signal.SIGINT,signal.SIGTERM,signal.SIGHUP]:signal.signal(number,stop)
def run(label,script,args=()):
 global child
 if stopping:raise InterruptedError('queue stopped before action')
 entry=dict(action=label,state='running',started_unix=time.time());metadata['actions'].append(entry);flush()
 child=subprocess.Popen([str(r/'repo-venv/bin/python'),'-u',str(root/script),*args],env={**os.environ,'PYTHONPATH':str(r/'code-pd-baselines')},stdin=subprocess.DEVNULL)
 entry['child_pid']=child.pid;flush()
 while True:
  try:code=child.wait(timeout=1);break
  except subprocess.TimeoutExpired:pass
 entry.update(exit_code=code,state='complete' if code==0 else 'failed',finished_unix=time.time());flush();child=None
 if stopping:raise InterruptedError('queue stopped after cleanup')
 if code!=0:raise RuntimeError(label+' failed')
def require_terminal(run_id,count):
 p=root/'raw'/run_id;d=json.loads((p/'campaign.json').read_text())
 assert d['status']=='restored' and d['restored'] and len(d['cases'])==count and all(x['state']=='measured' for x in d['cases'])
 return p
flush()
try:
 prerequisite=root/'raw/policy-local-v1/campaign.json'
 while True:
  if stopping:raise InterruptedError('queue stopped without signalling prerequisite')
  d=json.loads(prerequisite.read_text())
  if d['status']=='running':time.sleep(5);continue
  require_terminal('policy-local-v1',2);break
 run('audit-policy','analyze_control_diagnostics.py',['--run-id','policy-local-v1','--output',str(root/'policy-export.json')])
 templates,_=load_frozen_workloads(r/'concurrency-roofline')
 previous=export_campaign(require_terminal('transport-local-v1',5),templates,model_root=r/'models/qwen38-int4-redhat')
 policy=export_campaign(require_terminal('policy-local-v1',2),templates,model_root=r/'models/qwen38-int4-redhat')
 selected_id='morphx-control-both-96-ablation';selected=next(x for x in previous['groups'] if x['case']==selected_id)
 dominates=all(selected['short_p95_s']<=x['short_p95_s'] and selected['long_s']<=x['long_s'] and selected['output_tokens_per_s']>=x['output_tokens_per_s'] for x in policy['groups'])
 if not dominates:
  metadata.update(status='needs_joint_assessment',policy_rows=[{k:x[k] for k in ['case','short_p95_s','long_s','output_tokens_per_s']} for x in policy['groups']]);flush()
 else:
  freeze=dict(frozen_unix=time.time(),selected=selected_id,control='compiled-primary-control',rule=metadata['rule'],
   scope='Single diagnostic selection only, not a statistical difference between event/both or original-author ceiling.',
   source_sha256={str(p.relative_to(r)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [root/'diagnostic-export-1.json',root/'policy-export.json',root/'configs/morphx-local-cpu.json',root/'run_local_formal.py',Path(__file__)]})
  with (root/'selection.json').open('x') as f:json.dump(freeze,f,indent=2)
  metadata.update(status='running',selection_sha256=hashlib.sha256((root/'selection.json').read_bytes()).hexdigest());flush()
  cases=['compiled-primary-control',selected_id]
  run('formal-three-repeats','run_local_formal.py',['--run-id','formal-local-v1','--cases',*cases,'--groups','cold-high','resume-high','--repeat','3'])
  require_terminal('formal-local-v1',12)
  run('matched-serial-C4-solos','run_local_formal.py',['--run-id','solo-local-v1','--cases',*cases,'--groups','short-control','cold-high','resume-high','--solo'])
  require_terminal('solo-local-v1',6)
  metadata['status']='complete'
except BaseException as exc:metadata.update(status='stopped' if stopping else 'failed',error=repr(exc));raise
finally:metadata['finished_unix']=time.time();flush()
