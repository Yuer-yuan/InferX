"""Read-only record of a concurrent external campaign; do not stop or restore it."""
import json,subprocess,time,urllib.request
from pathlib import Path
r=Path('/home/guokc/mypro/gnn-mount-sda1/InferX');runs=[]
for p in sorted((r/'pd-baselines/raw').glob('pd-strategies-solo-*/campaign.json')):
 c=json.loads(p.read_text());original=json.loads((p.parent/'original-service.json').read_text());pid=c['controller_pid'];proc=Path(f'/proc/{pid}')
 runs.append(dict(run=p.parent.name,status=c['status'],controller_pid=pid,controller_exists=proc.exists(),controller_argv=(proc/'cmdline').read_bytes().decode().rstrip('\0').split('\0') if proc.exists() else None,original_service_pid=original['pid'],completed_groups=sum(s.get('state')=='measured' for s in c['cases']),current_groups=[{k:s.get(k) for k in ['case','group','state']} for s in c['cases'][-2:]]))
ports={}
for port in [18080,18081,18083]:
 try:
  with urllib.request.urlopen(f'http://127.0.0.1:{port}/v1/models',timeout=3) as response:ports[str(port)]=dict(status='responding',models=json.load(response))
 except Exception as exc:ports[str(port)]=dict(status='unavailable',error=str(exc))
a=json.loads((r/'tuning/best-service.json').read_text());report=dict(observed_unix=time.time(),read_only=True,interpretation='A separately launched solo campaign now holds GPU1. Do not interfere with its process or restore over it; earlier restoration verification remains a historical observation, not current endpoint availability.',manifest_equal=a==json.loads((r/'context-agents/runtime.json').read_text()),manifest_pid=a['pid'],manifest_process_exists=Path(f'/proc/{a["pid"]}').exists(),external_campaigns=runs,ports=ports,gpu_processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,process_name','--format=csv,noheader'],text=True))
(r/'pd-strategies/final-external-service-state.json').write_text(json.dumps(report,indent=2)+'\n');print('recorded external campaign state without signalling',[(x['run'],x['status']) for x in runs])
