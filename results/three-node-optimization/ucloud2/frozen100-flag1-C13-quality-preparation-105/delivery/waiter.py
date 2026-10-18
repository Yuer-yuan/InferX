"""Explicit Root release, strict dependency on the original106 owner; no polling HTTP."""
from pathlib import Path
import os,json,time,hashlib,subprocess,traceback
from owned_process import identity,same,Owned,write_json
R=Path(__file__).resolve().parent;P=R.parent/'W2-K3-C14-native13-perf-106';J=R/'queue-job.json';assert not J.exists()
j=dict(status='running',stage='waiting_original106_natural_terminal',owner=identity(os.getpid()),handles=[],started_ns=time.time_ns());write_json(J,j);active=None
try:
 for n,h in json.loads((R/'execution-freeze.json').read_text())['files'].items():assert hashlib.sha256((R/n).read_bytes()).hexdigest()==h,n
 cfg=json.loads((R/'server-config.json').read_text());assert not any(k.startswith('NINFER_VERIFY_W2') for k in cfg['environment']);assert cfg['environment']['NINFER_SCHED_BOUNDED_DECODE_MEMBERSHIP']=='1'
 while True:
  parent=json.loads((P/'job002.json').read_text());j['dependency']={k:parent.get(k) for k in ['status','stage','owner']};j['heartbeat_ns']=time.time_ns();write_json(J,j)
  if parent['status']!='running'and not same(parent['owner']):break
  time.sleep(10)
 assert parent['status']=='complete','106 failed; retain all files and stop queue without quality requests'
 assert all(not same(h) for h in parent['handles'])
 for run in parent['runs']:
  h=run.get('service');assert h is not None
  p=Path('/proc')/str(h['pid'])/'stat'
  if p.exists():assert p.read_text().rsplit(')',1)[1].split()[19]!=str(h['start_ticks']),'original106 service still alive'
 apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],text=True);assert not apps.strip(),apps
 j['GPU_idle_before_quality']=True;j['stage']='quality_owner_starting';write_json(J,j)
 active=Owned(['/usr/bin/python3',str(R/'owner.py'),'--release',str(R/'Root-execution-release.json')],R/'owner.log',role='frozen100 default first100 quality owner')
 j['handles'].append(active.record);j['stage']='quality_owner_running';write_json(J,j)
 while active.child.poll()is None:j['heartbeat_ns']=time.time_ns();write_json(J,j);time.sleep(10)
 rc=active.child.returncode;j['owner_terminal']=active.stop();active=None;assert rc==0,'quality owner failure preserved; no reload/retry'
 state=json.loads((R/'job.json').read_text());assert state['phase']=='original_first100_complete_scored_GPU_released'
 j.update(status='complete',stage='quality_complete_scored_terminal',finished_ns=time.time_ns());write_json(J,j)
except BaseException as e:j.update(status='failed_preserved',error=repr(e),traceback=traceback.format_exc(),finished_ns=time.time_ns());write_json(J,j);raise
finally:
 if active:j['exception_cleanup']=active.stop(timeout=60)
 write_json(J,j)
