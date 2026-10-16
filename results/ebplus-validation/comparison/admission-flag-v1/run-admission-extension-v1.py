"""Sequential owned HDD-only admission calibration/diagnostics; never launches formal work."""
import hashlib,json,os,subprocess,time,traceback
from pathlib import Path
r=Path(os.environ['INFERX_ROOT']);b=r/'ebplus-validation';code=b/'code-task3-v14';out=b/'admission-flag-v1';status=out/'extension-status.json'
meta=dict(pid=os.getpid(),status='preflight',started_unix=time.time(),actions=[],formal_actions=False)
def flush():
 p=status.with_suffix('.tmp');p.write_text(json.dumps(meta,indent=2)+'\n');p.replace(status)
def run(label,args,python='repo-venv/bin/python'):
 entry=dict(action=label,command=[str(r/python),*args],state='running',started_unix=time.time());meta['actions'].append(entry);flush()
 with (out/(label+'.log')).open('x') as log:
  p=subprocess.Popen(['taskset','-c','40-43,96-99',*entry['command']],cwd=code,env={**os.environ,'PYTHONPATH':str(code)},stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  entry['child_pid']=p.pid;flush();result=p.wait()
 entry.update(state='complete' if result==0 else 'failed',exit_code=result,finished_unix=time.time());flush()
 if result:raise RuntimeError(label+' failed; no dependent GPU launch')
try:
 assert not status.exists();flush()
 assert json.loads((out/'native-red-status.json').read_text())['status']=='expected-red-observed'
 assert (out/'native-admission-green.log').read_text().rstrip().endswith('OK')
 assert 'Ran 14 tests' in (out/'native-admission-green.log').read_text()
 assert '322 passed, 14 skipped' in (out/'full-cpu-green.log').read_text()
 prior=json.loads((b/'raw/diagnostic-v2/campaign.json').read_text());assert prior['status']=='restored' and prior['restored'] is True
 assert len(prior['cases'])==6 and all(s['state']=='measured' for s in prior['cases'])
 assert json.loads((b/'diagnostic-followup-v2.json').read_text())['status']=='ready-for-formal-review'
 assert (b/'diagnostic-export-v2-a.json').read_bytes()==(b/'diagnostic-export-v2-b.json').read_bytes()
 assert json.loads((b/'overlays/adapter-v6/manifest.json').read_text())['runtime_evidence_version']==2
 for name in ['mb-admission-false-v1','diagnostic-admission-false-v1']:assert not (b/'raw'/name).exists()
 meta['status']='mb-calibration';flush()
 run('mb-grid',['-u','scripts/run_ebplus_validation.py','--run-id','mb-admission-false-v1','--stage','mb-grid','--plan','configs/ebplus-admission-calibration.json','--prerequisite',str(b/'raw/diagnostic-v2')])
 mb=b/'raw/mb-admission-false-v1';s=json.loads((mb/'campaign.json').read_text());assert s['status']=='restored' and s['restored'] is True
 d=json.loads((mb/'mb-grid-result.json').read_text());assert d['status']=='complete' and len(d['rows'])==54 and d['scheduler_reserve_full_isl'] is False
 from inferx.ebplus_comparison import validate_admission_profile
 validate_admission_profile(dict(scheduler_reserve_full_isl=False),d)
 meta['status']='admission-diagnostic';flush()
 run('diagnostic',['-u','scripts/run_ebplus_validation.py','--run-id','diagnostic-admission-false-v1','--stage','compare','--plan','configs/ebplus-validation.json','--prerequisite',str(mb),'--hardware-run',str(b/'raw/hardware-v4'),'--mb-run',str(mb),'--overlay',str(b/'overlays/adapter-v6'),'--comparison','configs/ebplus-admission-diagnostic.json'])
 diagnostic=b/'raw/diagnostic-admission-false-v1';d=json.loads((diagnostic/'campaign.json').read_text())
 assert d['status']=='restored' and d['restored'] is True and len(d['cases'])==4 and all(s['state']=='measured' for s in d['cases'])
 meta['status']='strict-double-export';flush()
 for suffix in ['a','b']:
  run('export-'+suffix,['-m','scripts.analyze_ebplus_comparison','--frozen',str(r/'concurrency-roofline'),'--mixed',str(diagnostic),'--output',str(out/('diagnostic-export-'+suffix+'.json'))])
 raw=(out/'diagnostic-export-a.json').read_bytes();assert raw==(out/'diagnostic-export-b.json').read_bytes()
 report=json.loads(raw)
 meta.update(status='ready-for-joint-selection',export_sha256=hashlib.sha256(raw).hexdigest(),joint_metrics=[{k:g[k] for k in ['case','group','short_p95_s','long_s','output_tokens_per_s','strict_requests']} for g in report['mixed']['groups']])
except BaseException as exc:meta.update(status='failed',error=repr(exc),traceback=traceback.format_exc());raise
finally:meta['finished_unix']=time.time();flush()
