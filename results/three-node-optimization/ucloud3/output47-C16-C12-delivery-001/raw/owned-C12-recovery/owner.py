"""Unique current9270 C16, then C12 only if C16 fails the fixed SLO."""
import fcntl, hashlib, json, os, subprocess, time, traceback
from pathlib import Path
from owned_process import Owned, identity, write_json

R=Path(__file__).resolve().parent
BASE=Path('/workspace/InferX/experiments/ucloud3-c32-tonight')
RUNNER=Path('/workspace/InferX/code/submission/organizer/runner.py')
J=R/'job.json'
load=lambda p:json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb')as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def rows(p):
 if not p.exists():return []
 lines=p.read_text().splitlines();out=[]
 for i,line in enumerate(lines):
  try:out.append(json.loads(line))
  except json.JSONDecodeError:
   if i!=len(lines)-1:raise
 return out
def missing(h):
 try:return Path('/proc',str(h['pid']),'stat').read_text().rsplit(')',1)[1].split()[19]!=str(h['start_ticks'])
 except(FileNotFoundError,ProcessLookupError):return True
def idle():
 apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader'],text=True)
 assert not apps.strip(),apps
def percentile(values,q):
 a=sorted(values);i=(len(a)-1)*q;lo=int(i);return a[lo]+(a[min(lo+1,len(a)-1)]-a[lo])*(i-lo)
assert not J.exists()
lock=(R.parent/'.ucloud3-performance-owner.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
j=dict(status='running',stage='preflight',owner=identity(os.getpid()),handles=[],runs=[],started_ns=time.time_ns(),scope='original runner C16 then conditionalC12,128+8+1, fixed9270/source91621; oldT4 protected failure disclosed; no quality claim')
driver=telemetry=None
def save(stage=None,**kw):
 if stage:j['stage']=stage
 j.update(kw,heartbeat_ns=time.time_ns());write_json(J,j)
try:
 save()
 frozen=load(R/'identity.json');assert all(sha(R/k)==v for k,v in frozen['files'].items())
 assert sha(RUNNER)=='d80df390cf6d1d50129796b31fdd7f4d1e95d0043d3b942d47ac582118702fb0'
 prior=load(R.parent/'output47-current-official100-001/job.json');assert prior['status']=='failed'and prior['error']=='KeyboardInterrupt()'
 assert prior['client_cleanup']['reaped']and prior['service_cleanup']['reaped']
 assert all(missing(h)for h in [prior['owner']]+prior['handles'])
 cancel=R.parent/'output47-current-official100-001/actual-user-cancel-terminal.json';assert load(cancel)['disposition']=='USER_CANCELLED_PERFORMANCE_PRIORITY_NOT_COMPLETED'
 previous_perf=load(R.parent/'output47-C16-C12-official128-001/job.json')
 assert previous_perf['status']=='failed'and previous_perf['runs'][0]['status']=='complete'and previous_perf['runs'][0]['joint_PASS']is False
 assert previous_perf['runs'][0]['all137cold']
 assert previous_perf['runs'][1]['native_done']==0 and 'service'not in previous_perf['runs'][1]
 assert all(missing(h)for h in [previous_perf['owner']]+previous_perf['handles']+[previous_perf['runs'][0]['service']])
 failed_manifest=load('/workspace/InferX/runs/ucloud3-slo/output47-C12-K3-n128-r1/manifest.json')
 assert failed_manifest['cases'][0]['cleanup']['note']=='no server started'and failed_manifest['cases'][0]['tiers']==[]
 j['recovery_scope']='Port preflight-only C12 failure; no previous C12 service/HTTP. Keep complete C16 SLO FAIL, no rerun.'
 relay=load(R/'runtime-ready-compat.json')
 assert relay['binary_sha256']=='9270f73e92409bc94e4b8223adb755c8b1527e611eec21e1ef8778482d0b74c2'
 assert relay['source_sha256']=='91621a3c46a4c901a51863442704ea7c398ebdfb73ade9f3b1c4afac4ab235cc'
 for f in relay['files']:assert sha(f['path'])==f['sha256']
 binary=Path(relay['files'][0]['path'])
 seam=load(R/'model-seam-qualification.json');assert seam['status']=='SELECTED_ROUTE_PASS_OLD_T4_FAILURE_DISCLOSED'
 assert seam['binary_sha256']==relay['binary_sha256']and seam['source_sha256']==relay['source_sha256']
 assert seam['registered_selected_layers']==[i for i in range(64)if i>0 and(i+1)%4!=0]
 assert seam['model_owned_registration']and seam['actual_Graph_seam']
 for f in seam['evidence_files']:assert sha(f['path'])==f['sha256']
 assert all(missing(h)for h in seam['handles'])
 idle()
 gpu=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,memory.used,power.limit,driver_version','--format=csv,noheader'],text=True)
 assert '450.00 W'in gpu and'GPU-ef83db6a-8dad-6be1-ba4e-38c279abd877'in gpu
 for p in Path('/proc').glob('[0-9]*/comm'):
  try:assert p.read_text().strip()not in ['nvcc','ptxas','cicc','cc1plus','ninja','nsys','ncu','ninfer-serve'],'contending build/GPU job'
  except(FileNotFoundError,ProcessLookupError):pass
 runtime=dict(binary=str(binary),binary_sha256=relay['binary_sha256'],source_sha256=relay['source_sha256'])
 write_json(R/'runtime-identity.json',runtime)
 evidence={n:dict(path=str(p),sha256=sha(p))for n,p in [('runtime',R/'runtime-ready-compat.json'),('selected_model_seam',R/'model-seam-qualification.json'),('authorization',R/'authorization.json'),('quality_cancel_terminal',cancel)]}
 for name,kind in [('qualification.json','native32_s4_startup_qualification'),('numeric-ready.json','frozen_candidate_numeric_qualification')]:
  write_json(R/name,dict(kind=kind,passed=True,**runtime,evidence=evidence,scope='Root-authorized selected47 model-owned L1T1/L2T2044 PASS only; original oldT4 parity FAILED and separately diagnosed. Reuse exact9270 build/registration/helper gates, no full task quality claim.'))
 env=os.environ.copy()
 for k in list(env):
  if k.startswith(('NINFER_','CUDA_MPS_','UCLOUD2_','INFERX_OUTPUT_CAPTURE_')):env.pop(k)
 expected=load(R/'environment.json');env.update(expected,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',LD_LIBRARY_PATH=relay['LD_LIBRARY_PATH'])
 assert not any('TRACE'in k or 'LOG'in k for k in expected)
 save('ready',GPU_preflight=gpu,runtime=runtime,expected_env=expected)
 for c,kv in [(12,104448)]:
  idle();cfg=load(R/f'C{c}-config.json');argv=cfg['cases'][0]['command'];value=lambda f:argv[argv.index(f)+1]
  assert cfg['requests']==128 and cfg['warmup']==8 and cfg['seed']==20261016 and cfg['cases'][0]['concurrencies']==[c]
  for f,v in [('--max-concurrency',str(c)),('--kv-capacity',str(kv)),('--draft-tokens','3'),('--prefill-chunk','2048'),('--max-context','65536')]:assert value(f)==v
  assert argv[0]==str(binary)and '--no-prefix-reuse'in argv and '--no-disk-cache'in argv
  assert '--no-thinking'not in argv and '--greedy'not in argv
  run=Path(f'/workspace/InferX/runs/ucloud3-slo/output47-C{c}-K3-n128-recovery002');assert not run.exists()
  rr=dict(concurrency=c,KV=kv,run_dir=str(run),started_ns=time.time_ns(),status='running');j['runs'].append(rr)
  telemetry=Owned(['nvidia-smi','--query-gpu=timestamp,uuid,memory.used,power.draw,clocks.sm,temperature.gpu','--format=csv','-l','1'],R/f'C{c}-gpu.csv',role='1Hz power/memory, not occupancy')
  driver=Owned(['/usr/bin/python3',str(R/'run_candidate_sweep.py'),'--runner',str(RUNNER),'--config',str(R/f'C{c}-config.json'),'--output-dir',str(run)],R/f'C{c}-driver.log',env=env,role=f'original128+8+1 C{c}')
  j['handles'] +=[telemetry.record,driver.record];save(f'C{c}_startup_and_original128')
  while driver.child.poll()is None:
   mf=run/'manifest.json'
   if mf.exists():
    m=load(mf)
    if m.get('cases'):
     case=m['cases'][0];rr['case_status']=case['status'];rr['tiers']=case.get('tiers',[])
     if 'pid'in case and 'service'not in rr:
      p=Path('/proc',str(case['pid']))
      try:
       assert p.joinpath('stat').read_text().rsplit(')',1)[1].split()[19]==str(case['start_ticks'])
       actual=dict(x.decode().split('=',1)for x in p.joinpath('environ').read_bytes().split(b'\0')if x.startswith(b'NINFER_'))
       command=[x.decode()for x in p.joinpath('cmdline').read_bytes().split(b'\0')if x]
       assert actual==expected and command==argv and p.joinpath('exe').resolve()==binary.resolve()
       rr.update(service=dict(pid=case['pid'],start_ticks=case['start_ticks']),actual_env=actual,actual_argv=command)
      except(FileNotFoundError,ProcessLookupError):pass
   native=rows(Path(value('--request-log-jsonl')));rr['native_done']=sum(x.get('event')=='request_done'for x in native)
   starts=[x for x in native if x.get('event')=='server_start']
   if starts:rr['startup']=starts[0]
   save();time.sleep(2)
  rc=driver.child.returncode;rr['driver_cleanup']=driver.stop();driver=None;rr['telemetry_cleanup']=telemetry.stop();telemetry=None
  m=load(run/'manifest.json');rr['manifest_status']=m['status'];save()
  assert rc==0 and m['status']=='complete','infrastructure/runtime failure preserved; no silent retry'
  case=m['cases'][0];tier=case['tiers'][0];rr['tier']=tier;assert missing(dict(pid=case['pid'],start_ticks=case['start_ticks']))
  native=rows(Path(value('--request-log-jsonl')));done=[x for x in native if x.get('event')=='request_done'];errors=[x for x in native if x.get('event')=='request_error']
  exact_native=(len(done)==137 and not errors and sorted(x['result']['completion_tokens']for x in done)==[1]+[512]*136 and all(x['result']['prompt_tokens']==x['result']['computed_prefill_tokens']==8192 and x['result']['prefix_cache_hit_tokens']==0 and x['result']['prefix_reuse_path']=='full_reset'and x['result']['finish_reason']=='output_limit'for x in done))
  raw=[x for x in rows(run/f'candidate/c{c}/profile_export.jsonl')if x['metadata']['benchmark_phase']=='profiling'];clients=[]
  for rank,x in enumerate(sorted(raw,key=lambda x:x['metadata']['request_start_ns']),1):
   mm=x.get('metrics',{});tt=mm.get('time_to_first_token',{}).get('value');lat=mm.get('request_latency',{}).get('value')
   if tt is not None and lat is not None:
    tp=(lat-tt)/511;assert abs(tp-mm['inter_token_latency']['value'])<1e-8
    clients.append(dict(arrival_rank=rank,TTFT_ms=tt,TPOT_ms=tp,start_ns=x['metadata']['request_start_ns']))
  exact_client=len(clients)==128 and tier.get('error_requests')==0 and tier.get('lengths_verified')and tier.get('records')==tier.get('valid_records')==tier.get('successful_requests')==128
  measured=dict(TTFT_p95_ms=percentile([x['TTFT_ms']for x in clients],.95),TPOT_p95_ms=percentile([x['TPOT_ms']for x in clients],.95))if clients else {}
  if len(clients)==128:
   assert abs(measured['TTFT_p95_ms']-tier['metrics']['time_to_first_token']['p95'])<1e-6
   assert abs(measured['TPOT_p95_ms']-tier['metrics']['inter_token_latency']['p95'])<1e-6
  passed=bool(exact_native and exact_client and measured['TTFT_p95_ms']<=10000 and measured['TPOT_p95_ms']<=40)
  rr.update(status='complete',joint_PASS=passed,independent_metrics=measured,all137cold=bool(exact_native),all_startup_retained=True,native_errors=len(errors),client_rows=clients,TTFT_violation_ranks=[x['arrival_rank']for x in clients if x['TTFT_ms']>10000],TPOT_violation_ranks=[x['arrival_rank']for x in clients if x['TPOT_ms']>40],finished_ns=time.time_ns())
  idle();save(f'C{c}_complete')
 assert sha(binary)==runtime['binary_sha256']and all(sha(R/k)==v for k,v in frozen['files'].items())
 save('COMPLETE_C12_RECOVERY_NO_C16_RERUN',status='complete',finished_ns=time.time_ns())
except BaseException as e:
 save('FAILURE_preserved',status='failed',error=repr(e),traceback=traceback.format_exc(),finished_ns=time.time_ns());raise
finally:
 if driver:j['exception_driver_cleanup']=driver.stop(timeout=65)
 if telemetry:j['exception_telemetry_cleanup']=telemetry.stop()
 save()
