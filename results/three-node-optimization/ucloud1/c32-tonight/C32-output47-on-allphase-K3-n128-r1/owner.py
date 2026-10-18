import hashlib,json,os,time,subprocess,traceback
from pathlib import Path
from owned_process import Owned,identity,write_json,same
R=Path(__file__).resolve().parent;BASE=R.parent;BUILD=BASE/'output47-prefill-on-allphase-r1';RUN=Path('/workspace/InferX/runs/ucloud1-slo/c32-tonight-output47-on-allphase-K3-n128-r1');J=R/'job.json';RUNNER=Path('/workspace/InferX/code/submission/organizer/runner.py')
load=lambda p:json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb')as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def rows(p):
 if not p.exists():return []
 out=[];ll=p.read_text().splitlines()
 for i,l in enumerate(ll):
  try:out.append(json.loads(l))
  except json.JSONDecodeError:
   if i!=len(ll)-1:raise
 return out
def missing(h):
 try:return Path('/proc/'+str(h['pid'])+'/stat').read_text().rsplit(')',1)[1].split()[19]!=str(h['start_ticks'])
 except(FileNotFoundError,ProcessLookupError):return True
assert not J.exists()and not RUN.exists();j=dict(status='running',stage='preflight',owner=identity(os.getpid()),handles=[],started_ns=time.time_ns(),scope='C32/K3 fixed47 Prefill output S4 atop qualified40b3; fixed128+8 official runner; exact model-owned seam required; no rejected gate changes');write_json(J,j);driver=telemetry=None
try:
 f=load(R/'bundle-identity.json');assert all(sha(R/n)==v for n,v in f['files'].items())
 built=load(BUILD/'build-state.json');assert built['status']=='complete'and built['source_unchanged']and all(missing(h)for h in[built['owner']]+built['handles'])
 assert all(x['cleanup']['reaped']and x['cleanup']['exit_code']==0 for x in built['steps'])
 seams=load(R/'model-seam-qualification.json')
 assert seams['kind']=='root_authorized_selected47_execution'and seams['passed']and seams['selected_child_reaped']
 assert seams['binary_sha256']==built['binary_sha256']=='9270f73e92409bc94e4b8223adb755c8b1527e611eec21e1ef8778482d0b74c2'
 assert seams['source_sha256']=='91621a3c46a4c901a51863442704ea7c398ebdfb73ade9f3b1c4afac4ab235cc'
 assert all(sha(v['path'])==v['sha256']for v in seams['evidence'].values())
 previous=load(BASE/'C32-input-allphase-K3-n128-r1/job.json');assert previous['status']=='complete'and all(missing(h)for h in[previous['owner']]+previous['handles']+[previous['service']])
 j['model_seam_receipt']=seams;j['protected_T4_diagnostic_pending']=True;j['task_quality_qualified']=False
 sid=load(BUILD/'source-identity.json');pre={str(p.relative_to(BUILD/'source')):sha(p)for p in sorted((BUILD/'source').rglob('*'))if p.is_file()};assert pre==sid['files']==load(BUILD/'source-pre.json')==load(BUILD/'source-post.json');write_json(R/'source-pre.json',pre)
 binary=BUILD/'ninfer-serve';assert sha(binary)==built['binary_sha256'];assert sha(RUNNER)=='d80df390cf6d1d50129796b31fdd7f4d1e95d0043d3b942d47ac582118702fb0'
 ready=load(BASE/'weights025/cache025-ready.json');assert ready['status']=='complete'and ready['files_verified']==256 and ready['bytes_verified']==8690073600;assert all(missing(h)for h in[ready['owner']]+ready['children'])
 cachemanifest=load(BASE/'cache025-transfer-control/manifest.json');ff=[v for m in cachemanifest['matrices']for v in m['files']];assert len(ff)==256 and sum(v['bytes']for v in ff)==8690073600;assert all((BASE/'weights025'/v['relative_path']).stat().st_size==v['bytes']for v in ff)
 j['cache_ready_receipt']=ready;j['cache_manifest_sha256']=sha(BASE/'cache025-transfer-control/manifest.json');j['cache_SHA_scope']='Root direct transfer already verified all256 destination hashes; this preflight rechecks file count and exact sizes, preserving original receipt.'
 assert (BASE/'weights041/READY').exists()
 j['input_cache_ready_sha256']=sha(BASE/'weights041/READY.json')
 j['input_cache_manifest_sha256']=sha(BASE/'weights041/all64-cache-manifest.json')
 output_ready=load(BASE/'weights047/READY.json');assert output_ready['status']=='complete'and output_ready['weight_files_verified']==128 and output_ready['weight_bytes_verified']==1022361600
 assert all(missing(h)for h in[output_ready['owner']]+output_ready['children'])
 output_manifest=load(BASE/'weights047/all64-cache-manifest.json');selected=[m for m in output_manifest['cache']if m['layer']>0 and(m['layer']+1)%4!=0];assert len(selected)==47
 selected_files=[v for m in selected for v in m['files']];assert len(selected_files)==94 and sum(v['bytes']for v in selected_files)==750796800
 assert all((BASE/'weights047'/v['relative_path']).stat().st_size==v['bytes']for v in selected_files)
 j['output_cache_ready']=output_ready;j['output_manifest_sha256']=sha(BASE/'weights047/all64-cache-manifest.json');j['selected_layers']=[m['layer']for m in selected]

 assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
 for p in Path('/proc').glob('[0-9]*/comm'):
  try:assert p.read_text().strip()not in['nvcc','ptxas','cicc','cc1plus','ninja','nsys','ncu','pigz','gzip','xz','ninfer-serve'],'contention'
  except(FileNotFoundError,ProcessLookupError):pass
 gpu=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,memory.used,power.limit','--format=csv,noheader'],text=True);assert '450.00'in gpu and 'GPU-e7536e1c-450e-152d-e5e2-92c032a1c353'in gpu
 runtime=dict(binary=str(binary),binary_sha256=built['binary_sha256'],source_sha256=sid['aggregate_sha256']);write_json(R/'runtime-identity.json',runtime)
 evidence={n:dict(path=str(p),sha256=sha(p))for n,p in[('build',BUILD/'build-state.json'),('source',BUILD/'source-identity.json'),('cache',BASE/'weights025/cache025-ready.json'),('model_owned_47_seam',R/'model-seam-qualification.json')]};write_json(R/'qualification.json',dict(kind='native32_s4_startup_qualification',passed=True,**runtime,evidence=evidence,scope='Only build/CPU/cache are qualified here. Separate exact numeric receipt required before any model HTTP; startup diagnostic not a mathematical or quality claim.'))
 write_json(R/'numeric-ready.json',dict(kind='frozen_candidate_numeric_qualification',passed=True,**runtime,evidence=evidence,scope='Frozen025/041 parent retained; exact9270 selected L1T1/L2T2044 model-owned registration/Prefill/Graph reported PASS by ucloud3 and accepted by Root. Protected oldT4 parity failed and diagnostic remains pending; Root explicitly permits this performance run. No task-quality claim.'))
 cfg=load(R/'config.json');argv=cfg['cases'][0]['command'];value=lambda f:argv[argv.index(f)+1];assert cfg['requests']==128 and cfg['warmup']==8 and cfg['seed']==20261016 and cfg['cases'][0]['concurrencies']==[32]
 for f,v in[('--max-concurrency','32'),('--draft-tokens','3'),('--kv-capacity','278528'),('--prefill-chunk','2048'),('--max-context','65536')]:assert value(f)==v
 assert '--no-prefix-reuse'in argv and '--no-disk-cache'in argv and '--no-thinking'not in argv and '--greedy'not in argv
 env=os.environ.copy()
 for k in list(env):
  if k.startswith(('NINFER_','CUDA_MPS_','UCLOUD2_','INFERX_OUTPUT_CAPTURE_')):env.pop(k)
 expected=dict(NINFER_PREFILL_A8='all',NINFER_VERIFY_Q5_A8='1',NINFER_SCHED_POLICY='completion_queue',NINFER_SCHED_TTFT_MARGIN_MS='300',NINFER_PREFILL_FULLK='s4gptq64ffn',NINFER_S4_GPTQ_CACHE=str(BASE/'weights025'),NINFER_S4_TARGET_FFN='1',NINFER_S4_OMIT_LEGACY_FFN='1',NINFER_INPUT_S4='1',NINFER_INPUT_S4_CACHE=str(BASE/'weights041'),NINFER_OUTPUT_S4='1',NINFER_OUTPUT_S4_CACHE=str(BASE/'weights047'));env.update(expected,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',LD_LIBRARY_PATH='/workspace/InferX/envs/build/lib:/usr/local/cuda-13.2/lib64');j.update(runtime=runtime,expected_env=expected,GPU_preflight=gpu,command=argv)
 telemetry=Owned(['nvidia-smi','--query-gpu=timestamp,uuid,memory.used,power.draw,clocks.sm,temperature.gpu','--format=csv','-l','1'],R/'gpu.csv',role='memory/power1Hz not occupancy');driver=Owned(['/usr/bin/python3',str(R/'run_candidate_sweep.py'),'--runner',str(RUNNER),'--config',str(R/'config.json'),'--output-dir',str(RUN)],R/'driver.log',env=env,role='C32 startup then exact GPU gate then original128+8');j['handles']+=[telemetry.record,driver.record];j['stage']='startup_or_wait_numeric';write_json(J,j)
 while driver.child.poll()is None:
  mf=RUN/'manifest.json'
  if mf.exists():
   m=load(mf)
   if m.get('cases'):
    case=m['cases'][0];j['case_status']=case['status'];j['numeric_gate']=case.get('numeric_gate');j['tiers']=case.get('tiers',[])
    if 'pid'in case and'actual_proc_env'not in j:
     p=Path('/proc/'+str(case['pid']))
     try:
      assert p.joinpath('stat').read_text().rsplit(')',1)[1].split()[19]==str(case['start_ticks']);actual=dict(x.decode().split('=',1)for x in p.joinpath('environ').read_bytes().split(b'\0')if x.startswith(b'NINFER_'));command=[x.decode()for x in p.joinpath('cmdline').read_bytes().split(b'\0')if x];assert actual==expected and command==argv and p.joinpath('exe').resolve()==binary.resolve();j.update(service=dict(pid=case['pid'],start_ticks=case['start_ticks']),actual_proc_env=actual,actual_proc_argv=command)
     except(FileNotFoundError,ProcessLookupError):pass
  rr=rows(Path(value('--request-log-jsonl')));starts=[x for x in rr if x.get('event')=='server_start'];j['native_done']=sum(x.get('event')=='request_done'for x in rr);j['native_errors']=sum(x.get('event')=='request_error'for x in rr)
  if starts:j['startup']=starts[0]
  j['heartbeat_ns']=time.time_ns();write_json(J,j);time.sleep(1)
 rc=driver.child.returncode;j['driver_cleanup']=driver.stop();driver=None;j['telemetry_cleanup']=telemetry.stop();telemetry=None
 m=load(RUN/'manifest.json');j['manifest_status']=m['status'];j['case']=m['cases'][0];write_json(J,j);assert rc==0 and m['status']=='complete','first startup/performance failure preserved'
 case=m['cases'][0];tier=case['tiers'][0];j['tier']=tier;rr=rows(Path(value('--request-log-jsonl')));done=[x for x in rr if x.get('event')=='request_done'];assert len(done)==137 and not any(x.get('event')=='request_error'for x in rr);assert sorted(x['result']['completion_tokens']for x in done)==[1]+[512]*136
 assert all(x['result']['prompt_tokens']==x['result']['computed_prefill_tokens']==8192 and x['result']['prefix_cache_hit_tokens']==0 and x['result']['prefix_reuse_path']=='full_reset'and x['result']['finish_reason']=='output_limit'for x in done)
 assert all(x['request']['enable_thinking']is False and x['request']['message_count']==0 and x['request']['protocol']=='openai_completions'and x['request']['sampling']['temperature']==0 for x in done)
 assert tier['records']==tier['valid_records']==tier['successful_requests']==128 and tier['error_requests']==0 and tier['lengths_verified']
 clients=[x for x in rows(RUN/'candidate/c32/profile_export.jsonl')if x['metadata']['benchmark_phase']=='profiling'];assert len(clients)==128
 clients.sort(key=lambda x:x['metadata']['request_start_ns']);cr=[]
 for i,x in enumerate(clients,1):
  mm=x['metrics'];tt=mm['time_to_first_token']['value'];tp=(mm['request_latency']['value']-tt)/511;assert abs(tp-mm['inter_token_latency']['value'])<1e-8;cr.append(dict(arrival_rank=i,ttft_s=tt/1000,tpot_ms=tp,start_ns=x['metadata']['request_start_ns']))
 j.update(client_rows=cr,all137cold=True,all_startup_retained=True,TTFT_violations=[x['arrival_rank']for x in cr if x['ttft_s']>10],TPOT_violations=[x['arrival_rank']for x in cr if x['tpot_ms']>40],actual_batch_coverage_limit='Native cap32 and MTP stats retained; no exact B histogram is inferred from5second means. Startup Graph and operator gate are distinct from actual task coverage.')
 post={str(p.relative_to(BUILD/'source')):sha(p)for p in sorted((BUILD/'source').rglob('*'))if p.is_file()};assert pre==post and sha(binary)==runtime['binary_sha256'];write_json(R/'source-post.json',post)
 j.update(status='complete',stage='terminal',source_unchanged=True,finished_ns=time.time_ns())
except BaseException as e:j.update(status='failed',error=repr(e),traceback=traceback.format_exc(),finished_ns=time.time_ns());raise
finally:
 if driver:j['exception_driver_cleanup']=driver.stop(timeout=65)
 if telemetry:j['exception_telemetry_cleanup']=telemetry.stop()
 write_json(J,j)
