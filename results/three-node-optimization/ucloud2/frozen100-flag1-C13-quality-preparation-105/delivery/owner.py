"""Single frozen100 e4ed/c976 flag1 original100 C13 quality; explicit Root execution release required."""
import pathlib,json,hashlib,os,subprocess,time,datetime,traceback,socket,signal,urllib.request,argparse
R=pathlib.Path(__file__).resolve().parent;C=R.parent;J=C/'jobs'/(R.name+'.json');S=R/'service';PORT=18505;EVAL='/workspace/InferX/eval-envs/evalscope-1.13.0/bin/python'
parser=argparse.ArgumentParser();parser.add_argument('--release',type=pathlib.Path,required=True);args=parser.parse_args()
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def write(p,v):
 t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(v,indent=2,ensure_ascii=False)+'\n');os.replace(t,p)
def ident(pid):
 p=pathlib.Path('/proc')/str(pid);return dict(pid=pid,start=(p/'stat').read_text().rsplit(')',1)[1].split()[19],uid=p.stat().st_uid,exe=os.readlink(p/'exe'))
state=dict(owner=ident(os.getpid()),host='ucloud2',scope='frozen100 e4ed/c976 flag1 originalfirst100 C13 quality',task_precision_qualified=False);service=None;service_id=None;client=None;cid=None
def save(phase,**kw):state.update(phase=phase,updated_at=now(),**kw);write(J,state);write(R/'job.json',state);print(json.dumps(state),flush=True)
def idle():
 q=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],text=True);assert not q.strip(),q
 with socket.socket() as s:assert s.connect_ex(('127.0.0.1',PORT))!=0,'port occupied'
 return dict(at=now(),compute_processes=q)
def run(cmd,name,env):
 write(R/(name+'.command.json'),cmd)
 with (R/(name+'.log')).open('xb') as f:
  p=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT,cwd=R,env=env);i=ident(p.pid);save(name,active_child=i);rc=p.wait()
 write(R/(name+'.terminal.json'),dict(identity=i,returncode=rc,reaped=True,at=now()));save(name+'_terminal',active_child=None);assert rc==0,(name,rc)
def stop():
 global service
 if service is None:return
 if service.poll() is None:
  fd=os.pidfd_open(service.pid)
  try:assert ident(service.pid)==service_id;signal.pidfd_send_signal(fd,signal.SIGTERM)
  finally:os.close(fd)
  try:service.wait(timeout=30)
  except subprocess.TimeoutExpired:
   fd=os.pidfd_open(service.pid)
   try:assert ident(service.pid)==service_id;signal.pidfd_send_signal(fd,signal.SIGKILL)
   finally:os.close(fd)
   service.wait(timeout=15)
 write(S/'service-terminal.json',dict(identity=service_id,returncode=service.returncode,reaped=True,at=now()));service=None;save('owned_service_reaped',service=None)
 for _ in range(30):
  try:write(S/'GPU-released.json',idle());return
  except AssertionError:time.sleep(1)
 raise RuntimeError('GPU/port not released')
try:
 release=json.loads(args.release.read_text());expected=json.loads((R/'expected-runtime.json').read_text())
 assert release['execute_quality_after_lowC_complete'] is True and release['binary_sha256']==expected['binary_sha256'] and release['source_aggregate_sha256']==expected['source_aggregate_sha256']
 J.parent.mkdir(parents=True,exist_ok=True);save('explicit_Root_quality_release_verified');write(R/'executed-release.json',release)
 for n,v in json.loads((R/'source-freeze.json').read_text())['files'].items():assert sha(R/n)==v['sha256'],n
 binary=pathlib.Path(expected['binary']);assert sha(binary)==expected['binary_sha256'];write(R/'actual-binary-identity.json',dict(path=str(binary),bytes=binary.stat().st_size,sha256=sha(binary)))
 qualification=json.loads((R/'qualification-evidence.json').read_text());assert qualification['binary']['sha256']==expected['binary_sha256'] and qualification['source_aggregate']==expected['source_aggregate_sha256'];assert sha(R/'qualification-evidence.json')==release['qualification_evidence_sha256']
 # Prior source/native-boundary evidence does not claim new task accuracy or actual KV532480 fit.
 lock=(C/'.ucloud2-performance-owner.lock').open('a');__import__('fcntl').flock(lock,__import__('fcntl').LOCK_EX|__import__('fcntl').LOCK_NB)
 write(R/'GPU-before.json',idle());gpu=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,name,memory.total,power.limit','--format=csv,noheader'],text=True);assert len(gpu.strip().splitlines())==1 and '450.00' in gpu;write(R/'actual-GPU.json',dict(csv=gpu))
 env=os.environ.copy()
 for k in list(env):
  if k.startswith(('NINFER_','CUDA_MPS_')):env.pop(k)
 env.update(json.loads((R/'client-environment.json').read_text()));env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
 run([EVAL,str(R/'preflight.py')],'preflight-execution',env)
 cfg=json.loads((R/'server-config.json').read_text());cmd=cfg['command'];assert cmd[0]==str(binary)
 for flag,value in [('--port','18505'),('--max-concurrency','13'),('--kv-capacity','532480'),('--draft-tokens','3'),('--max-context','65536'),('--prefill-chunk','2048')]:assert cmd[cmd.index(flag)+1]==value
 senv=env.copy();senv.update(cfg['environment']);actual={k:v for k,v in senv.items() if k.startswith('NINFER_')}
 assert actual['NINFER_SCHED_BOUNDED_DECODE_MEMBERSHIP']=='1' and actual['NINFER_PROMPT_LOOKUP_N5_PRIORITY']==actual['NINFER_PROMPT_LOOKUP_N5_HYBRID']=='0'
 assert not any(any(w in k for w in ['QUALITY','TRACE','CAPTURE','PROFILE','ROUTE_LOG','MIXED','ORDERED']) for k in actual if k!='NINFER_SCHED_TRACE')
 assert actual['NINFER_INPUT_S4']==actual['NINFER_S4_TARGET_FFN']==actual['NINFER_S4_OMIT_LEGACY_FFN']=='1'
 S.mkdir(exist_ok=False);write(S/'command.json',cmd);write(S/'environment.json',actual)
 with (S/'server.log').open('xb') as f:service=subprocess.Popen(cmd,env=senv,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
 service_id=ident(service.pid);write(S/'service-identity.json',service_id);save('same_binary_model_loading',service=service_id)
 deadline=time.monotonic()+900
 while time.monotonic()<deadline:
  assert service.poll() is None,'service died; no automatic reload'
  try:
   with urllib.request.urlopen(f'http://127.0.0.1:{PORT}/v1/models',timeout=2) as response:models=json.load(response)
   write(S/'models.json',models);break
  except (OSError,ValueError):time.sleep(1)
 else:raise RuntimeError('startup timeout; no model restart')
 proc=pathlib.Path('/proc')/str(service.pid);assert ident(service.pid)==service_id
 observed=dict(x.decode().split('=',1) for x in (proc/'environ').read_bytes().split(b'\0') if x.startswith(b'NINFER_'));observedcmd=[x.decode() for x in (proc/'cmdline').read_bytes().split(b'\0') if x];assert observed==actual and observedcmd==cmd
 write(S/'actual-process.json',dict(identity=service_id,environment=observed,command=observedcmd))
 native=[json.loads(l) for l in (S/'native.jsonl').read_text().splitlines() if l.strip()];startup=[x for x in native if x['event']=='server_start'];assert len(startup)==1 and not any(x['event']=='request_done' for x in native)
 engine=startup[0]['engine'];assert engine['max_concurrency']==13 and engine['kv_capacity']==532480 and engine['speculative_backend']=='mtp' and engine['speculative_draft_window']==3 and engine['cuda_graph'] is True;write(S/'actual-startup.json',startup[0])
 log=(S/'server.log').read_text();assert sum('S4_complete_cache_readback layer=' in line and 'INPUT_S4_complete_cache_readback layer=' not in line for line in log.splitlines())==64 and log.count('INPUT_S4_complete_cache_readback layer=')==64
 with (R/'client.log').open('xb') as f:client=subprocess.Popen([EVAL,str(R/'recording_entry.py')],cwd=R,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
 cid=ident(client.pid);write(R/'client-identity.json',dict(identity=cid,log_path=str(R/'client.log')))
 while client.poll() is None:
  from checkpoints import snapshot
  snapshot(R)
  count=0
  for f in (R/'accuracy/native/predictions').rglob('*.jsonl'):
   for line in f.read_text().splitlines():
    try:json.loads(line);count+=1
    except ValueError:pass
  assert service.poll() is None,'service exited; preserve original client, no reload'
  save('original_first100_naturalEOS_collecting',completed=count,attempts=len(list((R/'http-attempts').glob('*/attempt.json'))),active_child=cid,service=service_id);time.sleep(20)
 from checkpoints import snapshot
 snapshot(R)
 client.wait();write(R/'client-terminal.json',dict(identity=cid,returncode=client.returncode,reaped=True,at=now()));save('original_client_naturally_terminal',active_child=None,client_returncode=client.returncode);stop()
 run([EVAL,str(R/'score_first100_cpu.py')],'original-all100-CPU-score',env)
 run([EVAL,str(R/'audit_http_native_join.py')],'original-all100-native-join',env)
 write(R/'route-evidence-scope.json',dict(source_aggregate_sha256=expected['source_aggregate_sha256'],binary_sha256=expected['binary_sha256'],same_generic_input_route_as_performance=True,quality_only_switch=False,bounded_decode_membership_enabled=True,trace='host scheduler rows only; no GPU profiler',exact_dynamic_route_counts_captured=False,note='Actual decode_round/row B from scheduler trace can establish B>8/T>32 execution with frozen K3 route plus frozen100 source and prior native boundary evidence. Capture-only host output-route logs are not used as execution proof; no perrequest kernel count fabricated.'))
 save('original_first100_complete_scored_GPU_released',active_child=None,service=None,score=json.loads((R/'CPU-score-001/summary.json').read_text())['metrics'])
except BaseException as e:
 error=dict(error=repr(e),traceback=traceback.format_exc(),at=now(),client_live=client is not None and client.poll() is None);write(R/'failure.json',error)
 if service is not None and not error['client_live']:stop()
 save('failed_preserved',**error);raise
