from pathlib import Path
import os,sys,json,time,subprocess,traceback,fcntl,hashlib
from owned_process import Owned,identity,write_json
R=Path(__file__).resolve().parent;P=R/'entrypoint-r2';J=R/'quality-r2-job.json';assert not J.exists()
lock=Path('/workspace/InferX/experiments/u3-c12-c16-latency-20261018/.ucloud3-performance-owner.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
sys.path.insert(0,str(P));import candidate_common as common
binding=json.loads((R/'rebuilt-recipe-binding.json').read_text());recipe_path=P/'candidate_recipes/100-k3-u3-47-rebuilt-20261018.json';recipe,rsha=common.load_recipe(recipe_path)
E={k:v for k,v in os.environ.items()if not k.startswith(('NINFER_','CUDA_MPS_','UCLOUD2_','INFERX_OUTPUT_CAPTURE_'))};E.update(PATH=str(R/'tools')+':'+str(R/'envs/build/bin')+':'+E.get('PATH',''),LD_LIBRARY_PATH=str(R/'envs/build/lib')+':/usr/local/cuda-13.2/lib64',OPENAI_API_KEY='EMPTY',MODELSCOPE_CACHE=str(R/'cache/modelscope'),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
j=dict(status='running',stage='preflight',owner=identity(os.getpid()),handles=[],runs=[],started_ns=time.time_ns(),binding=binding);active=service=None

def save(stage=None):
 if stage:j['stage']=stage
 j['heartbeat_ns']=time.time_ns();write_json(J,j)
def idle():
 apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader'],text=True);assert not apps.strip(),apps
 return apps
def refresh(child):
 now=identity(child.child.pid)
 for k in ('pid','start_ticks','uid'):assert str(now[k])==str(child.record[k]),'owned child identity changed'
 child.record.update(now)
def child_run(name,args):
 global active
 active=Owned(args,R/(name+'.log'),env=E,role=name);j['handles'].append(active.record);save(name)
 while active.child.poll()is None:save();time.sleep(2)
 rc=active.child.returncode;cleanup=active.stop();active=None
 j.setdefault('steps',[]).append(dict(name=name,argv=args,exit_code=rc,cleanup=cleanup));save();assert rc==0,(name,rc)
try:
 save();idle()
 assert json.loads((R/'build-job.json').read_text())['status']=='complete'
 assert json.loads((R/'asset-preparation-complete.json').read_text())['status']=='complete'
 for n,h in json.loads((R/'entrypoint-r2-runtime-freeze.json').read_text())['files'].items():assert common.hash_file(P/n)==h,n
 assert common.hash_file(binding['binary']['path'])==binding['binary']['sha256']
 child_run('portable-entrypoint-r2-CPU',['python3','-B','-m','unittest','discover','-s',str(P/'tests'),'-p','test_*.py','-v'])
 for mode,profile,port in [('quality','quality100',18541)]:
  idle();log=R/('service-'+profile+'-002');out=R/('performance-C14-001'if mode=='performance'else'quality100-002')
  command=['python3','-B',str(P/'serve_candidate.py'),'--recipe',str(recipe_path),'--profile',profile,'--root',str(R),'--port',str(port),'--log-dir',str(log)]
  run=dict(mode=mode,profile=profile,service_command=command,started_ns=time.time_ns(),state='loading');j['runs'].append(run)
  service=Owned(command,R/('service-'+profile+'-002.log'),env=E,role='portable-service-'+profile);j['handles'].append(service.record);run['service']=service.record;save('loading-'+profile)
  deadline=time.monotonic()+300;last=''
  while True:
   assert service.child.poll()is None,('service exited',service.child.returncode)
   try:
    launch=common.check_service(log/'launch.json',recipe,rsha,profile,f'http://127.0.0.1:{port}/v1');refresh(service);run.update(state='ready',service=service.record,launch=launch);save('ready-'+profile);break
   except (OSError,ValueError,KeyError)as error:last=repr(error)
   assert time.monotonic()<deadline,last
   save();time.sleep(1)
  args=['python3','-B',str(P/('perf_candidate.py'if mode=='performance'else'accuracy_candidate.py')),'--recipe',str(recipe_path),'--root',str(R),'--service-launch',str(log/'launch.json'),'--base-url',f'http://127.0.0.1:{port}/v1','--output-dir',str(out)]
  if mode=='performance':args+=['--concurrency','14']
  run['evaluation_command']=args;child_run('evaluate-r2-'+profile,args)
  run['evaluation_identity']=json.loads((out/'evaluation-identity.json').read_text());run['state']='evaluation_complete';save()
  refresh(service);run['cleanup']=service.stop(timeout=30);service=None
  assert not Path('/proc',str(run['service']['pid'])).exists();idle();run.update(state='complete_and_service_reaped',finished_ns=time.time_ns());save('completed-'+profile)
 j.update(status='complete',finished_ns=time.time_ns());save('complete')
except BaseException as error:
 j.update(status='failed',error=repr(error),traceback=traceback.format_exc(),finished_ns=time.time_ns());save();raise
finally:
 if active:j['active_cleanup']=active.stop(timeout=30);save()
 if service:
  if service.child.poll()is None:refresh(service)
  j['service_cleanup']=service.stop(timeout=30);save()
