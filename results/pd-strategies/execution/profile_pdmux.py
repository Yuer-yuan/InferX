"""Separate profiling wave; timing is excluded from formal comparisons."""
import asyncio,hashlib,json,os,threading,time,urllib.request
from pathlib import Path
from inferx.concurrency import run_load
from inferx.pd_baselines import GPU1,load_frozen_workloads,arrival_requests,snapshot_harness
from inferx.service_ownership import process_identity
from scripts.run_pd_baselines import OwnedEngine,run_with_restoration,observe,wait_idle,ENDPOINT
from scripts import run_concurrency_campaign as previous
root=Path(os.environ['INFERX_ROOT']);assert root==Path('/home/guokc/mypro/gnn-mount-sda1/InferX')
prior=root/'pd-baselines/raw/pd-strategies-pdmux-private-main-v1'
while True:
 try:closed=json.loads((prior/'campaign.json').read_text())
 except (FileNotFoundError,json.JSONDecodeError):closed={'status':'running'}
 if closed['status']!='running':
  assert closed['status']=='restored' and closed.get('restored');break
 time.sleep(1)
selected='sglang-pdmux-private-best'
assert all(x['state']=='measured' for x in closed['cases'] if x['case']==selected),'PD not qualified; no profiling'
case=next(c for c in json.loads((prior/'config.json').read_text()) if c['id']==selected)
output=root/'pd-baselines/raw/pd-strategies-pdmux-profile-v1';output.mkdir(exist_ok=False);previous.DIRECTORY=output
(output/'config.json').write_text(json.dumps([case],indent=2)+'\n')
saved=json.loads((root/'tuning/best-service.json').read_text());assert saved==json.loads((root/'context-agents/runtime.json').read_text())
identity=process_identity(saved['pid'],previous.BINARY);assert all(saved.get(k)==v for k,v in identity.items())
saved['original_argv']=Path(f'/proc/{saved["pid"]}/cmdline').read_bytes().decode().rstrip('\0').split('\0');(output/'original-service.json').write_text(json.dumps(saved,indent=2))
templates,grid=load_frozen_workloads(root/'concurrency-roofline');definition=dict(name='cold-low',rate=.5,long_mode='cold')
state=dict(status='running',started_unix=time.time(),cases=[],controller_pid=os.getpid(),workload_definitions=[definition],profiling_only=True,
 harness_sha256=snapshot_harness(root/'code-pd-baselines',output/'harness',[str(p.relative_to(root/'code-pd-baselines')) for folder in ['scripts','inferx'] for p in sorted((root/'code-pd-baselines'/folder).rglob('*.py'))]))
def flush():(output/'campaign.json').write_text(json.dumps(state,indent=2)+'\n')
def measure():
 label=selected+'-r0-cold-low';item=dict(case=selected,repetition=0,group='cold-low',state='starting');state['cases'].append(item);flush();previous.gpu_free()
 try:
  with OwnedEngine(case,root,output/label) as server:
   previous.quality_gate(server,templates)
   quality=json.loads((server.directory/'quality-result.json').read_text());assert quality['status']=='passed';item['quality']=quality
   short=templates['tool-24576'];seeds=[dict(id=f'seed-{i}',prompt_id='tool-24576',payload=short['payload'],expected_tool=short['expected_tool']) for i in range(4)]
   with (server.directory/'profile-seeds.jsonl').open('x') as out:warm=asyncio.run(run_load(ENDPOINT,seeds,out,'closed',4))
   assert all(r['summary']['ok'] and r.get('functional_ok') is True for r in warm);wait_idle()
   traces=output/'traces';traces.mkdir();profile=dict(scope='Separate GPU-only trace around24s in a mixed wave; excluded from performance comparisons',events=[])
   def post(endpoint,payload):
    req=urllib.request.Request(ENDPOINT+endpoint,data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(req,timeout=120) as resp:return resp.status,resp.read().decode()
   def capture():
    try:
     deadline=time.monotonic()+30
     while not (server.directory/'cold-low/requests.jsonl').exists():
      if time.monotonic()>deadline:raise TimeoutError('wave did not start')
      time.sleep(.01)
     time.sleep(24)
     profile['events'].append(dict(unix=time.time(),action='start',response=post('/start_profile',dict(output_dir=str(traces),activities=['GPU'],with_stack=False,record_shapes=False,profile_id='pdmux-mixed',profile_prefix='pdmux'))))
     time.sleep(2)
     profile['events'].append(dict(unix=time.time(),action='stop',response=post('/stop_profile',{})))
     profile['status']='captured'
    except Exception as exc:profile.update(status='failed',error=f'{type(exc).__name__}: {exc}')
    finally:(output/'profile-controller.json').write_text(json.dumps(profile,indent=2)+'\n')
   thread=threading.Thread(target=capture,daemon=True);thread.start()
   try:observe(server,'cold-low',arrival_requests(definition,templates,grid,label),())
   finally:thread.join(timeout=270)
   assert not thread.is_alive(),'profiling thread still running';item['state']='measured'
 except Exception as exc:item.update(state='failed',error=f'{type(exc).__name__}: {exc}');raise
 finally:flush()
flush()
try:
 run_with_restoration(saved,previous.restore,[root/'context-agents/runtime.json',root/'tuning/best-service.json'],lambda:previous.final_shutdown_check(saved),measure);state['status']='restored'
except BaseException as exc:state.update(status='interrupted_or_failed',error=f'{type(exc).__name__}: {exc}');raise
finally:state.update(finished_unix=time.time(),restored=(output/'restored-service.json').exists());flush()
