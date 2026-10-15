"""Separate diagnostic GPU timeline, excluded from performance statistics."""
import argparse,asyncio,copy,json,os,threading,time,urllib.request
from pathlib import Path
from inferx.concurrency import run_load
from inferx.pd_baselines import load_frozen_workloads,arrival_requests
from inferx.service_ownership import process_identity
from scripts.run_pd_baselines import run_with_restoration,OwnedEngine,observe,wait_idle
from scripts import run_concurrency_campaign as previous
parser=argparse.ArgumentParser();parser.add_argument('--case',required=True);parser.add_argument('--run-id',required=True);args=parser.parse_args();assert Path(args.run_id).name==args.run_id
r=Path(os.environ['INFERX_ROOT']);out=r/'sota-validation/raw'/args.run_id;out.mkdir(exist_ok=False);previous.DIRECTORY=out
saved=json.loads((r/'tuning/best-service.json').read_text());assert saved==json.loads((r/'context-agents/runtime.json').read_text());identity=process_identity(saved['pid'],previous.BINARY);assert all(saved.get(k)==v for k,v in identity.items());saved['original_argv']=Path(f'/proc/{saved["pid"]}/cmdline').read_bytes().decode().rstrip('\0').split('\0');(out/'original-service.json').write_text(json.dumps(saved,indent=2)+'\n');(out/'controller.py').write_bytes(Path(__file__).read_bytes())
case=copy.deepcopy(next(c for c in json.loads((r/'sota-validation/configs/sota-validation.json').read_text())['cases'] if c['id']==args.case));case['environment'].update(PYTHONPATH=str(r/'sota-validation/overlays/morphx-pd-profile-v3')+':'+str(r/'code-pd-baselines'),INFERX_MORPHX_STREAM_MAP=str(out/'stream-ids'),INFERX_MORPHX_IR=str(out/'ir'),SMSCHED_CONTROL_TRACE='1')
meta=dict(status='running',scope='Profiled kernel lifetimes, not formal performance numbers',case=case,controller_pid=os.getpid(),started_unix=time.time())
def flush():(out/'campaign.json').write_text(json.dumps(meta,indent=2)+'\n')
def post(path,payload):
 req=urllib.request.Request('http://127.0.0.1:18083'+path,data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
 with urllib.request.urlopen(req,timeout=60) as response:return response.read().decode()
def measure():
 templates,grid=load_frozen_workloads(r/'concurrency-roofline');definition=next(d for d in grid['mixed_cases'] if d['name']=='cold-high');definition['rate']=.5
 previous.gpu_free()
 with OwnedEngine(case,r,out/'engine') as server:
  previous.quality_gate(server,templates);quality=json.loads((server.directory/'quality-result.json').read_text());assert quality['status']=='passed';meta['quality']=quality;flush()
  short=templates['tool-24576'];seeds=[dict(id=f'seed-{i}',prompt_id='tool-24576',payload=short['payload'],expected_tool=short['expected_tool']) for i in range(4)]
  with (out/'warmup.jsonl').open('x') as output:rows=asyncio.run(run_load('http://127.0.0.1:18083',seeds,output,'closed',4))
  assert all(x['functional_ok'] and x['summary']['ok'] and not x['summary']['truncated'] for x in rows);wait_idle()
  def profile_window():
   try:
    time.sleep(20);meta['profile_start_unix']=time.time();meta['profile_start_reply']=post('/start_profile',dict(output_dir=str(out/'trace'),activities=['GPU'],with_stack=False,record_shapes=False));flush()
    time.sleep(8);meta['profile_stop_reply']=post('/stop_profile',{});meta['profile_stop_unix']=time.time();flush()
   except Exception as exc:meta['profile_error']=repr(exc);flush()
  thread=threading.Thread(target=profile_window,daemon=True);thread.start()
  try:
   result=observe(server,'profiled-cold',arrival_requests(definition,templates,grid,'profile'),());meta['result']=result
   assert result['summary']['errors']==0 and result['summary']['tool_valid']==25
  finally:
   thread.join(timeout=180)
   if thread.is_alive():raise RuntimeError('profile request thread did not complete')
flush()
try:
 run_with_restoration(saved,previous.restore,[r/'context-agents/runtime.json',r/'tuning/best-service.json'],lambda:previous.final_shutdown_check(saved,gpu0_state='stopped'),measure);meta['status']='restored'
except BaseException as exc:meta.update(status='interrupted_or_failed',error=repr(exc));raise
finally:meta.update(finished_unix=time.time(),restored=(out/'restored-service.json').exists());flush()
