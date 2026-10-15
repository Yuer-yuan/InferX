"""Full27B/128K qualification of process-local MorphX engine/compiler port."""
import json,hashlib,os,time
from pathlib import Path
from inferx.pd_baselines import load_frozen_workloads,snapshot_harness
from inferx.service_ownership import process_identity
from scripts.run_pd_baselines import run_with_restoration,OwnedEngine,observe
from scripts import run_concurrency_campaign as previous
r=Path(os.environ['INFERX_ROOT']);out=r/'sota-validation/raw/sglang-port-v4';out.mkdir(parents=True,exist_ok=False);previous.DIRECTORY=out
saved=json.loads((r/'tuning/best-service.json').read_text());assert saved==json.loads((r/'context-agents/runtime.json').read_text())
identity=process_identity(saved['pid'],previous.BINARY);assert all(saved.get(k)==v for k,v in identity.items());saved['original_argv']=Path(f'/proc/{saved["pid"]}/cmdline').read_bytes().decode().rstrip('\0').split('\0')
(out/'original-service.json').write_text(json.dumps(saved,indent=2)+'\n');(out/'controller.py').write_bytes(Path(__file__).read_bytes())
base=next(c for c in json.loads((r/'code-pd-baselines/configs/pd-strategies.json').read_text())['cases'] if c['id']=='sglang-pdmux-private-best')
base.update(startup_timeout_s=1800,qualification_note='Full model/compiler qualification only; primary P/D streams; custom compiled+kernel-qualified BF16 Marlin; no efficacy claim')
base['environment'].update(PYTHONPATH=str(r/'sota-validation/overlays/morphx-pd-v2')+':'+str(r/'code-pd-baselines'),INFERX_MORPHX_COMPILE='1',SMSCHED_LLM_TRACE_LIMIT='1024',SMSCHED_CONTROL_TRACE='1',SMSCHED_PROFILE='1',TRITON_CACHE_DIR=str(r/'sota-validation/cache/triton-port-v2'))
base['environment']['INFERX_MORPHX_MARLIN']=str(r/'sota-validation/builds/marlin-morphx-v2/marlin_morphx.so')
lib=r/'sota-validation/builds/libcuda_native_ex_v3.so';base['environment']['LD_LIBRARY_PATH']=':'.join([str(r/'toolchains/cuda-12.8.1/lib'),str(r/'toolchains/ninfer-deps/usr/lib/x86_64-linux-gnu'),str(r/'source/pd-baselines/gdrcopy/src'),str(r/'pd-baselines/deps/usr/lib/x86_64-linux-gnu'),os.environ.get('LD_LIBRARY_PATH','')])
metadata=dict(status='running',scope=base['qualification_note'],controller_pid=os.getpid(),started_unix=time.time(),cases=[],library_sha256=hashlib.sha256(lib.read_bytes()).hexdigest(),harness_sha256=snapshot_harness(r/'code-pd-baselines',out/'harness',[str(p.relative_to(r/'code-pd-baselines')) for folder in ['scripts','inferx'] for p in sorted((r/'code-pd-baselines'/folder).rglob('*.py'))]))
def flush():(out/'campaign.json').write_text(json.dumps(metadata,indent=2)+'\n')
def measure():
 templates,_=load_frozen_workloads(r/'concurrency-roofline')
 for mode,hook in [('compiler-unhooked',False),('compiler-profile-hooked',True)]:
  case=json.loads(json.dumps(base));case['id']='morphx-sg-'+mode;case['morphx']['enabled']=hook;case['environment']['INFERX_MORPHX_IR']=str(out/(mode+'-ir'))
  if hook:case['environment'].update(LD_PRELOAD=str(lib)+':'+str(r/'pd-baselines/builds/libpreload.so'),NEUTRINO_HOOK_DRIVER=str(lib))
  entry=dict(case=case['id'],status='starting');metadata['cases'].append(entry);flush()
  try:
   previous.gpu_free()
   server=OwnedEngine(case,r,out/mode);server.args += ['--watchdog-timeout','600']
   with server:
    previous.quality_gate(server,templates);entry['quality']=json.loads((server.directory/'quality-result.json').read_text());flush()
    if entry['quality'].get('status')!='passed':raise RuntimeError('quality gate failed')
    short=templates['tool-24576'];long=templates['cold-131072']
    seeds=[dict(id=f'seed-short-{i}',prompt_id='tool-24576',payload=short['payload'],expected_tool=short['expected_tool']) for i in range(4)]
    result=observe(server,'full128k',[dict(id='probe-long',prompt_id='cold-131072',payload=long['payload'],expected_tool=long['expected_tool'])],seeds)
    if result['summary']['errors'] or result['long']['protocol_ok']!=1:raise RuntimeError('complete128K gate failed')
    entry.update(status='passed',result=result)
  except Exception as exc:entry.update(status='failed',error=f'{type(exc).__name__}: {exc}');flush();print(json.dumps(entry),flush=True);break
  flush();print(json.dumps(entry),flush=True)
flush()
try:
 run_with_restoration(saved,previous.restore,[r/'context-agents/runtime.json',r/'tuning/best-service.json'],lambda:previous.final_shutdown_check(saved,gpu0_state='stopped'),measure);metadata['status']='restored'
except BaseException as exc:metadata.update(status='failed_or_interrupted',error=f'{type(exc).__name__}: {exc}');raise
finally:metadata.update(finished_unix=time.time(),restored=(out/'restored-service.json').exists());flush()
