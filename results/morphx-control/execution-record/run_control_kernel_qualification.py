"""Owned GPU1 qualification, one campaign, immutable inputs and final restoration."""
import hashlib,json,os,time
from pathlib import Path
from inferx.pd_baselines import run_owned_command
from inferx.service_ownership import process_identity
from scripts.run_pd_baselines import run_with_restoration
from scripts import run_concurrency_campaign as previous
r=Path(os.environ['INFERX_ROOT']);out=r/'morphx-control/raw/kernels-v1';out.mkdir(parents=True,exist_ok=False);previous.DIRECTORY=out
saved=json.loads((r/'tuning/best-service.json').read_text());assert saved==json.loads((r/'context-agents/runtime.json').read_text())
identity=process_identity(saved['pid'],previous.BINARY);assert all(saved.get(k)==v for k,v in identity.items())
saved['original_argv']=Path(f'/proc/{saved["pid"]}/cmdline').read_bytes().decode().rstrip('\0').split('\0');(out/'original-service.json').write_text(json.dumps(saved,indent=2)+'\n');(out/'controller.py').write_bytes(Path(__file__).read_bytes())
metadata=dict(status='running',scope='Actual numerical/lifetime qualification only; no full-model or performance claim.',controller_pid=os.getpid(),started_unix=time.time(),cases=[])
libraries={'abi-only':r/'sota-validation/builds/libcuda_query_abi_v4.so',**{name:r/f'morphx-control/builds/libcuda_{name}_v1.so' for name in ['fill','event','both']}}
def flush():(out/'campaign.json').write_text(json.dumps(metadata,indent=2)+'\n')
def measure():
 base=dict(os.environ,CUDA_VISIBLE_DEVICES=previous.GPU,SMSCHED_LLM_TRACE_LIMIT='2048',SMSCHED_CONTROL_TRACE='1',SMSCHED_SPLIT_HINT='96',SMSCHED_LLM_TRICK_MODE='16',CUDA_HOME=str(r/'toolchains/cuda-12.8.1'),PYTHONPATH=str(r/'code-pd-baselines'))
 base['LD_LIBRARY_PATH']=':'.join([str(r/'toolchains/cuda-12.8.1/lib'),str(r/'toolchains/ninfer-deps/usr/lib/x86_64-linux-gnu'),str(r/'source/pd-baselines/gdrcopy/src'),str(r/'pd-baselines/deps/usr/lib/x86_64-linux-gnu'),base.get('LD_LIBRARY_PATH','')])
 for variant,library in libraries.items():
  for test,script,profile in [('vector-off','sota-validation/triton_vector_probe.py','0'),('vector-profile','sota-validation/triton_vector_probe.py','1'),('marlin-off','sota-validation/marlin_probe_v3.py','0'),('marlin-profile','sota-validation/marlin_probe_v3.py','1'),('concurrent-marlin','morphx-control/concurrent_marlin_probe.py','1')]:
   previous.gpu_free();name=variant+'-'+test
   env=dict(base,SMSCHED_PROFILE=profile,TRITON_CACHE_DIR=str(out/(name+'-cache')),LD_PRELOAD=str(library)+':'+str(r/'pd-baselines/builds/libpreload.so'),NEUTRINO_HOOK_DRIVER=str(library))
   command=[str(r/'pd-baselines/sglang0519-venv/bin/python'),str(r/script),str(out/(name+'-artifacts'))]
   entry=dict(variant=variant,test=test,command=command,library_sha256=hashlib.sha256(library.read_bytes()).hexdigest(),script_sha256=hashlib.sha256((r/script).read_bytes()).hexdigest(),status='running');metadata['cases'].append(entry);flush();start=time.monotonic()
   try:
    with (out/(name+'.log')).open('x') as log:code=run_owned_command(command,env,log,timeout=240)
    entry.update(status='passed' if code==0 else 'failed',returncode=code)
   except Exception as exc:entry.update(status='failed',error=repr(exc))
   entry['elapsed_s']=time.monotonic()-start;flush();print(json.dumps(entry),flush=True)
   # Preserve the first variant failure; other libraries remain independent probes.
   if entry['status']!='passed':break
flush()
try:
 run_with_restoration(saved,previous.restore,[r/'context-agents/runtime.json',r/'tuning/best-service.json'],lambda:previous.final_shutdown_check(saved,gpu0_state='stopped'),measure);metadata['status']='restored'
except BaseException as exc:metadata.update(status='interrupted_or_failed',error=repr(exc));raise
finally:metadata.update(finished_unix=time.time(),restored=(out/'restored-service.json').exists());flush()
