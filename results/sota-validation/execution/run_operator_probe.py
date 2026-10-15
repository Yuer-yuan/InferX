"""Owned GPU1 numerical qualification only; original service always restored."""
import hashlib,json,os,time
from pathlib import Path
from inferx.pd_baselines import run_owned_command
from inferx.service_ownership import process_identity
from scripts.run_pd_baselines import run_with_restoration
from scripts import run_concurrency_campaign as previous
r=Path(os.environ['INFERX_ROOT']);out=r/'sota-validation/raw/operators-native-ex-v1';out.mkdir(parents=True,exist_ok=False)
previous.DIRECTORY=out
saved=json.loads((r/'tuning/best-service.json').read_text());assert saved==json.loads((r/'context-agents/runtime.json').read_text())
identity=process_identity(saved['pid'],previous.BINARY);assert all(saved.get(k)==v for k,v in identity.items())
saved['original_argv']=Path(f'/proc/{saved["pid"]}/cmdline').read_bytes().decode().rstrip('\0').split('\0')
(out/'original-service.json').write_text(json.dumps(saved,indent=2)+'\n');(out/'controller.py').write_bytes(Path(__file__).read_bytes())
metadata=dict(status='running',scope='Numerical correctness qualification only, no full model or PD performance claim.',controller_pid=os.getpid(),started_unix=time.time(),cases=[])
def flush():(out/'campaign.json').write_text(json.dumps(metadata,indent=2)+'\n')
def measure():
 previous.gpu_free()
 base=dict(os.environ,CUDA_VISIBLE_DEVICES=previous.GPU,SMSCHED_PROFILE='0',SMSCHED_LLM_TRACE_LIMIT='512')
 base['LD_LIBRARY_PATH']=':'.join([str(r/'toolchains/cuda-12.8.1/lib'),str(r/'toolchains/ninfer-deps/usr/lib/x86_64-linux-gnu'),str(r/'source/pd-baselines/gdrcopy/src'),str(r/'pd-baselines/deps/usr/lib/x86_64-linux-gnu'),base.get('LD_LIBRARY_PATH','')])
 cases=[('cooperative-native',r/'sota-validation/builds/cooperative_native'),('cooperative-transformed',r/'sota-validation/builds/cooperative_transformed')]+[(name,r/'pd-baselines/builds/ninfer-morphx/tests'/name) for name in ['ninfer_gdn_gating_proj_test','ninfer_gqa_attention_test','ninfer_bidirectional_gqa_attention_test','ninfer_kv_cache_append_prefix_test','ninfer_sparse_moe_test']]
 for mode,library in [('unhooked',None),('original-capture-port',r/'pd-baselines/builds/libcuda_mapped_capture.so'),('native-ex-v1',r/'sota-validation/builds/libcuda_native_ex_v1.so')]:
  env=dict(base)
  if library:
   env['LD_PRELOAD']=str(library)+':'+str(r/'pd-baselines/builds/libpreload.so');env['NEUTRINO_HOOK_DRIVER']=str(library)
  else:
   env.pop('LD_PRELOAD',None);env.pop('NEUTRINO_HOOK_DRIVER',None)
  for name,binary in cases:
   if name=='cooperative-native' and library:continue
   previous.gpu_free()
   entry=dict(mode=mode,test=name,binary=str(binary),binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),library_sha256=hashlib.sha256(library.read_bytes()).hexdigest() if library else None,log=f'{mode}-{name}.log',status='running');metadata['cases'].append(entry);flush()
   start=time.monotonic()
   try:
    with (out/entry['log']).open('x') as log:code=run_owned_command([str(binary)],env,log,timeout=120 if name.startswith('ninfer') else 30)
    text=(out/entry['log']).read_text(errors='replace');entry.update(status='passed' if code==0 else 'failed',returncode=code,patched_kernel_names=[line.split('name=',1)[-1] for line in text.splitlines() if '[MorphX-llm] phase=patched' in line],native_launch_lines=[line for line in text.splitlines() if '[MorphX-llm]' in line and any(s in line for s in ['phase=attrs-native','phase=ptsz-native','phase=ptsz-ex-native'])])
   except Exception as exc:entry.update(status='failed',error=f'{type(exc).__name__}: {exc}')
   entry['elapsed_s']=time.monotonic()-start;flush();print(mode,name,entry['status'],entry.get('returncode'),flush=True)
flush()
try:
 run_with_restoration(saved,previous.restore,[r/'context-agents/runtime.json',r/'tuning/best-service.json'],lambda:previous.final_shutdown_check(saved,gpu0_state='stopped'),measure)
 metadata['status']='restored'
except BaseException as exc:
 metadata.update(status='interrupted_or_failed',error=f'{type(exc).__name__}: {exc}');raise
finally:
 metadata.update(finished_unix=time.time(),restored=(out/'restored-service.json').exists());flush()
