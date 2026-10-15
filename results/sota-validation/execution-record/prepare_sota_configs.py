"""Freeze matched cases only after exact full-model kernel/runtime qualification."""
import copy,hashlib,json,os,re
from pathlib import Path
r=Path(os.environ['INFERX_ROOT']);raw=r/'sota-validation/raw/sglang-copy-v7';metadata=json.loads((raw/'campaign.json').read_text());assert metadata['status']=='restored' and metadata['restored']
assert len(metadata['cases'])==1 and metadata['cases'][0]['status']=='passed'
library=r/'sota-validation/builds/libcuda_copy_v2.so';sha=hashlib.sha256(library.read_bytes()).hexdigest();assert metadata['library_sha256']==sha
log=(raw/'compiler-profile-hooked/server.log').read_text();names=sorted(set(re.findall(r'\[MorphX-llm\] phase=patched[^\n]*name=(\S+)',log)));assert names
base=next(c for c in json.loads((r/'code-pd-baselines/configs/pd-strategies.json').read_text())['cases'] if c['id']=='sglang-pdmux-private-best')
base.update(startup_timeout_s=1200,interleaving='SGLang P/D on two primary-context streams; MorphX actual compiler/profiler/schedule_2 controls resources',capacity_note='Samecheckpoint131328KV/C4/16BF16SSM/FP8KV/eager; privateMarlinworkspaces and identicalSGLangadmission; noGreenContextSMpartition',qualification_note='Fullcode/4x24Ktools/128K passed; actualcompiledMarlin/Triton, nativecuBLAS/FlashInfer remain explicitpartialcoverage')
env=base['environment'];env.update(PYTHONPATH=str(r/'sota-validation/overlays/morphx-pd-v2')+':'+str(r/'code-pd-baselines'),INFERX_MORPHX_COMPILE='1',INFERX_MORPHX_MARLIN=str(r/'sota-validation/builds/marlin-morphx-v2/marlin_morphx.so'),SMSCHED_LLM_TRACE_LIMIT='2048',SMSCHED_PROFILE='1',TRITON_CACHE_DIR=str(r/'sota-validation/cache/triton-port-v2'))
env['LD_LIBRARY_PATH']=':'.join([str(r/'toolchains/cuda-12.8.1/lib'),str(r/'toolchains/ninfer-deps/usr/lib/x86_64-linux-gnu'),str(r/'source/pd-baselines/gdrcopy/src'),str(r/'pd-baselines/deps/usr/lib/x86_64-linux-gnu'),os.environ.get('LD_LIBRARY_PATH','')])
env.pop('SMSCHED_CONTROL_TRACE',None)
base['morphx']=dict(enabled=False,patched_launches=0,scope='Same actual transformed compiler modules; no runtime interposer so original entries execute')
cases=[]
for name,split,mode in [('compiled-primary-control',0,0),('morphx-copy60',60,0),('morphx-copy96',96,0),('morphx-copy96-no-launch-cap',96,16)]:
 c=copy.deepcopy(base);c['id']=name;c['environment']['INFERX_MORPHX_IR']=str(r/'sota-validation/ir'/name)
 if split:
  c['morphx']=dict(enabled=True,patched_launches=len(names),observed_kernel_names=names,qualification_raw=str(raw),qualification_library_sha256=sha,coverage_note='Observed deduplicated names; not a complete launch/time fraction or exact author framework reproduction')
  c['environment'].update(LD_PRELOAD=str(library)+':'+str(r/'pd-baselines/builds/libpreload.so'),NEUTRINO_HOOK_DRIVER=str(library),SMSCHED_SPLIT_HINT=str(split),SMSCHED_LLM_TRICK_MODE=str(mode))
 cases.append(c)
report=dict(scope='MorphX-on-SGLangport; same-engine compiler/runtime causal controls, no exact author GDRCopy/privatevLLM reproduction',selection='Predeclared at most3diagnosticsettings: author-default60,96, and96withlaunch-CI-cap-disabled ablation (agent scheduler remains). Select jointly on absolute short/long and throughput; freeze beforefresh3mixedrepeats.',qualification_library_sha256=sha,cases=cases)
p=r/'sota-validation/configs/sota-validation.json';p.parent.mkdir(parents=True,exist_ok=True)
with p.open('x') as f:json.dump(report,f,indent=2);f.write('\n')
print(p,flush=True)
