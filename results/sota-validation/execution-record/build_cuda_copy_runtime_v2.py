"""Preserve compiler/scheduler; change only GDR-unavailable control transport."""
import hashlib,json,os,shutil,subprocess
from pathlib import Path
r=Path(os.environ['INFERX_ROOT']);src=r/'sota-validation/runtime-native-ex-v3';out=r/'sota-validation/runtime-cuda-copy-v2';out.mkdir(exist_ok=False)
for name in ['runtime','include']:shutil.copytree(src/name,out/name)
original=json.loads((src/'manifest.json').read_text());p=out/'include/common.h';before=p.read_bytes();assert hashlib.sha256(before).hexdigest()==original['source_sha256']['include/common.h']
s=before.decode();a=s.index('class GdrPool {');b=s.index('typedef void (*fixSMHandle)',a);replacement=(r/'code-pd-baselines/scripts/morphx_cuda_copy_pool.hpp').read_text();p.write_text(s[:a]+replacement+'\n'+s[b:])
command=list(original['command']);command=[str(out/'runtime/src/cuda-llm.cpp') if x==str(src/'runtime/src/cuda-llm.cpp') else '-I'+str(out/'include') if x=='-I'+str(src/'include') else x for x in command];library=r/'sota-validation/builds/libcuda_copy_v2.so';command[command.index('-o')+1]=str(library)
with (out/'build.log').open('x') as log:subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
report=dict(command=command,source_sha256={str(p.relative_to(out)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(out.rglob('*')) if p.is_file()},library_sha256=hashlib.sha256(library.read_bytes()).hexdigest(),base_library_sha256=original['library_sha256'],changes=['Control counters/SM bounds stay in GPU VRAM as in author design; CPU access uses thread-private nonblocking CUDA copy streams and synchronized pinned staging; no global copy lock.','Original morphable compiler/in-place profiler/schedule_2 logic unchanged fromnative-ex-v3.'],limitations=['GDRCopy unavailable; DMA control latency differs from author mapped VRAM.','GPU numerical/model/performance qualification pending.'])
(out/'manifest.json').write_text(json.dumps(report,indent=2)+'\n');print(report['library_sha256'],flush=True)
