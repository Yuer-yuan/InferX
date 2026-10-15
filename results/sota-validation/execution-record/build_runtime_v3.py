import hashlib,json,os,shutil,subprocess
from pathlib import Path
from inferx.morphx_validation import patch_proc_address_queries
r=Path(os.environ['INFERX_ROOT']);src=r/'sota-validation/runtime-native-ex-v2';dest=r/'sota-validation/runtime-native-ex-v3';dest.mkdir(exist_ok=False)
for name in ['runtime','include']:shutil.copytree(src/name,dest/name)
p=dest/'runtime/src/cuda-llm.cpp';before=p.read_bytes();assert hashlib.sha256(before).hexdigest()==json.loads((src/'manifest.json').read_text())['source_sha256']['runtime/src/cuda-llm.cpp']
s=patch_proc_address_queries(before.decode())
p.write_text(s)
cuda=r/'toolchains/cuda-12.8.1';library=r/'sota-validation/builds/libcuda_native_ex_v3.so'
command=['taskset','-c','0-3,56-59','/usr/bin/g++-13','-shared','-std=c++20','-O3','-g','-fPIC','-DSMSCHED_MAPPED_HOST_CONTROL','-DSMSCHED_ALLOW_DEFAULT_STREAM','-DSMSCHED_THREADSAFE_PROFILE','-DSMSCHED_CAPTURE_BYPASS',str(p),'-I'+str(dest/'include'),'-I'+str(cuda/'include'),'-I'+str(r/'source/pd-baselines/gdrcopy/include'),'-I'+str(r/'pd-baselines/deps/usr/include'),'-L'+str(cuda/'lib'),'-L'+str(r/'source/pd-baselines/gdrcopy/src'),'-lcudart','-lgdrapi','-ldl','-Wl,-soname,libcuda.so','-o',str(library)]
with (dest/'build.log').open('x') as log:subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
report=dict(command=command,before_sha256=hashlib.sha256(before).hexdigest(),library_sha256=hashlib.sha256(library.read_bytes()).hexdigest(),source_sha256={str(p.relative_to(dest)):hashlib.sha256(p.read_bytes()).hexdigest() for p in dest.rglob('*') if p.is_file()},changes=['Preserve native query version/flag rejection and per-thread function selection; allv2changes retained'],scope='GPU qualification pending')
(dest/'manifest.json').write_text(json.dumps(report,indent=2)+'\n');print(report['library_sha256'],flush=True)
