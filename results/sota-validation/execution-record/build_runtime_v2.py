import hashlib,json,os,shutil,subprocess
from pathlib import Path
from inferx.morphx_validation import patch_profile_wave_accounting,remove_debug_kernel_skip
r=Path(os.environ['INFERX_ROOT']);src=r/'sota-validation/runtime-native-ex-v1';dest=r/'sota-validation/runtime-native-ex-v2';dest.mkdir(exist_ok=False)
for name in ['runtime','include']:shutil.copytree(src/name,dest/name)
p=dest/'runtime/src/cuda-llm.cpp';before=p.read_bytes();assert hashlib.sha256(before).hexdigest()=='11211c0d0dba1a5108dd61622a5f200c58da8d04f46cc45bdb86ad6c50e4dc8a'
s=remove_debug_kernel_skip(patch_profile_wave_accounting(before.decode()))
anchor='    info->h_minSM = minSM;\n    info->h_maxSM = maxSM;'
assert s.count(anchor)==1
s=s.replace(anchor,'''    if((info->h_minSM != minSM || info->h_maxSM != maxSM) && getenv("SMSCHED_CONTROL_TRACE")) {
        static std::atomic<int> count{0};
        if(count.fetch_add(1) < 4096) fprintf(stderr, "[MorphX-control] stream=%p name=%s old=%d:%d new=%d:%d\\n", info->stream, info->name, info->h_minSM, info->h_maxSM, minSM, maxSM);
    }
'''+anchor)
anchor='    gCI[info->name].store(CI);';assert s.count(anchor)==1
s=s.replace(anchor,anchor+'\n    if(getenv("SMSCHED_CONTROL_TRACE")) fprintf(stderr,"[MorphX-profile] name=%s CI=%f waves=%lu\\n",info->name,CI,profileWaves.size());')
p.write_text(s)
cuda=r/'toolchains/cuda-12.8.1';library=r/'sota-validation/builds/libcuda_native_ex_v2.so'
command=['taskset','-c','0-3,56-59','/usr/bin/g++-13','-shared','-std=c++20','-O3','-g','-fPIC','-DSMSCHED_MAPPED_HOST_CONTROL','-DSMSCHED_ALLOW_DEFAULT_STREAM','-DSMSCHED_THREADSAFE_PROFILE','-DSMSCHED_CAPTURE_BYPASS',str(p),'-I'+str(dest/'include'),'-I'+str(cuda/'include'),'-I'+str(r/'source/pd-baselines/gdrcopy/include'),'-I'+str(r/'pd-baselines/deps/usr/include'),'-L'+str(cuda/'lib'),'-L'+str(r/'source/pd-baselines/gdrcopy/src'),'-lcudart','-lgdrapi','-ldl','-Wl,-soname,libcuda.so','-o',str(library)]
with (dest/'build.log').open('x') as log:subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
report=dict(command=command,before_sha256=hashlib.sha256(before).hexdigest(),library_sha256=hashlib.sha256(library.read_bytes()).hexdigest(),source_sha256={str(p.relative_to(dest)):hashlib.sha256(p.read_bytes()).hexdigest() for p in dest.rglob('*') if p.is_file()},changes=['Exact integer accounting of original descending profiling waves','Remove debug-only float32 silent-success skip','Opt-in bounded actual SM control/profile diagnostics'],scope='GPU qualification pending')
(dest/'manifest.json').write_text(json.dumps(report,indent=2)+'\n');print(report['library_sha256'],flush=True)
