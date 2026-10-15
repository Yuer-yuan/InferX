"""Compile installed/pinned BF16 Marlin JIT source with actual MorphX pass."""
import hashlib,json,os,shutil,subprocess
from pathlib import Path
r=Path(os.environ['INFERX_ROOT']);out=r/'sota-validation/builds/marlin-morphx-v1';out.mkdir(exist_ok=False)
site=r/'pd-baselines/sglang0519-venv/lib/python3.12/site-packages';jit=site/'sglang/kernels/jit'
for name in ['csrc','include']:shutil.copytree(jit/name,out/name)
old=r/'cache/sglang/jit/sm89/sgl_kernel_jit_gptq_marlin_bf16_t/build-cdfae12ead5ed9fa/deps-19d3e762b336d324'
source=(old/'cuda.cu').read_text().replace(str(jit/'csrc'),str(out/'csrc'));(out/'cuda.cu').write_text(source)
plugin=r/'pd-baselines/builds/morphx-pass-finish-barrier/libsmsched.so';cuda=r/'toolchains/cuda-12.8.1'
command=['taskset','-c','0-3,56-59','/usr/bin/clang++-18','-x','cuda','--cuda-gpu-arch=sm_89','--cuda-path='+str(cuda),'--gcc-install-dir=/usr/lib/gcc/x86_64-linux-gnu/13','-Wno-unknown-cuda-version','-include',str(r/'code-pd-baselines/scripts/cuda18_compat.cuh'),'-fpass-plugin='+str(plugin),'-DSGL_CUDA_ARCH=890','-std=c++20','-O3','-fPIC','-I'+str(site/'tvm_ffi/include'),'-I'+str(out/'include'),'-MD','-MF',str(out/'cuda.o.d'),'-c',str(out/'cuda.cu'),'-o',str(out/'cuda.o')]
report=dict(command=command,plugin_sha256=hashlib.sha256(plugin.read_bytes()).hexdigest(),source_sha256={str(p.relative_to(out)):hashlib.sha256(p.read_bytes()).hexdigest() for p in out.rglob('*') if p.is_file()},scope='CPU custom compilation; GPU correctness/coverage not qualified')
(out/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
with (out/'compile.log').open('x') as log:done=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
report['returncode']=done.returncode
if done.returncode==0:
 link=['/usr/bin/g++-13',str(out/'cuda.o'),'-shared','-L'+str(site/'tvm_ffi/lib'),'-ltvm_ffi','-L'+str(cuda/'lib64'),'-lcudart','-o',str(out/'marlin_morphx.so')];subprocess.run(link,check=True);report['link_command']=link;report['library_sha256']=hashlib.sha256((out/'marlin_morphx.so').read_bytes()).hexdigest()
(out/'manifest.json').write_text(json.dumps(report,indent=2)+'\n');print('marlin compile',done.returncode,flush=True)
