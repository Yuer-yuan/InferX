import hashlib,json,os,subprocess,urllib.request
from pathlib import Path
root=Path(os.environ['INFERX_ROOT'])
dest=root/'toolchains/cuda-12.8.1'; dest.mkdir(parents=True,exist_ok=True)
base='https://developer.download.nvidia.com/compute/cuda/redist/'
j=json.load(urllib.request.urlopen(base+'redistrib_12.8.1.json'))
(root/'toolchains/cuda-12.8.1-manifest.json').write_text(json.dumps(j,indent=2))
# NInfer uses cudart and cublas; CUDA headers additionally require CCCL.
for k in ('cuda_nvcc','cuda_cudart','cuda_cccl','cuda_nvtx','cuda_profiler_api','libcublas'):
 item=j[k]['linux-x86_64']; archive=root/'toolchains'/Path(item['relative_path']).name
 if not archive.exists():
  subprocess.run(['curl','-fL','--retry','5','--silent','--show-error','-o',str(archive),base+item['relative_path']],check=True)
 with archive.open('rb') as f: actual=hashlib.file_digest(f,'sha256').hexdigest()
 assert actual==item['sha256'], k
 subprocess.run(['tar','-xJf',str(archive),'--strip-components=1','-C',str(dest)],check=True)
 print(k,'verified and extracted',flush=True)
# NVIDIA component archives use lib/, while nvcc's profile searches lib64/.
if not (dest/'lib64').exists():
 (dest/'lib64').symlink_to('lib',target_is_directory=True)
