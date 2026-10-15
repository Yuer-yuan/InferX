"""Independent numerical oracle for actual compiled Triton MorphX kernels."""
import json,os,re,sys,time
from pathlib import Path
from inferx.morphx_triton import install
r=Path(os.environ['INFERX_ROOT']);out=Path(sys.argv[1]);out.mkdir(exist_ok=False)
key=install(r/'pd-baselines/builds/morphx-pass-finish-barrier/libsmsched.so','/usr/bin/opt-18',out/'ir')
import triton
import triton.language as tl
import torch
@triton.jit
def vector_oracle(X,Y,N:tl.constexpr,BLOCK:tl.constexpr):
 offsets=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
 values=tl.load(X+offsets,offsets<N,other=0)
 tl.store(Y+offsets,values*3+7,offsets<N)
n=1048613;x=torch.arange(n,dtype=torch.float32,device='cuda');y=torch.full_like(x,-999.)
start=time.monotonic();kernel=vector_oracle[(triton.cdiv(n,128),)](x,y,n,128,num_warps=4);torch.cuda.synchronize()
expected=x.cpu()*3+7;actual=y.cpu();assert torch.equal(actual,expected),f'{(actual!=expected).sum().item()} wrong values'
for _ in range(3):
 y.fill_(-999);vector_oracle[(triton.cdiv(n,128),)](x,y,n,128,num_warps=4);torch.cuda.synchronize();assert torch.equal(y.cpu(),expected)
report=dict(status='passed',n=n,entry=kernel.name,ptx_entries=re.findall(r'.visible .entry ([a-zA-Z_][a-zA-Z0-9_]*)',kernel.asm['ptx']),key=key,elapsed_s=time.monotonic()-start,scope='Independent vector correctness, not model or scheduling performance')
(out/'result.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
