"""Compare custom compiler Marlin against original NVCC on identical tensors."""
import os,json,time,sys
from pathlib import Path
import torch
import tvm_ffi
from sglang.kernels.ops.quantization import gptq_marlin as marlin
from sgl_kernel.scalar_type import scalar_types
r=Path(os.environ['INFERX_ROOT']);out=Path(sys.argv[1]);out.mkdir(exist_ok=False)
custom=tvm_ffi.load_module(str(r/'sota-validation/builds/marlin-morphx-v2/marlin_morphx.so'))
original=marlin._jit_gptq_marlin_module
reports=[]
for m,n,k in [(1,1024,5120),(4,1024,5120),(128,1024,5120),(512,1024,5120),(128,34816,5120)]:
 torch.manual_seed(20261015+m+n+k)
 a=torch.randn((m,k),device='cuda',dtype=torch.bfloat16)*.1
 b=torch.randint(-2147483648,2147483647,(k//16,n*2),device='cuda',dtype=torch.int32)
 scales=torch.rand((k//128,n),device='cuda',dtype=torch.bfloat16)*.01
 def run(use_custom):
  marlin._jit_gptq_marlin_module=(lambda dtype:custom) if use_custom else original
  workspace=torch.zeros(128*16,device='cuda',dtype=torch.int32)
  value=marlin.gptq_marlin_gemm(a,None,b,scales,None,None,None,None,workspace,scalar_types.uint4b8,m,n,k,is_k_full=True,use_atomic_add=False,use_fp32_reduce=True)
  torch.cuda.synchronize();return value.cpu()
 expected=run(False);actual=run(True)
 assert torch.isfinite(actual).all(),(m,n,k,'nonfinite')
 difference=(actual.float()-expected.float()).abs();max_abs=difference.max().item()
 assert torch.allclose(actual,expected,rtol=.03,atol=.01),(m,n,k,max_abs)
 report=dict(m=m,n=n,k=k,status='passed',max_abs=max_abs,exact=bool(torch.equal(actual,expected)));reports.append(report);print(json.dumps(report),flush=True)
(out/'result.json').write_text(json.dumps(dict(status='passed',cases=reports,scope='Kernel compiler/runtime numerical qualification vs original NVCC, no serving efficacy'),indent=2)+'\n')
