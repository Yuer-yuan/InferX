"""Actual two-stream transformed Marlin numerics; no serving efficacy claim."""
import json,os,sys
from pathlib import Path
import torch,tvm_ffi
from sglang.kernels.ops.quantization import gptq_marlin as marlin
from sgl_kernel.scalar_type import scalar_types
r=Path(os.environ['INFERX_ROOT']);out=Path(sys.argv[1]);out.mkdir(exist_ok=False)
custom=tvm_ffi.load_module(str(r/'sota-validation/builds/marlin-morphx-v2/marlin_morphx.so'))
reference=tvm_ffi.load_module(str(r/'cache/sglang/jit/sm89/sgl_kernel_jit_gptq_marlin_bf16_t/build-cdfae12ead5ed9fa/deps-19d3e762b336d324/sgl_kernel_jit_gptq_marlin_bf16_t.so'))
streams=[torch.cuda.Stream(),torch.cuda.Stream()];reports=[]
for pair in [[(1,34816,5120),(128,34816,5120)],[(4,34816,5120),(512,34816,5120)]]:
 inputs=[];expected=[]
 for m,n,k in pair:
  torch.manual_seed(20261016+m+n+k)
  a=torch.randn((m,k),device='cuda',dtype=torch.bfloat16)*.1
  b=torch.randint(-2147483648,2147483647,(k//16,n*2),device='cuda',dtype=torch.int32)
  scales=torch.rand((k//128,n),device='cuda',dtype=torch.bfloat16)*.01
  workspace=torch.zeros(128*16,device='cuda',dtype=torch.int32)
  inputs.append((a,b,scales,workspace,m,n,k))
 def call(i,module):
  a,b,scales,workspace,m,n,k=inputs[i];marlin._jit_gptq_marlin_module=lambda dtype:module
  return marlin.gptq_marlin_gemm(a,None,b,scales,None,None,None,None,workspace,scalar_types.uint4b8,m,n,k,is_k_full=True,use_atomic_add=False,use_fp32_reduce=True)
 for i in range(2):expected.append(call(i,reference).cpu())
 torch.cuda.synchronize()
 # Profile both actual stream/kernel geometries before the asynchronous stress.
 for i in range(2):
  with torch.cuda.stream(streams[i]):call(i,custom)
 torch.cuda.synchronize()
 outputs=[]
 for repeat in range(12):
  for i in range(2):
   with torch.cuda.stream(streams[i]):outputs.append((i,call(i,custom)))
 torch.cuda.synchronize()
 for i,value in outputs:
  actual=value.cpu();difference=(actual.float()-expected[i].float()).abs().max().item()
  assert torch.allclose(actual,expected[i],rtol=.03,atol=.01),(pair[i],difference)
  reports.append(dict(shape=pair[i],max_abs=difference,exact=bool(torch.equal(actual,expected[i]))))
(out/'result.json').write_text(json.dumps(dict(status='passed',streams=[s.cuda_stream for s in streams],checks=reports,scope='Two explicit-stream numerical/lifetime stress, not measured SM occupancy or performance; inspect actual dynamic-bound logs separately.'),indent=2)+'\n')
print(json.dumps(dict(status='passed',checks=len(reports),streams=[s.cuda_stream for s in streams])))
