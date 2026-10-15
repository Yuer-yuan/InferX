"""CPU-only feasibility: route Triton LLVM IR through the actual MorphX pass."""
import hashlib,json,os,re,subprocess,time
from pathlib import Path
import triton
import triton.language as tl
from triton.compiler import ASTSource
from triton.backends.compiler import GPUTarget
from triton import knobs

r=Path(os.environ['INFERX_ROOT']);out=r/'sota-validation/triton-ir-probe-v1';out.mkdir(exist_ok=False)
plugin=r/'pd-baselines/builds/morphx-pass-finish-barrier/libsmsched.so';key=hashlib.sha256(plugin.read_bytes()+b'triton-cpu-ir-v1').hexdigest()
class ProbeDone(Exception):pass
@triton.jit
def vector_oracle(X,Y,N:tl.constexpr,BLOCK:tl.constexpr):
 offsets=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
 values=tl.load(X+offsets,offsets<N,other=0)
 tl.store(Y+offsets,values*3+7,offsets<N)

def hook(*args):
 if not args:return key,key
 backend,stages,options,language,capability=args
 before=stages['llir']
 def instrument(src,metadata):
  llir=before(src,metadata);(out/'original.ll').write_text(llir)
  command=['/usr/bin/opt-18','-load-pass-plugin='+str(plugin),'-passes=default<O3>','-S',str(out/'original.ll'),'-o',str(out/'morphable.ll')]
  done=subprocess.run(command,capture_output=True,text=True);(out/'opt.stdout').write_text(done.stdout);(out/'opt.stderr').write_text(done.stderr)
  report=dict(scope='CPU compile-only; no GPU execution/model/performance claim.',triton=triton.__version__,plugin_sha256=hashlib.sha256(plugin.read_bytes()).hexdigest(),command=command,returncode=done.returncode,original_ir_sha256=hashlib.sha256(llir.encode()).hexdigest())
  if done.returncode==0:
   transformed=(out/'morphable.ll').read_text();report['kernels']=re.findall(r'define[^\n]*?ptx_kernel[^@]*@([^ (]+)',transformed);report['transformed_ir_sha256']=hashlib.sha256(transformed.encode()).hexdigest()
  (out/'result.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
  raise ProbeDone()
 stages['llir']=instrument
knobs.runtime.add_stages_inspection_hook=hook
try:
 triton.compile(ASTSource(vector_oracle,signature={'X':'*fp32','Y':'*fp32','N':'constexpr','BLOCK':'constexpr'},constexprs={'N':32768,'BLOCK':128}),target=GPUTarget('cuda',89,32),options={'num_warps':4})
except ProbeDone:pass
