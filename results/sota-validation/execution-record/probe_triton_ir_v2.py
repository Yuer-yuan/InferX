"""CPU feasibility; drop only newer optimization promises for LLVM18 parsing."""
import hashlib,json,os,re,subprocess
from pathlib import Path
from triton._C.libtriton import llvm
r=Path(os.environ['INFERX_ROOT']);out=r/'sota-validation/triton-ir-probe-v2';out.mkdir(exist_ok=False)
original=(r/'sota-validation/triton-ir-probe-v1/original.ll').read_text()
text=original.replace('captures(none)','nocapture').replace('or disjoint ','or ')
text=re.sub(r' range\(i32 0, [0-9]+\)', '',text)
text=text.replace('target datalayout =','target triple = "nvptx64-nvidia-cuda"\ntarget datalayout =',1)
(out/'compatible.ll').write_text(text)
plugin=r/'pd-baselines/builds/morphx-pass-finish-barrier/libsmsched.so'
command=['/usr/bin/opt-18','-load-pass-plugin='+str(plugin),'-passes=default<O3>','-S',str(out/'compatible.ll'),'-o',str(out/'morphable.ll')]
done=subprocess.run(command,capture_output=True,text=True);(out/'opt.stderr').write_text(done.stderr)
report=dict(returncode=done.returncode,command=command,scope='CPU feasibility, no GPU/model/performance claim',normalization='captures(none)->nocapture; conservatively remove or disjoint and return range promises; add correct NVPTX target triple')
if done.returncode==0:
 llir=(out/'morphable.ll').read_text();report['ir_kernels']=re.findall(r'define[^\n]*?ptx_kernel[^@]*@([^ (]+)',llir)
 try:
  llvm.init_targets();ptx=llvm.translate_to_asm(llir,'nvptx64-nvidia-cuda','sm_89','+ptx86',['nvptx-mad-wide-opt'],True,False);(out/'morphable.ptx').write_text(ptx);report['ptx_kernels']=re.findall(r'.visible .entry ([a-zA-Z_][a-zA-Z0-9_]*)',ptx)
 except Exception as exc:report['translation_error']=str(exc)
report['sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir() if p.is_file()};(out/'result.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
