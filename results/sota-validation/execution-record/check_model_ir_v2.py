import os,json,subprocess,re
from pathlib import Path
from inferx.morphx_triton import llvm18_compatible_ir
r=Path(os.environ['INFERX_ROOT']);out=r/'sota-validation/model-ir-check-v2';out.mkdir(exist_ok=False)
source=r/'sota-validation/raw/sglang-port-v1/compiler-unhooked-ir/3701297/00e0e7feb27f42c6ac50ce569f93ced5/original.ll'
s,changes=llvm18_compatible_ir(source.read_text());(out/'compatible.ll').write_text(s)
cmd=['/usr/bin/opt-18','-load-pass-plugin='+str(r/'pd-baselines/builds/morphx-pass-finish-barrier/libsmsched.so'),'-passes=default<O3>','-S',str(out/'compatible.ll'),'-o',str(out/'morphable.ll')]
p=subprocess.run(cmd,capture_output=True,text=True);(out/'opt.stderr').write_text(p.stderr);report=dict(command=cmd,returncode=p.returncode,changes=changes)
if not p.returncode:
 from triton._C.libtriton import llvm
 llvm.init_targets();ptx=llvm.translate_to_asm((out/'morphable.ll').read_text(),'nvptx64-nvidia-cuda','sm_89','+ptx86',['nvptx-mad-wide-opt'],True,False);(out/'morphable.ptx').write_text(ptx);report['entries']=re.findall(r'.visible .entry ([a-zA-Z_][a-zA-Z0-9_]*)',ptx)
(out/'result.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
