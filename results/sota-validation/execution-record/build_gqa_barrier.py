"""Recompile one existing CUDA translation unit; old source/build stay unchanged."""
import hashlib,json,os,shlex,shutil,subprocess
from pathlib import Path
r=Path(os.environ['INFERX_ROOT']);old=r/'pd-baselines/builds/ninfer-morphx';out=r/'sota-validation/builds/gqa-finish-barrier-v1';out.mkdir(exist_ok=False)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
obj='src/CMakeFiles/ninfer_ops.dir/ops/launcher/gqa_attention_decode.cu.o'
command=shlex.split(subprocess.check_output(['ninja','-t','commands',obj],cwd=old,text=True).splitlines()[-1]);assert command[0]=='/usr/bin/clang++-18'
plugin=r/'pd-baselines/builds/morphx-pass-finish-barrier/libsmsched.so';commands=[]
command=[('-fpass-plugin='+str(plugin)) if a.startswith('-fpass-plugin=') else a for a in command]
for flag,name in [('-o','gqa_attention_decode.cu.o'),('-MF','gqa_attention_decode.cu.o.d'),('-MT','gqa_attention_decode.cu.o')]:command[command.index(flag)+1]=str(out/name)
commands.append(command);original_sha=sha(old/'src/libninfer_ops.a')
(out/'provenance.json').write_text(json.dumps(dict(scope='Only decoder CUDA object recompiled with preexisting end-of-virtual-CTA barrier compiler; no quality/performance claim.',commands=commands,plugin_sha256=sha(plugin),original_archive_sha256=original_sha),indent=2)+'\n')
with (out/'compile.log').open('x') as log:subprocess.run(['taskset','-c','0-3,56-59',*command],cwd=old,stdout=log,stderr=subprocess.STDOUT,check=True)
shutil.copy2(old/'src/libninfer_ops.a',out/'libninfer_ops.a');subprocess.run(['ar','r',str(out/'libninfer_ops.a'),str(out/'gqa_attention_decode.cu.o')],check=True)
line=subprocess.check_output(['ninja','-t','commands','tests/ninfer_gqa_attention_test'],cwd=old,text=True).splitlines()[-1];tokens=shlex.split(line);begin=tokens.index('&&')+1;finish=tokens.index('&&',begin);link=tokens[begin:finish];assert link[0]=='/usr/bin/g++-13'
link=[str(out/'libninfer_ops.a') if a=='src/libninfer_ops.a' else str(old/a) if not a.startswith('-') and (old/a).is_file() else a for a in link]
link[link.index('-o')+1]=str(out/'ninfer_gqa_attention_test')
link=[('-Wl,--dependency-file='+str(out/'test-link.d')) if a.startswith('-Wl,--dependency-file=') else a for a in link]
subprocess.run(link,cwd=old,check=True);assert sha(old/'src/libninfer_ops.a')==original_sha
report=json.loads((out/'provenance.json').read_text());report.update(commands=[command,link],output_sha256={p.name:sha(p) for p in out.iterdir() if p.is_file() and p.name!='provenance.json'});(out/'provenance.json').write_text(json.dumps(report,indent=2)+'\n');print('rebuilt only decoder object and private GQA test',flush=True)
