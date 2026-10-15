"""Archive exact custom source bytes; large unchanged wheel/model stay on HDD."""
import difflib,hashlib,json,os,shutil,subprocess,tarfile
from pathlib import Path
r=Path(os.environ['INFERX_ROOT']);out=r/'sota-validation/source-capsule-v2';out.mkdir(exist_ok=False)
original=r/'source/pd-baselines/MorphX';patch=r/'pd-baselines/builds/morphx-pass-finish-barrier/source.patch'
assert subprocess.check_output(['git','diff','--binary'],cwd=original)==patch.read_bytes()
for name in ['sota-validation/runtime-cuda-copy-v2/runtime','sota-validation/runtime-cuda-copy-v2/include',
             'sota-validation/builds/marlin-morphx-v2/csrc','sota-validation/builds/marlin-morphx-v2/include',
             'source/pd-baselines/MorphX/smsched-pass']:
    target=out/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copytree(r/name,target)
for name in ['sota-validation/runtime-cuda-copy-v2/manifest.json','sota-validation/builds/marlin-morphx-v2/manifest.json',
             'sota-validation/builds/marlin-morphx-v2/cuda.cu','pd-baselines/builds/morphx-pass-finish-barrier/source.patch',
             'pd-baselines/builds/morphx-pass-finish-barrier/compile_commands.json','code-pd-baselines/scripts/cuda18_compat.cuh']:
    target=out/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(r/name,target)
overlay=r/'sota-validation/overlays/morphx-pd-v2';base=r/'pd-baselines/sglang0519-venv/lib/python3.12/site-packages'
inventory=json.loads((overlay/'source-hashes.json').read_text());changes=[];diff=[]
for name,expected in inventory.items():
    p=overlay/name;after=p.read_bytes();assert hashlib.sha256(after).hexdigest()==expected,name
    before=(base/name).read_bytes() if (base/name).is_file() else None
    if before==after:continue
    target=out/'overlay-changed'/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(after)
    changes.append(dict(file=name,before_sha256=hashlib.sha256(before).hexdigest() if before is not None else None,after_sha256=expected))
    diff.extend(difflib.unified_diff((before or b'').decode().splitlines(keepends=True),after.decode().splitlines(keepends=True),fromfile='a/'+name,tofile='b/'+name))
(out/'overlay-source.patch').write_text(''.join(diff));(out/'overlay-changes.json').write_text(json.dumps(changes,indent=2)+'\n')
for name in ['overlay-manifest.json','source-hashes.json']:shutil.copy2(overlay/name,out/('overlay-'+name))
licenses=out/'licenses';licenses.mkdir()
shutil.copy2(original/'LICENSE',licenses/'MorphX-MIT.txt')
shutil.copy2(base/'sglang-0.5.19.dist-info/licenses/LICENSE',licenses/'SGLang-Apache-2.0.txt')
(licenses/'NOTICE.txt').write_text('MorphX runtime/compiler source: Copyright (c) 2026 Storage Research Group @ Tsinghua University, MIT.\nSGLang overlay and kernel headers: SGLang contributors and original per-file holders, Apache-2.0; retain per-file notices.\nThis capsule contains the locally adapted source bytes used in InferX research; original license texts are included here.\n')
files={str(p.relative_to(out)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(out.rglob('*')) if p.is_file()}
(out/'capsule-manifest.json').write_text(json.dumps(dict(scope='Source-only reconstruction capsule. Dependency wheels/toolchains/model remain pinned on HDD. Rebuilt binaries must be requalified, not assumed byte-identical.',files_sha256=files),indent=2,sort_keys=True)+'\n')
archive=r/'sota-validation/source-capsule-v2.tar.gz'
subprocess.run(['tar','--sort=name','--mtime=@0','--owner=0','--group=0','--numeric-owner','-czf',str(archive),'-C',str(out),'.'],check=True)
with tarfile.open(archive,'r:gz') as t:
    for name,expected in files.items():assert hashlib.sha256(t.extractfile('./'+name).read()).hexdigest()==expected
print(json.dumps(dict(overlay_changed_files=len(changes),source_files=len(files),archive_bytes=archive.stat().st_size,archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest())))
