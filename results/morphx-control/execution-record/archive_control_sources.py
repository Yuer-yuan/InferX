"""Preserve exact source/manifest bytes, not large installed wheels or model weights."""
import hashlib,json,os,shutil,tarfile
from pathlib import Path
r=Path(os.environ['INFERX_ROOT']);root=r/'morphx-control';out=root/'source-capsule-v1';out.mkdir(exist_ok=False);builds=[]
paths=[r/'sota-validation/runtime-query-abi-v4',*[root/f'runtime-{v}-v1' for v in ['fill','event','both']],*[root/f'runtime-{v}-local-v2' for v in ['abi-only','fill','event','both']]]
for src in paths:
 manifest=json.loads((src/'manifest.json').read_text())
 for name,h in manifest['source_sha256'].items():assert hashlib.sha256((src/name).read_bytes()).hexdigest()==h,name
 library=Path(manifest['command'][manifest['command'].index('-o')+1]);assert hashlib.sha256(library.read_bytes()).hexdigest()==manifest['library_sha256']
 dest=out/src.relative_to(r);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copytree(src,dest)
 builds.append(dict(path=str(src.relative_to(r)),library=str(library.relative_to(r)),library_sha256=manifest['library_sha256']))
for folder,names in [('inferx',['morphx_control.py','morphx_validation.py']),('scripts',['morphx_control_ops.hpp','morphx_cuda_copy_pool.hpp','prepare_morphx_control.py'])]:
 for name in names:
  src=r/'code-pd-baselines'/folder/name;dest=out/'adapter'/folder/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dest)
with tarfile.open(r/'code-pd-baselines/results/sota-validation/reproduction/source-capsule-v2.tar.gz') as old:
 for name in ['MorphX-MIT.txt','SGLang-Apache-2.0.txt','NOTICE.txt']:
  dest=out/'licenses'/name;dest.parent.mkdir(exist_ok=True);dest.write_bytes(old.extractfile('./licenses/'+name).read())
files={str(p.relative_to(out)):dict(sha256=hashlib.sha256(p.read_bytes()).hexdigest(),bytes=p.stat().st_size) for p in sorted(out.rglob('*')) if p.is_file()}
report=dict(scope='Exact eight source/build manifest trees and adapters; kernels/compiler/overlay/model/venvs are reused pinned dependencies from prior SOTA capsule. Libraries stay on HDD, hashes included. No new efficacy or universal qualification claim.',builds=builds,payload=files)
(out/'capsule-manifest.json').write_text(json.dumps(report,indent=2)+'\n');archive=root/'source-capsule-v1.tar.gz'
assert not archive.exists()
with tarfile.open(archive,'w:gz') as t:
 for p in sorted(out.rglob('*')):
  if p.is_file():t.add(p,arcname=str(p.relative_to(out)))
summary=dict(archive=archive.name,sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),payload_files=len(files),payload_bytes=sum(x['bytes'] for x in files.values()),builds=builds)
(root/'source-capsule-summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary))
