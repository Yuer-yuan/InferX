import os,json,hashlib,shutil
from pathlib import Path
from inferx.morphx_sglang import patch_marlin_loader
r=Path(os.environ['INFERX_ROOT']);src=r/'sota-validation/overlays/morphx-pd-v1';out=r/'sota-validation/overlays/morphx-pd-v2';assert not out.exists();shutil.copytree(src,out,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
p=out/'sglang/kernels/ops/quantization/gptq_marlin.py';before=p.read_bytes();p.write_text(patch_marlin_loader(before.decode()))
manifest=json.loads((out/'overlay-manifest.json').read_text());manifest['custom_marlin']=dict(before_sha256=hashlib.sha256(before).hexdigest(),after_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),library=str(r/'sota-validation/builds/marlin-morphx-v2/marlin_morphx.so'),library_sha256=hashlib.sha256((r/'sota-validation/builds/marlin-morphx-v2/marlin_morphx.so').read_bytes()).hexdigest())
inventory={str(p.relative_to(out)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(out.rglob('*')) if p.is_file() and p.name not in {'source-hashes.json','overlay-manifest.json'}};(out/'source-hashes.json').write_text(json.dumps(inventory,indent=2)+'\n');manifest['source_hashes_sha256']=hashlib.sha256((out/'source-hashes.json').read_bytes()).hexdigest();(out/'overlay-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(out)
