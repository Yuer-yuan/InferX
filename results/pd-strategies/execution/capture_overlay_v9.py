"""Preserve exact runtime bytes, metadata and patches, including intermediate ports."""
import difflib,gzip,hashlib,json
from pathlib import Path
r=Path('/home/guokc/mypro/gnn-mount-sda1/InferX');destination=r/'pd-strategies/reproduction-v9';destination.mkdir(exist_ok=True)
for name in ['pdmux-graph-bound-v9']:
 overlay=r/'pd-strategies/overlays'/name;out=destination/name;out.mkdir(exist_ok=False);manifest=json.loads((overlay/'overlay-manifest.json').read_text());original=Path(manifest['source']);patch=[]
 for relative,expected in manifest['files'].items():
  before=(original/relative).read_bytes();after=(overlay/'sglang'/relative).read_bytes();assert hashlib.sha256(before).hexdigest()==expected['before_sha256'];assert hashlib.sha256(after).hexdigest()==expected['after_sha256']
  patch.extend(difflib.unified_diff(before.decode().splitlines(keepends=True),after.decode().splitlines(keepends=True),fromfile='a/sglang/'+relative,tofile='b/sglang/'+relative))
 runtime=(overlay/'inferx_pd_runtime.py').read_bytes();assert hashlib.sha256(runtime).hexdigest()==manifest['runtime_sha256']
 inventory=(overlay/'source-hashes.json').read_bytes();assert hashlib.sha256(inventory).hexdigest()==manifest['source_hashes_sha256']
 (out/'source.patch').write_text(''.join(patch));(out/'inferx_pd_runtime.py').write_bytes(runtime);(out/'overlay-manifest.json').write_bytes((overlay/'overlay-manifest.json').read_bytes());(out/'source-hashes.json.gz').write_bytes(gzip.compress(inventory,mtime=0))
files={str(p.relative_to(destination)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(destination.rglob('*')) if p.is_file()};(destination/'artifact-hashes.json').write_text(json.dumps(files,indent=2)+'\n');print('preserved',len(files),'files')
