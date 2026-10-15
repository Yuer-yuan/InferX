"""Read-only rehash of immutable runtime/compiler/overlay provenance."""
import hashlib,json,os
from pathlib import Path
r=Path(os.environ['INFERX_ROOT']);out=r/'sota-validation/runtime-artifact-audit.json';sources={}
def sha(path):
    data=path.read_bytes();digest=hashlib.sha256(data).hexdigest()
    sources[str(path.relative_to(r))]=dict(sha256=digest,bytes=len(data));return digest
checked=[]
for folder,library in [('sota-validation/runtime-cuda-copy-v2','sota-validation/builds/libcuda_copy_v2.so'),
                       ('sota-validation/builds/marlin-morphx-v2','sota-validation/builds/marlin-morphx-v2/marlin_morphx.so')]:
    root=r/folder;manifest=json.loads((root/'manifest.json').read_text());sha(root/'manifest.json')
    assert sha(r/library)==manifest['library_sha256'],library
    for name,expected in manifest['source_sha256'].items():
        path=root/name;assert path.is_file(),str(path)
        assert sha(path)==expected,str(path)
    checked.append(dict(root=folder,library=library,source_files=len(manifest['source_sha256'])))
marlin=json.loads((r/'sota-validation/builds/marlin-morphx-v2/manifest.json').read_text())
plugin=r/'pd-baselines/builds/morphx-pass-finish-barrier/libsmsched.so'
assert sha(plugin)==marlin['plugin_sha256']
root=r/'sota-validation/overlays/morphx-pd-v2';manifest=json.loads((root/'overlay-manifest.json').read_text());sha(root/'overlay-manifest.json')
assert sha(root/'source-hashes.json')==manifest['source_hashes_sha256']
inventory=json.loads((root/'source-hashes.json').read_text())
for name,expected in inventory.items():assert sha(root/name)==expected,name
checked.append(dict(root=str(root.relative_to(r)),source_files=len(inventory)))
report=dict(scope='Immutable sources and actual library files rehashed; this is not a time-weighted transformation-coverage or paper performance-ceiling claim.',checked=checked,source_sha256=sources)
out.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n');print(json.dumps(dict(checked=checked,files_rehashed=len(sources))))
