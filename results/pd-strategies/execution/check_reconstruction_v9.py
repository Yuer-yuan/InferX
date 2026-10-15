import gzip,hashlib,json,shutil,subprocess
from pathlib import Path
r=Path('/home/guokc/mypro/gnn-mount-sda1/InferX');evidence=r/'pd-strategies/reproduction-v9';checks=r/'pd-strategies/reconstruction-check-v9';checks.mkdir(exist_ok=False);rows=[]
for artifact in sorted(p for p in evidence.iterdir() if p.is_dir()):
 manifest=json.loads((artifact/'overlay-manifest.json').read_text());destination=checks/artifact.name;destination.mkdir();shutil.copytree(manifest['source'],destination/'sglang',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
 done=subprocess.run(['patch','--batch','-p1','-i',str(artifact/'source.patch')],cwd=destination,capture_output=True,text=True);assert done.returncode==0,done.stderr+done.stdout
 (destination/'inferx_pd_runtime.py').write_bytes((artifact/'inferx_pd_runtime.py').read_bytes());inventory=gzip.decompress((artifact/'source-hashes.json.gz').read_bytes());assert hashlib.sha256(inventory).hexdigest()==manifest['source_hashes_sha256'];expected=json.loads(inventory)
 actual={str(p.relative_to(destination)):hashlib.sha256(p.read_bytes()).hexdigest() for p in destination.rglob('*') if p.is_file()};assert actual==expected,artifact.name
 rows.append(dict(overlay=artifact.name,status='passed',verified_files=len(actual),inventory_sha256=hashlib.sha256(inventory).hexdigest()))
(evidence/'reconstruction-check.json').write_text(json.dumps(dict(scope='Reconstructed exact package/runtime bytes from pinned installed source (including separately documented singleton fix), archived unified patch and runtime. Compared entire file set and every SHA; CPU-only, no performance claim.',checks=rows),indent=2)+'\n');print(rows)
