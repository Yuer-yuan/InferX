"""Rehash eight immutable runtime trees/libraries and all source-capsule payloads."""
import hashlib,json,os,time
from pathlib import Path
r=Path(os.environ['INFERX_ROOT']);root=r/'morphx-control';capsule=root/'source-capsule-v1'
m=json.loads((capsule/'capsule-manifest.json').read_text());payload={}
for name,entry in m['payload'].items():
 p=capsule/name;h=hashlib.sha256(p.read_bytes()).hexdigest();assert h==entry['sha256'],name;payload[name]=h
builds=[];source_count=0
for entry in m['builds']:
 p=r/entry['path'];manifest=json.loads((p/'manifest.json').read_text());sources={}
 for name,h in manifest['source_sha256'].items():
  assert hashlib.sha256((p/name).read_bytes()).hexdigest()==h,name;sources[name]=h;source_count+=1
 library=r/entry['library'];actual=hashlib.sha256(library.read_bytes()).hexdigest();assert actual==entry['library_sha256']==manifest['library_sha256']
 builds.append(dict(**entry,source_sha256=sources,manifest_sha256=hashlib.sha256((p/'manifest.json').read_bytes()).hexdigest()))
report=dict(verified_unix=time.time(),capsule_payload_count=len(payload),capsule_payload_sha256=payload,builds=builds,runtime_source_checks=source_count,
 scope='Eight immutable MorphX control runtime trees and shared libraries plus185capsulepayloads. Reused LLVM/Marlin/SGLang dependencies remain pinned by the predecessor source inventories; not a claim to rehash every installed venv file.')
assert len(payload)==185 and len(builds)==8
(root/'final-source-verification.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n');print(json.dumps(dict(payload_files=len(payload),builds=len(builds),runtime_source_checks=source_count)))
