"""Strict immutable-source exports; profiling is indexed separately."""
import gzip,hashlib,json,sys
from pathlib import Path
from scripts.analyze_pd_baselines import export
r=Path('/home/guokc/mypro/gnn-mount-sda1/InferX');base=r/'pd-strategies';raw=r/'pd-baselines/raw'
formal=['pd-strategies-matched-formal-r1','pd-strategies-simple-formal-r2','pd-strategies-simple-formal-final-r3','pd-strategies-pdmux-private-main-v1'];study=[];separate=[]
for p in sorted(raw.glob('pd-strategies-*/campaign.json')):
 c=json.loads(p.read_text());assert c['status']!='running',p
 assert c.get('restored'),p
 (separate if c.get('profiling_only') else study).append(p.parent.name)
assert all(n in study for n in formal)
result={}
for label,names in [('formal',formal),('all-study',study)]:
 index=base/f'{label}-index-final';index.mkdir(exist_ok=False)
 for name in names:(index/name).symlink_to(raw/name,target_is_directory=True)
 directories=[base/f'{label}-final-a',base/f'{label}-final-b']
 for destination in directories:export(index,destination)
 hashes={}
 for a in sorted(directories[0].iterdir()):
  b=directories[1]/a.name;assert a.read_bytes()==b.read_bytes(),a.name;hashes[a.name]=hashlib.sha256(a.read_bytes()).hexdigest()
 summary=json.loads((directories[0]/'summary.json').read_text())
 for name,expected in summary['source_hashes'].items():
  data=(index/name).read_bytes();assert hashlib.sha256(data).hexdigest()==expected['sha256'] and len(data)==expected['bytes'],name
 packed=base/f'{label}-summary.json.gz';packed.write_bytes(gzip.compress((directories[0]/'summary.json').read_bytes(),mtime=0))
 result[label]=dict(campaigns=names,index=str(index),canonical_export=str(directories[0]),byte_identical_files=hashes,input_files_verified=len(summary['source_hashes']),packed_summary=str(packed),packed_summary_sha256=hashlib.sha256(packed.read_bytes()).hexdigest())
result['profiling_only_excluded_from_performance']=separate
(base/'final-export-reproducibility.json').write_text(json.dumps(result,indent=2)+'\n');print({k:v.get('input_files_verified') for k,v in result.items() if isinstance(v,dict)})
