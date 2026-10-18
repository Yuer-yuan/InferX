from pathlib import Path
import json,hashlib,re,os
R=Path(__file__).resolve().parent
OLD=Path('/workspace/InferX/reproductions/selected100-single-node-20261018')
old=json.loads((OLD/'runtime-identity.json').read_text())['runtime_dependencies']
text=(R/'new-libraries-resolve-measured-binary.log').read_text();assert 'not found'not in text
paths=sorted({os.path.normpath(p)for p in re.findall(r'(?:=>\s+|^\s*)(/[^\s]+)',text,re.M)})
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb')as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
new={p:dict(bytes=Path(p).stat().st_size,sha256=sha(p),realpath=str(Path(p).resolve()))for p in paths}
assert len(old)==len(new)==55
comparisons=[]
for oldpath,identity in old.items():
 mapped=oldpath.replace('/workspace/InferX/envs/build',str(R/'envs/build'))
 assert mapped in new,(oldpath,mapped)
 comparisons.append(dict(old_path=oldpath,new_path=mapped,path_changed=oldpath!=mapped,old=identity,new=new[mapped],same_bytes=identity['bytes']==new[mapped]['bytes']and identity['sha256']==new[mapped]['sha256']))
oldtools=json.loads((OLD/'evaluation-dependency-check.json').read_text())['status']['tools'];newtools=json.loads((R/'original-runner-check.log').read_text())['tools']
norm=lambda k:re.sub(r'[-_.]+','-',k).lower()
packages={}
for tool in ('perf','accuracy'):
 before={norm(k):v for k,v in oldtools[tool]['packages'].items()};after={norm(k):v for k,v in newtools[tool]['packages'].items()}
 changed={k:dict(old=before[k],new=after.get(k))for k in before if after.get(k)!=before[k]}
 extra={k:v for k,v in after.items()if k not in before}
 packages[tool]=dict(before_count=len(before),after_count=len(after),changed_or_missing=changed,extra=extra,old_ready=oldtools[tool]['ready'],new_ready=newtools[tool]['ready'],python_old=oldtools[tool]['python_version'],python_new=newtools[tool]['python_version'])
 assert not changed,(tool,changed)
 assert set(extra)<= {'pip'},(tool,extra)
 assert oldtools[tool]['version']==newtools[tool]['version']
result=dict(status='COMPARED',measured_environment=str(OLD),fresh_prefix=str(R),scope='664 performance/quality used original installed environments. This comparison follows fresh installation and CPU original runner/data checks only; no engine rebuild or GPU replay in the new environment.',runtime_libraries=comparisons,library_count=55,same_library_bytes=sum(x['same_bytes']for x in comparisons),relocated_paths=sum(x['path_changed']for x in comparisons),evaluation_packages=packages)
(R/'old-vs-new-environment-comparison.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items()if k!='runtime_libraries'},indent=2))
