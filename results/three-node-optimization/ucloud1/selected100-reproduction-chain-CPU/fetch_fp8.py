#!/usr/bin/env python3
"""Future acquisition only; pinned per-file revision and full content SHA."""
import argparse,hashlib,json,pathlib,urllib.parse,urllib.request,os
p=argparse.ArgumentParser();p.add_argument('--directory',type=pathlib.Path,required=True)
p.add_argument('--manifest',type=pathlib.Path,default=pathlib.Path(__file__).with_name('FP8-acquisition-manifest.json'))
a=p.parse_args();m=json.loads(a.manifest.read_text());a.directory.mkdir(parents=True,exist_ok=True)
def hash_file(f):
    h=hashlib.sha256()
    with f.open('rb') as s:
        for b in iter(lambda:s.read(8<<20),b''):h.update(b)
    return h.hexdigest()
for v in m['files']:
    if not v['per_file_revision']:continue  # .mv/.msc describe the old mount, not model content
    name=v['file'];assert pathlib.PurePosixPath(name).name==name
    target=a.directory/name
    if target.exists():
        assert target.stat().st_size==v['bytes'] and hash_file(target)==v['sha256'],name
        print(json.dumps({'file':name,'existing_verified':True}),flush=True);continue
    partial=target.with_name(target.name+'.partial');assert not partial.exists(),str(partial)
    url='https://modelscope.cn/models/'+m['repo']+'/resolve/'+v['per_file_revision']+'/'+urllib.parse.quote(name)
    h=hashlib.sha256();size=0
    with urllib.request.urlopen(url,timeout=180) as response,partial.open('xb') as out:
        for b in iter(lambda:response.read(8<<20),b''):
            size+=len(b);assert size<=v['bytes'],name;h.update(b);out.write(b)
    assert size==v['bytes'] and h.hexdigest()==v['sha256'],name+' content mismatch; partial retained'
    os.replace(partial,target);print(json.dumps({'file':name,'bytes':size,'sha256':h.hexdigest(),'url':url}),flush=True)
