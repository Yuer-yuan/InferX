#!/usr/bin/env python3
"""CPU-only cross-check of reproduction excerpts, source pins and document commands."""
import hashlib,json,re,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4]
STAGE=Path(__file__).resolve().parent

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def subset(actual,excerpt,path=''):
    if isinstance(excerpt,dict):
        assert isinstance(actual,dict),path
        for key,value in excerpt.items():
            assert key in actual,(path,key)
            subset(actual[key],value,path+'.'+key)
    else:assert type(actual)==type(excerpt) and actual==excerpt,(path,actual,excerpt)

def main():
    data=json.loads((ROOT/'submission/basic/presentation/clean-reproduction.json').read_text())
    for rel,spec in data['sources'].items():
        p=ROOT/rel
        assert p.stat().st_size==spec['bytes'] and sha(p)==spec['sha256'],rel
    source=ROOT/'results/three-node-optimization/ucloud3/selected100-single-node-reproduction-20261018'
    perf=source/'performance-delivery/performance-C14-001/summary.json'
    quality=source/'quality-delivery/quality100-002/evaluation/summary.json'
    subset(json.loads(perf.read_text()),json.loads(data['performance_json']))
    subset(json.loads(quality.read_text()),json.loads(data['quality_json']))
    docs=[
        'submission/track-a/reproduction.md','submission/track-a/reproduction-en.md',
        'submission/track-a/quantization-reproduction.md','submission/track-a/quantization-reproduction-en.md',
        'submission/basic/defense-notes.md','submission/basic/defense-notes-en.md',
        'submission/basic/presentation/README.md','submission/README.md','submission/REPRODUCE.md',
        'submission/track-a/README.md','submission/records/2026-10-18-submission-checklist.md',
        'submission/basic/demo.md','demo/selected100/README.md',
        'submission/track-a/benchmark.md','submission/track-a/model-selection.md',
        'submission/track-a/performance-analysis.md','submission/track-a/experiment-index.md']
    reports=[]
    for rel in docs:
        p=ROOT/rel;s=p.read_text();broken=[];blocks=0
        for link in re.findall(r'\]\(([^)]+)\)',s):
            if re.match(r'^(https?://|#|mailto:)',link):continue
            if not (p.parent/link.split('#')[0]).exists():broken.append(link)
        assert not broken,(rel,broken)
        for block in re.findall(r'```(?:bash|sh)\n(.*?)\n```',s,re.S):
            blocks+=1
            r=subprocess.run(['bash','-n'],input=block,text=True,capture_output=True)
            assert r.returncode==0,(rel,blocks,r.stderr)
        reports.append(dict(path=rel,sha256=sha(p),bytes=p.stat().st_size,shell_blocks=blocks,relative_links='PASS'))
    notes={}
    for lang,file,pattern in [('zh','defense-notes.md',r'^### (\d+)｜.*?｜(\d+)秒(?:主讲)?$'),('en','defense-notes-en.md',r'^## (\d+)\. .*? — (\d+)s(?: main talk)?$')]:
        s=(ROOT/'submission/basic'/file).read_text();items=[(int(n),int(t)) for n,t in re.findall(pattern,s,re.M)]
        assert [n for n,_ in items]==list(range(1,32)),(lang,items)
        spoken_seconds=sum(t for _,t in items)
        assert spoken_seconds==288
        assert ('腾出的12秒' in s if lang=='zh' else 'frees 12 seconds' in s)
        notes[lang]=dict(pages=len(items),page_seconds=spoken_seconds,transitions_and_pointing_seconds=12,total_seconds=300)
    result=dict(status='PASS',scope='Actual raw JSON subsets, fixed source hashes, document shell syntax/relative links, and narration budget. No GPU run or dependency install performed.',source_files=len(data['sources']),performance_source=str(perf.relative_to(ROOT)),quality_source=str(quality.relative_to(ROOT)),documents=reports,narration=notes)
    (STAGE/'material-content-audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print('PASS:',len(data['sources']),'source pins; raw excerpts exact;',len(reports),'documents;',sum(x['shell_blocks'] for x in reports),'shell blocks; 31 slides, 288s pages +12s transitions each language.')
if __name__=='__main__':main()
