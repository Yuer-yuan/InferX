"""CPU-only parser/IR verification of cached real Triton kernels, no execution."""
import os,json,hashlib,re,subprocess,concurrent.futures
from pathlib import Path
from inferx.morphx_triton import llvm18_compatible_ir
r=Path(os.environ['INFERX_ROOT']);out=r/'sota-validation/ir-preflight-v2';out.mkdir(exist_ok=False)
seen={}
for base in [r/'cache/triton',r/'sota-validation/cache/triton-port-v1',r/'sota-validation/cache/triton-port-v2']:
 if not base.exists():continue
 for p in base.rglob('*.llir'):
  raw=p.read_bytes();seen.setdefault(hashlib.sha256(raw).hexdigest(),(p,raw))
print('unique IR modules',len(seen),flush=True)
def verify(item):
 sha,(p,raw)=item;entry=dict(source=str(p),source_sha256=sha)
 try:
  source,changes=llvm18_compatible_ir(raw.decode());entry['changes']=changes
  done=subprocess.run(['/usr/bin/opt-18','-passes=verify','-disable-output','-'],input=source,text=True,capture_output=True,timeout=30)
  entry['returncode']=done.returncode
  if done.returncode:
   (out/(sha+'.ll')).write_text(source);(out/(sha+'.stderr')).write_text(done.stderr);entry['error']=done.stderr[:4000]
 except Exception as exc:entry['error']=repr(exc);entry['returncode']=-1
 return entry
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:results=list(pool.map(verify,seen.items()))
report=dict(scope='CPU parser/IR verifier only; neither MorphX pass/codegen nor GPU/model qualification',count=len(results),failed=sum(e['returncode']!=0 for e in results),cases=results);(out/'result.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='cases'}),flush=True)
errors=sorted(set(e.get('error','').splitlines()[0] for e in results if e['returncode']))
print(json.dumps(dict(unique_error_heads=errors)),flush=True)
