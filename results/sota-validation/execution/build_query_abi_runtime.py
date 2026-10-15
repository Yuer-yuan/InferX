"""Build ABI correction into a new immutable library; no GPU qualification claim."""
import hashlib,json,os,shutil,subprocess
from pathlib import Path
from inferx.morphx_validation import upgrade_proc_address_queries
r=Path(os.environ['INFERX_ROOT']);src=r/'sota-validation/runtime-cuda-copy-v2';out=r/'sota-validation/runtime-query-abi-v4'
manifest=json.loads((src/'manifest.json').read_text())
for name,h in manifest['source_sha256'].items():assert hashlib.sha256((src/name).read_bytes()).hexdigest()==h,name
library=r/'sota-validation/builds/libcuda_query_abi_v4.so';assert not library.exists()
changed=upgrade_proc_address_queries((src/'runtime/src/cuda-llm.cpp').read_text());out.mkdir(exist_ok=False)
for name in ['runtime','include']:shutil.copytree(src/name,out/name)
(out/'runtime/src/cuda-llm.cpp').write_text(changed)
command=[str(out/'runtime/src/cuda-llm.cpp') if x==str(src/'runtime/src/cuda-llm.cpp') else '-I'+str(out/'include') if x=='-I'+str(src/'include') else x for x in manifest['command']]
command[command.index('-o')+1]=str(library);command[command.index('-c')+1]='40-43,96-99'
with (out/'build.log').open('x') as f:subprocess.run(command,stdout=f,stderr=subprocess.STDOUT,check=True)
result=dict(base_library_sha256=manifest['library_sha256'],command=command,
 library_sha256=hashlib.sha256(library.read_bytes()).hexdigest(),
 source_sha256={str(p.relative_to(out)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(out.rglob('*')) if p.is_file()},
 changes=['Real legacy four-argument query signature/typedef/call; native-resolved ABI selects four/five-argument self-query replacement; unknown ABI retains native pointer.'],
 qualification='CPU typed four/five-argument oracle and real toolchain compilation only; actual GPU/model requalification required before use. Historical measured v2 unchanged.')
(out/'manifest.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
