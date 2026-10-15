"""Fresh host-affinity-only variants; actual GPU kernels/policy are unchanged."""
import hashlib,json,os,shutil,subprocess
from pathlib import Path
from inferx.morphx_control import patch_agent_cpu_hint
r=Path(os.environ['INFERX_ROOT']);root=r/'morphx-control';audit=json.loads((root/'model-audit.json').read_text());assert all(x['status']=='passed' for x in audit['rows'])
for row in audit['rows']:
 variant=row['variant'];src=r/'sota-validation/runtime-query-abi-v4' if variant=='abi-only' else root/('runtime-'+variant+'-v1')
 original=json.loads((src/'manifest.json').read_text());assert original['library_sha256']==row['library_sha256']
 old_lib=Path(original['command'][original['command'].index('-o')+1]);assert hashlib.sha256(old_lib.read_bytes()).hexdigest()==row['library_sha256']
 for name,h in original['source_sha256'].items():assert hashlib.sha256((src/name).read_bytes()).hexdigest()==h,name
 destination=root/('runtime-'+variant+'-local-v2');library=root/'builds'/('libcuda_'+variant.replace('-','_')+'_local_v2.so');assert not library.exists();destination.mkdir(exist_ok=False)
 for name in ['runtime','include']:shutil.copytree(src/name,destination/name)
 before=(src/'runtime/src/cuda-llm.cpp').read_text();after=patch_agent_cpu_hint(before);(destination/'runtime/src/cuda-llm.cpp').write_text(after)
 command=[x.replace(str(src),str(destination)) for x in original['command']];command[command.index('-o')+1]=str(library);command[2]='40-43,96-99'
 with (destination/'build.log').open('x') as f:subprocess.run(command,stdout=f,stderr=subprocess.STDOUT,check=True)
 report=dict(command=command,library_sha256=hashlib.sha256(library.read_bytes()).hexdigest(),base_library_sha256=row['library_sha256'],variant=variant,
 source_sha256={str(p.relative_to(destination)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(destination.rglob('*')) if p.is_file()},
 changes=['Only host CPU bind helper differs; SMSCHED_AGENT_CPU overrides defaultCPU0 from thread start, checked/logged. Progress profiler, schedule_2, launch and numerical paths byte-identical.'],
 qualification='GPU math/kernel gates of exact pre-affinity base retained; actual full model/CPU36 requalification required before efficacy.')
 (destination/'manifest.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(dict(variant=variant,sha256=report['library_sha256'])),flush=True)
