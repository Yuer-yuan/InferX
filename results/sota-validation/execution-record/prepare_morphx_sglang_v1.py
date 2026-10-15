import hashlib,json,os,shutil
from pathlib import Path
from inferx.sglang_pd import prepare_overlay
from inferx.morphx_sglang import patch_primary_stream_groups
r=Path(os.environ['INFERX_ROOT']);dest=r/'sota-validation/overlays/morphx-pd-v1'
source=r/'pd-baselines/sglang0519-venv/lib/python3.12/site-packages/sglang'
manifest=prepare_overlay(source,dest,'pdmux-chunked',128,True,True,False,True)
p=dest/'sglang/srt/multiplex/pdmux_context.py';before=p.read_bytes();p.write_text(patch_primary_stream_groups(before.decode()))
manifest['primary_stream_context_patch']=dict(before_sha256=hashlib.sha256(before).hexdigest(),after_sha256=hashlib.sha256(p.read_bytes()).hexdigest())
shutil.copy2(r/'code-pd-baselines/inferx/morphx_sglang.py',dest/'inferx_morphx_runtime.py')
(dest/'sitecustomize.py').write_text('''import os
if os.environ.get('INFERX_MORPHX_COMPILE')=='1':
    try:
        from pathlib import Path
        from inferx.morphx_triton import install
        root=Path(os.environ['INFERX_ROOT'])
        install(root/'pd-baselines/builds/morphx-pass-finish-barrier/libsmsched.so','/usr/bin/opt-18',Path(os.environ['INFERX_MORPHX_IR'])/str(os.getpid()))
        print('MORPHX_TRITON_INSTALLED pid='+str(os.getpid()),flush=True)
    except BaseException:
        import traceback
        traceback.print_exc()
        os._exit(78)
''')
# Archive every added/changed source, leaving installed and old overlays untouched.
inventory={str(p.relative_to(dest)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(dest.rglob('*')) if p.is_file() and p.name not in {'source-hashes.json','overlay-manifest.json'}}
(dest/'source-hashes.json').write_text(json.dumps(inventory,indent=2)+'\n')
manifest.update(method='MorphX-on-SGLang port; actual compiler/profiler/dynamic SM runtime, primary P/D streams; SGLang admission and hybrid state',source_hashes_sha256=hashlib.sha256((dest/'source-hashes.json').read_bytes()).hexdigest(),extra_sources={k:v for k,v in inventory.items() if k in ['sitecustomize.py','inferx_morphx_runtime.py']})
(dest/'overlay-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(str(dest))
