"""Separate diagnostic overlay adds CUPTI stream IDs only; formal source frozen."""
import os,json,hashlib,shutil
from pathlib import Path
r=Path(os.environ['INFERX_ROOT']);source=r/'sota-validation/overlays/morphx-pd-v2';out=r/'sota-validation/overlays/morphx-pd-profile-v3';assert not out.exists();shutil.copytree(source,out,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
p=out/'inferx_morphx_runtime.py'
p.write_text(p.read_text()+'''
# Diagnostic-only stream identity mapping. No scheduling change.
_initialize_without_cupti_ids=initialize_groups
def initialize_groups(namespace,gpu_id,config,torch_module=None):
    _initialize_without_cupti_ids(namespace,gpu_id,config,torch_module)
    import ctypes,json,os
    from pathlib import Path
    root=Path(os.environ['INFERX_ROOT'])
    cupti=ctypes.CDLL(str(root/'pd-baselines/sglang0519-venv/lib/python3.12/site-packages/nvidia/cuda_cupti/lib/libcupti.so.12'))
    api=cupti.cuptiGetStreamIdEx
    api.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_uint8,ctypes.POINTER(ctypes.c_uint32)];api.restype=ctypes.c_int
    mapping={}
    for role,stream in zip(['prefill','decode'],namespace['STREAM_GROUPS'][0]):
        value=ctypes.c_uint32()
        code=api(None,stream.cuda_stream,0,ctypes.byref(value))
        if code:raise RuntimeError('CUPTI stream mapping error '+str(code))
        mapping[role]={'handle':stream.cuda_stream,'cupti_stream_id':value.value}
    folder=Path(os.environ['INFERX_MORPHX_STREAM_MAP']);folder.mkdir(parents=True,exist_ok=True)
    with (folder/(str(os.getpid())+'.json')).open('x') as f:json.dump({'pid':os.getpid(),'streams':mapping},f,indent=2)
    logging.getLogger('inferx.morphx').info('MORPHX_CUPTI_IDS %s',json.dumps(mapping,sort_keys=True))
''')
manifest=json.loads((out/'overlay-manifest.json').read_text());manifest['diagnostic_only']='CUPTI stream identity logging; exclude this profiled lane from performance comparisons.'
inventory={str(p.relative_to(out)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(out.rglob('*')) if p.is_file() and p.name not in {'source-hashes.json','overlay-manifest.json'}};(out/'source-hashes.json').write_text(json.dumps(inventory,indent=2)+'\n');manifest['source_hashes_sha256']=hashlib.sha256((out/'source-hashes.json').read_bytes()).hexdigest();(out/'overlay-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(out)
