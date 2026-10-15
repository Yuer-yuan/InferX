"""Process-local Triton 3.7.1 bridge to the pinned MorphX LLVM18 pass.

This adapter creates real `_pk` kernels. It supplies no P/D admission policy,
changes no installed package, and makes no performance/whole-model claim.
"""
import hashlib
import json
from pathlib import Path
import re
import subprocess
import uuid


def llvm18_compatible_ir(source):
    triples=re.findall(r'^target triple = "([^"]+)"',source,re.MULTILINE)
    if triples and triples!=['nvptx64-nvidia-cuda']:
        raise ValueError('expected one NVPTX target triple')
    if re.search(r'captures\((?!none\))',source):
        raise ValueError('unsupported capture attribute')
    changes={'target_triple':int(not triples)}
    if not triples:source='target triple = "nvptx64-nvidia-cuda"\n'+source
    # These are optimization promises, not operations. Weakening them is safe;
    # unknown newer syntax is left for opt's parser to reject, never guessed.
    source,changes['captures_none']=re.subn(r'\bcaptures\(none\)', 'nocapture',source)
    source,changes['disjoint_or']=re.subn(r'\bor disjoint ', 'or ',source)
    source,changes['return_range']=re.subn(r'\brange\(i[0-9]+ -?[0-9]+, -?[0-9]+\) ', '',source)
    changes['gep_wrap_promises']=0
    def gep(match):
        flags=match.group(1).split()
        removed=sum(flag in {'nuw','nusw'} for flag in flags)
        changes['gep_wrap_promises']+=removed
        return 'getelementptr '+('inbounds ' if 'inbounds' in flags else '')
    source=re.sub(r'\bgetelementptr ((?:inbounds |nuw |nusw )+)',gep,source)
    return source,changes


def finalize_ptx(ptx,original,ptx_version,capability,keep_debug):
    names=re.findall(r'\.visible \.entry ([a-zA-Z_][a-zA-Z0-9_]*)',ptx)
    if sorted(names)!=sorted([original,original+'_pk']):
        raise ValueError(f'unexpected MorphX PTX entries: {names}')
    ptx=re.sub(r'\.version \d+\.\d+',f'.version {ptx_version//10}.{ptx_version%10}',ptx,flags=re.MULTILINE)
    ptx=re.sub(r'\.target sm_\d+',f'.target sm_{capability}',ptx,flags=re.MULTILINE)
    if not keep_debug:ptx=re.sub(r',\s*debug|debug,\s*','',ptx)
    return ptx,original


def compilation_key(plugin,revision='morphx-triton-v1'):
    return hashlib.sha256(Path(plugin).read_bytes()+Path(__file__).read_bytes()+revision.encode()).hexdigest()


def install(plugin,opt,artifacts):
    """Install before any Triton compilation in this owned Python process."""
    import triton
    from triton import knobs
    from triton._C.libtriton import llvm
    from triton.backends.nvidia.compiler import get_features,get_ptx_version_from_options,sm_arch_from_capability
    if triton.__version__!='3.7.1':raise ValueError('bridge qualified only against Triton3.7.1')
    if knobs.runtime.add_stages_inspection_hook is not None:raise RuntimeError('another Triton stage hook is active')
    plugin,opt=Path(plugin).resolve(strict=True),Path(opt).resolve(strict=True)
    artifacts=Path(artifacts).resolve();artifacts.mkdir(parents=True,exist_ok=True)
    key=compilation_key(plugin,hashlib.sha256(opt.read_bytes()).hexdigest())
    def hook(*args):
        if not args:return key,key
        backend,stages,options,language,capability=args
        if capability!=89:raise ValueError('bridge qualification is SM89 only')
        before=stages['llir'];state={}
        def instrument(src,metadata):
            original=before(src,metadata)
            kernels=re.findall(r'define[^\n]*?ptx_kernel[^@]*@([^ (]+)',original)
            if len(kernels)!=1:raise ValueError(f'expected one original IR kernel: {kernels}')
            state['original']=kernels[0]
            out=artifacts/uuid.uuid4().hex;out.mkdir(exist_ok=False);state['out']=out
            (out/'original.ll').write_text(original)
            compatible,changes=llvm18_compatible_ir(original);(out/'compatible.ll').write_text(compatible)
            command=[str(opt),'-load-pass-plugin='+str(plugin),'-passes=default<O3>','-S',str(out/'compatible.ll'),'-o',str(out/'morphable.ll')]
            done=subprocess.run(command,capture_output=True,text=True,timeout=180)
            (out/'opt.stderr').write_text(done.stderr)
            (out/'manifest.json').write_text(json.dumps(dict(key=key,command=command,returncode=done.returncode,changes=changes,kernel=kernels[0]),indent=2)+'\n')
            if done.returncode:raise RuntimeError('MorphX LLVM pass failed; see '+str(out/'opt.stderr'))
            return (out/'morphable.ll').read_text()
        def to_ptx(src,metadata):
            version=get_ptx_version_from_options(options,backend.target.arch)
            ptx=llvm.translate_to_asm(src,'nvptx64-nvidia-cuda',sm_arch_from_capability(capability),get_features(options,backend.target.arch),['nvptx-mad-wide-opt'],options.enable_fp_fusion,False)
            ptx,name=finalize_ptx(ptx,state['original'],version,capability,knobs.compilation.dump_ir_extract_di_local_variables)
            metadata['name']=name;(state['out']/'morphable.ptx').write_text(ptx)
            return ptx
        stages['llir']=instrument;stages['ptx']=to_ptx
    knobs.runtime.add_stages_inspection_hook=hook
    return key
