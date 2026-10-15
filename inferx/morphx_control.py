"""Immutable MorphX control-port variants; no replacement of its scheduling policy."""
import hashlib
import json
from pathlib import Path
import shutil

from inferx.morphx_validation import upgrade_proc_address_queries

BASE_LIBRARY_SHA256='64f3268499a920e30db33d48b8ecee9137216e3dd494e0a93d0e2119a542af3a'
BASE_RUNTIME_SHA256='42c4893adb412045b543dcf4a9f777d851fd155e646f58c31f9cb3ca9fb4e399'
BASE_CONFIG_SHA256='ab40127993593d35fc4acf2ef02219384ebbe65a495427f8391ee171bb8db6d7'


def load_qualified_base_case(root: Path) -> dict:
    """Read the frozen HDD configuration before creating a run or stopping a service."""
    raw=(Path(root)/'sota-validation/configs/sota-validation.json').read_bytes()
    if hashlib.sha256(raw).hexdigest()!=BASE_CONFIG_SHA256:
        raise ValueError('base config hash mismatch')
    return next(c for c in json.loads(raw)['cases'] if c['id']=='morphx-copy96-no-launch-cap')

_DRIVER=r'''
#include "morphx_control_ops.hpp"
static inferx_control::Driver& inferxControlDriver() {
    static inferx_control::Driver driver{
        reinterpret_cast<decltype(inferx_control::Driver::fill32)>(dlsym(shared_lib,"cuMemsetD32Async")),
        reinterpret_cast<decltype(inferx_control::Driver::create_event)>(dlsym(shared_lib,"cuEventCreate")),
        reinterpret_cast<decltype(inferx_control::Driver::record_event)>(dlsym(shared_lib,"cuEventRecord")),
        reinterpret_cast<decltype(inferx_control::Driver::query_event)>(dlsym(shared_lib,"cuEventQuery")),
        reinterpret_cast<decltype(inferx_control::Driver::destroy_event)>(dlsym(shared_lib,"cuEventDestroy_v2"))};
    if(!driver.valid()) {fprintf(stderr,"MorphX control: missing real driver entry\n");std::abort();}
    return driver;
}
'''


def patch_control_runtime(source: str, *, device_fill: bool, event_completion: bool) -> str:
    if not (device_fill or event_completion) or 'inferxControlDriver' in source:
        raise ValueError('unexpected or already patched control source')
    if source.count('driver-selected self-query ABI')!=2:
        raise ValueError('correct query ABI before transport adaptation')
    def replace(old,new):
        nonlocal source
        if source.count(old)!=1:raise ValueError('unexpected control source anchor')
        source=source.replace(old,new,1)
    replace('struct KernelInfo {',_DRIVER+'\nstruct KernelInfo {')
    if device_fill:
        replace('    CUDA_CHECK(cudaMemcpyAsync(launchInfo->d_agents, &agents, sizeof(int), cudaMemcpyHostToDevice, hStream));',
                '    CU_CHECK(inferx_control::initialize_agents(inferxControlDriver(), reinterpret_cast<CUdeviceptr>(launchInfo->d_agents), static_cast<unsigned int>(agents), hStream));')
    if event_completion:
        replace('    void* d_agents;','    void* d_agents;\n    inferx_control::Completion completion;')
        start=source.index('            uint64_t tmp = gGdrPool->get(kernels[i]->fetched);',source.index('void agentThread()'))
        end=source.index('            if(finished >= kernels[i]->gridSize) {',start)
        replace(source[start:end+len('            if(finished >= kernels[i]->gridSize) {')],
            '            bool complete=false;\n            CU_CHECK(kernels[i]->completion.poll(inferxControlDriver(), complete));\n            if(complete) {\n                CU_CHECK(kernels[i]->completion.release(inferxControlDriver()));')
        replace('    if(profile) {\n        profile->join();',
            '    // Record on the actual launch stream before publishing ownership.\n'
            '    if(!profile) {\n'
            '        CU_CHECK(launchInfo->completion.record_after_launch(inferxControlDriver(), hStream, res));\n'
            '    }\n\n    if(profile) {\n        profile->join();')
        replace('    if(!profile) streamInfo->pending->push(launchInfo);',
            '    if(!profile && !streamInfo->pending->push(launchInfo)) {\n'
            '        fprintf(stderr,"MorphX control: pending queue full\\n");std::abort();\n    }')
        replace('#include "morphx_control_ops.hpp"',
            '#ifndef SMSCHED_THREADSAFE_PROFILE\n#error Event completion requires ownership-safe publication\n#endif\n#include "morphx_control_ops.hpp"')
    return source


def prepare_control_runtime(source: Path, destination: Path, *, device_fill: bool, event_completion: bool) -> dict:
    source=Path(source).resolve();destination=Path(destination).resolve()
    if source.is_relative_to(destination) or destination.is_relative_to(source):
        raise ValueError('source/destination overlap')
    if destination.exists():raise FileExistsError(destination)
    manifest=json.loads((source/'manifest.json').read_text())
    if manifest['library_sha256']!=BASE_LIBRARY_SHA256:
        raise ValueError('unexpected base library hash')
    for name,expected in manifest['source_sha256'].items():
        relative=Path(name)
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('unexpected source hash path')
        if hashlib.sha256((source/relative).read_bytes()).hexdigest()!=expected:
            raise ValueError('base source hash mismatch: '+name)
    original=(source/'runtime/src/cuda-llm.cpp').read_bytes()
    if hashlib.sha256(original).hexdigest()!=BASE_RUNTIME_SHA256:
        raise ValueError('unexpected pinned runtime hash')
    changed=patch_control_runtime(upgrade_proc_address_queries(original.decode()),device_fill=device_fill,event_completion=event_completion)
    destination.parent.mkdir(parents=True,exist_ok=True);destination.mkdir(exist_ok=False)
    try:
        for name in ['runtime','include']:shutil.copytree(source/name,destination/name)
        (destination/'runtime/src/cuda-llm.cpp').write_text(changed)
        shutil.copyfile(Path(__file__).parents[1]/'scripts/morphx_control_ops.hpp',destination/'include/morphx_control_ops.hpp')
        hashes={str(p.relative_to(destination)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(destination.rglob('*')) if p.is_file()}
        result=dict(base_library_sha256=manifest['library_sha256'],base_runtime_sha256=BASE_RUNTIME_SHA256,
            device_fill=device_fill,event_completion=event_completion,source_sha256=hashes,
            qualification='Prepared only; build and actual GPU/model gates required.',
            retained='Original progress profiler, schedule_2, live SM bounds, synchronized counter initialization, native launch semantics.')
        (destination/'preparation.json').write_text(json.dumps(result,indent=2)+'\n');return result
    except BaseException:
        shutil.rmtree(destination);raise
