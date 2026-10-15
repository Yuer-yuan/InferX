"""Opt-in mapped-host control pool for a MorphX port without gdrdrv.

This changes the control transport, so measured results must identify this port
and cannot be presented as reproduction of the upstream GDRCopy configuration.
"""
import os
import subprocess
from pathlib import Path


def main():
    root = Path(os.environ['INFERX_ROOT'])
    source = root/'source/pd-baselines/MorphX'
    path = source/'include/common.h'
    original = subprocess.check_output(['git','show','HEAD:include/common.h'],cwd=source,text=True)
    ported = original.replace('#include <gdrapi.h>', '''#include <gdrapi.h>
#include <atomic>
#include <cstring>
#ifdef SMSCHED_MAPPED_HOST_CONTROL
inline int mapped_control_copy(void *destination, const void *source, size_t count) {
    std::atomic_thread_fence(std::memory_order_seq_cst);
    std::memcpy(destination, source, count);
    std::atomic_thread_fence(std::memory_order_seq_cst);
    return 0;
}
#define gdr_copy_to_mapping(handle,destination,source,count) mapped_control_copy(destination,source,count)
#define gdr_copy_from_mapping(handle,destination,source,count) mapped_control_copy(destination,source,count)
#endif''')
    start = ported.index('        g = gdr_open();')
    end = ported.index('        for(size_t i = 0;', start)
    ported = (ported[:start]+'''#ifdef SMSCHED_MAPPED_HOST_CONTROL
        void *host = nullptr;
        CHECK_CUDA_ERROR(cudaHostAlloc(&host, GDR_POOL_SIZE, cudaHostAllocMapped));
        CHECK_CUDA_ERROR(cudaHostGetDevicePointer((void**)&d_pool, host, 0));
        h_pool = reinterpret_cast<uintptr_t>(host);
        std::memset(host, 0, GDR_POOL_SIZE);
#else
'''+ported[start:end]+'#endif\n'+ported[end:])
    start = ported.index('    ~GdrPool() {')+len('    ~GdrPool() {')
    end = ported.index('\n    }', start)
    ported = (ported[:start]+'''
#ifdef SMSCHED_MAPPED_HOST_CONTROL
        CHECK_CUDA_ERROR(cudaFreeHost(reinterpret_cast<void*>(h_pool)));
#else'''+ported[start:end]+'\n#endif'+ported[end:])
    if path.read_text() not in (original, ported):
        raise RuntimeError('unexpected common.h edits; refusing overwrite')
    path.write_text(ported)
    (root/'pd-baselines/builds/morphx-mapped-transport.patch').write_text(
        subprocess.check_output(['git','diff','--','include/common.h'],cwd=source,text=True))
    path=source/'runtime/src/cuda-llm.cpp'
    original=subprocess.check_output(['git','show','HEAD:runtime/src/cuda-llm.cpp'],cwd=source,text=True)
    guard='''    if(stream == 0) {
        std::cerr << "Default stream is not supported" << std::endl;
        exit(EXIT_FAILURE);
    }'''
    assert guard in original
    legacy=original.replace(guard,'#ifndef SMSCHED_ALLOW_DEFAULT_STREAM\n'+guard+'\n#endif',1)
    ported=legacy.replace(
        'std::unordered_map<const char *, std::atomic<float>> gCI;\n'
        'std::unordered_map<const char *, std::atomic<float>> gWaveTime;',
        '''#ifdef SMSCHED_THREADSAFE_PROFILE
#include "morphx_profile_table.hpp"
// The agent can still read these tables while exit handlers drain CUDA.
// Keep them alive until process teardown, and serialize map insertion/rehash.
auto& gCI = *new MorphxProfileTable;
auto& gWaveTime = *new MorphxProfileTable;
#else
std::unordered_map<const char *, std::atomic<float>> gCI;
std::unordered_map<const char *, std::atomic<float>> gWaveTime;
#endif''',1)
    publication='        streamInfo->pending->push(launchInfo);'
    assert ported.count(publication)==1
    ported=ported.replace(publication,'''#ifndef SMSCHED_THREADSAFE_PROFILE
'''+publication+'\n#endif',1)
    boundary='''    for(int i = 0; i < numArgs - 6; ++i) {
        free(nargs[i]);'''
    assert boundary in ported
    ported=ported.replace(boundary,'''#ifdef SMSCHED_THREADSAFE_PROFILE
    // Publishing transfers ownership to the agent; do not touch launchInfo after.
    if(!profile) streamInfo->pending->push(launchInfo);
#endif
'''+boundary,1)
    threadsafe=ported
    launch_name='    CU_CHECK(real_cuFuncGetName(&funcName, f));'
    assert ported.count(launch_name)==1
    ported=ported.replace(launch_name,launch_name+'''
#ifdef SMSCHED_CAPTURE_BYPASS
    if(hStream != nullptr) {
        // Captured kernels would retain per-launch control buffers after the
        // agent frees them. Preserve native graph nodes instead of patching.
        using CaptureQuery = CUresult (*)(CUstream, CUstreamCaptureStatus*);
        static auto queryCapture = reinterpret_cast<CaptureQuery>(
            dlsym(shared_lib, "cuStreamIsCapturing"));
        CUstreamCaptureStatus capture = CU_STREAM_CAPTURE_STATUS_NONE;
        if(!queryCapture || queryCapture(hStream, &capture) != CUDA_SUCCESS ||
                capture != CU_STREAM_CAPTURE_STATUS_NONE) {
            traceLLMKernel("capture-bypass", funcName, gridSize, blockDimX,
                blockDimY, blockDimZ, sharedMemBytes, hStream);
            launchMutex.unlock();
            return real_cuLaunchKernel(f, gridDimX, gridDimY, gridDimZ,
                blockDimX, blockDimY, blockDimZ, sharedMemBytes,
                hStream, kernelParams, extra);
        }
    }
#endif''',1)
    if path.read_text() not in (original,legacy,threadsafe,ported):
        raise RuntimeError('unexpected runtime edits')
    path.write_text(ported)


if __name__ == '__main__':
    main()
