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


if __name__ == '__main__':
    main()
