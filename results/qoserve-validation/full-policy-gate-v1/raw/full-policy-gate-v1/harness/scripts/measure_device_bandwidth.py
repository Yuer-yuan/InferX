"""Empirical GPU1 D2D-copy bandwidth reference; no inference DRAM-counter claim."""
import ctypes as c
import json
import os
import statistics
import subprocess
from pathlib import Path

from inferx.concurrency import bandwidth_bytes_per_s

root = Path(os.environ['INFERX_ROOT'])
directory = root/'concurrency-roofline'
if os.environ.get('CUDA_VISIBLE_DEVICES') != 'GPU-4d9bef9d-631c-bb36-3ff4-589d6a184189':
    raise RuntimeError('GPU1 UUID selection required before CUDA initialization')
runtime = c.CDLL(str(root/'toolchains/cuda-12.8.1/lib/libcudart.so.12'))
signatures = {
    'cudaSetDevice': [c.c_int], 'cudaMalloc': [c.POINTER(c.c_void_p), c.c_size_t],
    'cudaFree': [c.c_void_p], 'cudaMemset': [c.c_void_p, c.c_int, c.c_size_t],
    'cudaMemcpyAsync': [c.c_void_p, c.c_void_p, c.c_size_t, c.c_int, c.c_void_p],
    'cudaMemcpy': [c.c_void_p, c.c_void_p, c.c_size_t, c.c_int],
    'cudaDeviceSynchronize': [], 'cudaEventCreate': [c.POINTER(c.c_void_p)],
    'cudaEventDestroy': [c.c_void_p], 'cudaEventRecord': [c.c_void_p, c.c_void_p],
    'cudaEventSynchronize': [c.c_void_p],
    'cudaEventElapsedTime': [c.POINTER(c.c_float), c.c_void_p, c.c_void_p]}
for name, signature in signatures.items():
    getattr(runtime, name).argtypes = signature
    getattr(runtime, name).restype = c.c_int
runtime.cudaGetErrorString.argtypes = [c.c_int]
runtime.cudaGetErrorString.restype = c.c_char_p


def call(name, *args):
    error = getattr(runtime, name)(*args)
    if error:
        raise RuntimeError(name + ': ' + runtime.cudaGetErrorString(error).decode())


size, copies = 256*1024*1024, 64
buffers, events, samples = [], [], []
result = dict(buffer_bytes=size, copies_per_sample=copies, samples=5,
    traffic_definition='2*payload_bytes per D2D copy, accounting for read and write',
    limitation='Copy bandwidth is not model DRAM traffic or arithmetic intensity.',
    cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'))
try:
    call('cudaSetDevice', 0)
    for _ in range(2):
        p = c.c_void_p()
        call('cudaMalloc', c.byref(p), size)
        buffers.append(p)
    call('cudaMemset', buffers[0], 0xA5, size)
    call('cudaMemset', buffers[1], 0, size)
    call('cudaDeviceSynchronize')
    for _ in range(2):
        e = c.c_void_p()
        call('cudaEventCreate', c.byref(e))
        events.append(e)
    for _ in range(8):
        call('cudaMemcpyAsync', buffers[1], buffers[0], size, 3, None)
    call('cudaDeviceSynchronize')
    for _ in range(5):
        call('cudaEventRecord', events[0], None)
        for _ in range(copies):
            call('cudaMemcpyAsync', buffers[1], buffers[0], size, 3, None)
        call('cudaEventRecord', events[1], None)
        call('cudaEventSynchronize', events[1])
        elapsed = c.c_float()
        call('cudaEventElapsedTime', c.byref(elapsed), *events)
        samples.append(dict(elapsed_ms=elapsed.value,
                            bytes_per_s=bandwidth_bytes_per_s(size, copies, elapsed.value)))
    check = (c.c_ubyte*16)()
    call('cudaMemcpy', check, buffers[1], 16, 2)
    assert bytes(check) == b'\xa5'*16
    result.update(ok=True, copy_verified=True, measurements=samples,
                  median_GB_per_s=statistics.median(x['bytes_per_s'] for x in samples)/1e9)
    result['gpu_after'] = subprocess.check_output(['nvidia-smi', '-i', '1',
        '--query-gpu=uuid,clocks.sm,clocks.mem,pstate,power.draw', '--format=csv,noheader'], text=True).strip()
except Exception as exc:
    result.update(ok=False, error=f'{type(exc).__name__}: {exc}')
    raise
finally:
    for e in events:
        runtime.cudaEventDestroy(e)
    for p in buffers:
        runtime.cudaFree(p)
    (directory/'bandwidth.json').write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
