// Correctness fixture for the actual MorphX LLVM transformation on SM89.
#include <cuda_runtime.h>
extern "C" __global__ void morphx_probe(unsigned *out, unsigned count) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) out[i] = i * 3u + 7u;
}
