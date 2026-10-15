// Independent integer oracle for BF16 atomics, including neighbouring16-bit lanes.
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <vector>
#define CUDA_CHECK(expr) do { cudaError_t e=(expr); if(e!=cudaSuccess) { \
    std::fprintf(stderr,"%s: %s\n",#expr,cudaGetErrorString(e)); return 1; } } while(0)
__global__ void accumulate(__nv_bfloat16 *values, unsigned *old_values, unsigned count) {
    unsigned i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<count) old_values[i]=(unsigned)__bfloat162float(
        atomicAdd(&values[i%256],__float2bfloat16(1.f)));
}
int main() {
    CUdevice dev; CUuuid uuid;
    if(cuInit(0)!=CUDA_SUCCESS || cuDeviceGet(&dev,0)!=CUDA_SUCCESS ||
       cuDeviceGetUuid(&uuid,dev)!=CUDA_SUCCESS) return 1;
    const unsigned char expected[]={0x4d,0x9b,0xef,0x9d,0x63,0x1c,0xbb,0x36,
                                   0x3f,0xf4,0x58,0x9d,0x6a,0x18,0x41,0x89};
    if(std::memcmp(uuid.bytes,expected,16)) { std::fputs("refusing foreignGPU\n",stderr); return 4; }
    constexpr unsigned count=16384;
    __nv_bfloat16 *values; unsigned *old_values;
    cudaStream_t stream; CUDA_CHECK(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
    CUDA_CHECK(cudaMalloc(&values,256*sizeof(*values)));
    CUDA_CHECK(cudaMalloc(&old_values,count*sizeof(*old_values)));
    CUDA_CHECK(cudaMemsetAsync(values,0,256*sizeof(*values),stream));
    accumulate<<<count/256,256,0,stream>>>(values,old_values,count);
    CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaStreamSynchronize(stream));
    std::vector<__nv_bfloat16> out(256); std::vector<unsigned> returned(count);
    CUDA_CHECK(cudaMemcpy(out.data(),values,256*sizeof(*values),cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(returned.data(),old_values,count*sizeof(*old_values),cudaMemcpyDeviceToHost));
    for(unsigned bucket=0;bucket<256;++bucket) {
        if(__bfloat162float(out[bucket])!=64.f) return 3;
        std::vector<unsigned> history;
        for(unsigned i=bucket;i<count;i+=256) history.push_back(returned[i]);
        std::sort(history.begin(),history.end());
        for(unsigned i=0;i<64;++i) if(history[i]!=i) return 3;
    }
    CUDA_CHECK(cudaFree(values)); CUDA_CHECK(cudaFree(old_values));
    CUDA_CHECK(cudaStreamDestroy(stream));
    std::printf("{\"correct\":true,\"atomic_operations\":%u,\"buckets\":256}\n",count);
    return 0;
}
