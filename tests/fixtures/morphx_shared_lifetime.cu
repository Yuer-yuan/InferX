// A native CTA retains shared memory until every thread has finished.
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>
#include <vector>
#define CUDA_CHECK(call) do { auto e=(call); if(e!=cudaSuccess) { \
    std::fprintf(stderr,"%s: %s\n",#call,cudaGetErrorString(e));return 1; } } while(0)
__global__ void late_shared_read(unsigned* output) {
    extern __shared__ volatile unsigned scratch[];
    if(threadIdx.x==0) scratch[0]=blockIdx.x+1234;
    __syncthreads();
    unsigned index=blockIdx.x*blockDim.x+threadIdx.x;
    if(threadIdx.x<32) {output[index]=scratch[0];return;}
    asm volatile("nanosleep.u32 100000;" ::: "memory");
    output[index]=scratch[0];
}
int main() {
    CUdevice dev;CUuuid uuid;
    if(cuInit(0)!=CUDA_SUCCESS || cuDeviceGet(&dev,0)!=CUDA_SUCCESS ||
       cuDeviceGetUuid(&uuid,dev)!=CUDA_SUCCESS) return 1;
    const unsigned char expected[]={0x4d,0x9b,0xef,0x9d,0x63,0x1c,0xbb,0x36,
                                   0x3f,0xf4,0x58,0x9d,0x6a,0x18,0x41,0x89};
    if(std::memcmp(uuid.bytes,expected,16)) return 4;
    constexpr unsigned blocks=256,threads=128,count=blocks*threads;
    unsigned* output;cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
    CUDA_CHECK(cudaMalloc(&output,count*sizeof(unsigned)));
    late_shared_read<<<blocks,threads,4,stream>>>(output);
    CUDA_CHECK(cudaGetLastError());CUDA_CHECK(cudaStreamSynchronize(stream));
    std::vector<unsigned> values(count);
    CUDA_CHECK(cudaMemcpy(values.data(),output,count*sizeof(unsigned),cudaMemcpyDeviceToHost));
    unsigned wrong=0;
    for(unsigned i=0;i<count;++i) if(values[i]!=i/threads+1234) ++wrong;
    CUDA_CHECK(cudaFree(output));CUDA_CHECK(cudaStreamDestroy(stream));
    std::printf("{\"checked\":%u,\"incorrect\":%u}\n",count,wrong);
    return wrong ? 3 : 0;
}
