// Independent integer oracle for preserving Cooperative launch attributes.
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cstdio>
#include <vector>

extern "C" __global__ void morphx_ordinary(unsigned* data, unsigned count) {
    unsigned i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<count)data[i]=3*i+7;
}
extern "C" __global__ void morphx_cooperative(unsigned* data, unsigned* result, unsigned count) {
    auto grid=cooperative_groups::this_grid();
    unsigned i=blockIdx.x*blockDim.x+threadIdx.x;
    if(!grid.is_valid()) { if(i==0)*result=0x7bad01u; return; }
    if(i<count)data[i]=5*i+11;
    grid.sync();
    if(i==0) {
        unsigned sum=0;
        for(unsigned j=0;j<count;j++)sum+=data[j];
        *result=sum;
    }
}
static bool check(cudaError_t code,const char* stage) {
    if(code==cudaSuccess)return true;
    std::fprintf(stderr,"stage=%s error=%s\n",stage,cudaGetErrorString(code));return false;
}
int main() {
    constexpr unsigned count=128;
    unsigned *data=nullptr,*result=nullptr;
    cudaStream_t stream;
    if(!check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking),"stream"))return 2;
    if(!check(cudaMalloc(&data,count*sizeof(unsigned)),"data"))return 2;
    if(!check(cudaMalloc(&result,sizeof(unsigned)),"result"))return 2;
    morphx_ordinary<<<2,64,0,stream>>>(data,count);
    std::vector<unsigned> host(count);
    if(!check(cudaMemcpyAsync(host.data(),data,count*sizeof(unsigned),cudaMemcpyDeviceToHost,stream),"ordinary-copy") || !check(cudaStreamSynchronize(stream),"ordinary-sync"))return 2;
    for(unsigned i=0;i<count;i++)if(host[i]!=3*i+7)return 3;
    cudaLaunchAttribute attr={};attr.id=cudaLaunchAttributeCooperative;attr.val.cooperative=1;
    cudaLaunchConfig_t config={};config.gridDim=dim3(2);config.blockDim=dim3(64);config.stream=stream;config.attrs=&attr;config.numAttrs=1;
    if(!check(cudaLaunchKernelEx(&config,morphx_cooperative,data,result,count),"cooperative-launch"))return 2;
    unsigned observed=0;
    if(!check(cudaMemcpyAsync(&observed,result,sizeof(unsigned),cudaMemcpyDeviceToHost,stream),"cooperative-copy") || !check(cudaStreamSynchronize(stream),"cooperative-sync"))return 2;
    const unsigned expected=5*count*(count-1)/2+11*count;
    std::printf("{\"checked\":%u,\"expected\":%u,\"observed\":%u,\"incorrect\":%u}\n",count,expected,observed,unsigned(observed!=expected));
    cudaFree(result);cudaFree(data);cudaStreamDestroy(stream);
    return observed==expected?0:4;
}
