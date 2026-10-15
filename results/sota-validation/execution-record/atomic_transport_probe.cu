#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <initializer_list>
#define CK(x) do{auto e=(x);if(e!=cudaSuccess){std::fprintf(stderr,"%s\n",cudaGetErrorString(e));return 2;}}while(0)
__global__ void counter(unsigned* value){if(threadIdx.x==0)atomicAdd(value,1u);}
int main(){
 unsigned *device=nullptr,*host=nullptr,*mapped=nullptr;
 CK(cudaMalloc(&device,8));CK(cudaHostAlloc(&host,8,cudaHostAllocMapped));CK(cudaHostGetDevicePointer(&mapped,host,0));
 cudaEvent_t begin,end;CK(cudaEventCreate(&begin));CK(cudaEventCreate(&end));
 for(int n:{128,4096,32768})for(int mode=0;mode<2;++mode){
  unsigned* pointer=mode?mapped:device;float total=0;
  for(int repeat=0;repeat<8;++repeat){
   CK(cudaMemset(pointer,0,4));CK(cudaEventRecord(begin));counter<<<n,32>>>(pointer);CK(cudaEventRecord(end));CK(cudaEventSynchronize(end));
   float ms;CK(cudaEventElapsedTime(&ms,begin,end));total+=ms;
   unsigned result;CK(cudaMemcpy(&result,pointer,4,cudaMemcpyDeviceToHost));if(result!=(unsigned)n){std::fprintf(stderr,"counter mismatch %u != %d\n",result,n);return 3;}
  }
  std::printf("{\"grid\":%d,\"transport\":\"%s\",\"mean_gpu_ms\":%.6f,\"repeats\":8,\"counter_exact\":true}\n",n,mode?"mapped-host":"device",total/8);
 }
 CK(cudaEventDestroy(begin));CK(cudaEventDestroy(end));CK(cudaFree(device));CK(cudaFreeHost(host));return 0;
}
