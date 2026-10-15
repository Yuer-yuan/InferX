#include <cassert>
#include <cstring>
#include <iostream>
#include <mutex>
using CUresult = int;
using CUfunction = void*;
using CUstream = void*;
constexpr int CUDA_SUCCESS=0, CUDA_ERROR_INVALID_VALUE=1, CUDA_ERROR_NOT_SUPPORTED=801;
struct CUlaunchConfig {
    unsigned gridDimX=2, gridDimY=3, gridDimZ=1;
    unsigned blockDimX=32, blockDimY=2, blockDimZ=1, sharedMemBytes=128;
    CUstream hStream=nullptr;
    void* attrs=nullptr;
    unsigned numAttrs=0;
};
using cuLaunchKernel_func_t = CUresult (*)(CUfunction,unsigned,unsigned,unsigned,unsigned,unsigned,unsigned,unsigned,CUstream,void**,void**);
using cuLaunchKernelEx_func_t = CUresult (*)(const CUlaunchConfig*,CUfunction,void**,void**);
static void* shared_lib=nullptr;
static std::mutex launchMutex;
static int virtual_calls=0, native_calls=0, ptsz_calls=0, init_calls=0;
static const CUlaunchConfig* expected=nullptr;
static void** expected_args=nullptr;
static void** expected_extra=nullptr;
static CUfunction expected_function=nullptr;
static CUresult native_ex(const CUlaunchConfig* config, CUfunction f, void** args, void** extra) {
    assert(config==expected && f==expected_function && args==expected_args && extra==expected_extra);
    ++native_calls;
    return 123;
}
static cuLaunchKernelEx_func_t real_cuLaunchKernelEx = native_ex;
static void ld_init() { ++init_calls; shared_lib=reinterpret_cast<void*>(1); }
static CUresult function_name(const char** name, CUfunction) { *name="fixture"; return CUDA_SUCCESS; }
static auto real_cuFuncGetName=function_name;
static void traceLLMKernel(const char*,const char*,int,unsigned,unsigned,unsigned,unsigned,CUstream) {}
static CUresult cuLaunchKernel(CUfunction f,unsigned gx,unsigned gy,unsigned gz,unsigned bx,unsigned by,unsigned bz,unsigned shared,CUstream stream,void** args,void** extra) {
    assert(f==expected_function && args==expected_args && extra==expected_extra);
    assert(gx==expected->gridDimX && gy==expected->gridDimY && gz==expected->gridDimZ);
    assert(bx==expected->blockDimX && by==expected->blockDimY && bz==expected->blockDimZ && shared==expected->sharedMemBytes && stream==expected->hStream);
    ++virtual_calls; return 456;
}
static CUresult ptsz_ex(const CUlaunchConfig* config, CUfunction f, void** args, void** extra) {
    assert(config==expected && f==expected_function && args==expected_args && extra==expected_extra);
    ++ptsz_calls; return 789;
}
static CUresult ptsz_kernel(CUfunction f,unsigned gx,unsigned gy,unsigned gz,unsigned bx,unsigned by,unsigned bz,unsigned shared,CUstream stream,void** args,void** extra) {
    int previous=virtual_calls;
    assert(cuLaunchKernel(f,gx,gy,gz,bx,by,bz,shared,stream,args,extra)==456);
    virtual_calls=previous; ++ptsz_calls; return 789;
}
static void* dlsym(void*, const char* name) {
    if(std::strcmp(name,"cuLaunchKernelEx_ptsz")==0) return reinterpret_cast<void*>(ptsz_ex);
    if(std::strcmp(name,"cuLaunchKernel_ptsz")==0) return reinterpret_cast<void*>(ptsz_kernel);
    assert(false);return nullptr;
}
// PATCHED_DISPATCH
int main() {
    CUlaunchConfig config;
    void* args[1]={reinterpret_cast<void*>(22)};
    void* extra[1]={reinterpret_cast<void*>(33)};
    expected=&config;expected_args=args;expected_extra=extra;expected_function=reinterpret_cast<void*>(44);
    assert(cuLaunchKernelEx(nullptr,expected_function,args,extra)==CUDA_ERROR_INVALID_VALUE);
    config.attrs=reinterpret_cast<void*>(55);config.numAttrs=1;
    assert(cuLaunchKernelEx(&config,expected_function,args,extra)==123);
    config.numAttrs=3; // Unknown future attributes must also pass through intact.
    config.hStream=reinterpret_cast<void*>(66);
    assert(cuLaunchKernelEx(&config,expected_function,args,extra)==123);
    assert(native_calls==2 && virtual_calls==0);
    config.numAttrs=0;
    assert(cuLaunchKernelEx(&config,expected_function,args,extra)==456);
    assert(virtual_calls==1);
    config.hStream=nullptr;
    assert(cuLaunchKernelEx_ptsz(&config,expected_function,args,extra)==789);
    config.numAttrs=2;
    assert(cuLaunchKernelEx_ptsz(&config,expected_function,args,extra)==789);
    assert(cuLaunchKernel_ptsz(expected_function,2,3,1,32,2,1,128,nullptr,args,extra)==789);
    assert(ptsz_calls==3 && virtual_calls==1 && init_calls==1);
    std::cout<<"dispatch oracle passed\n";
}
