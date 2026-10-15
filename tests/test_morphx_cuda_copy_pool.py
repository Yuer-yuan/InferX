"""Exercise the real copy-backed pool against an asynchronous CUDA API oracle."""
from pathlib import Path
import subprocess


def test_control_pool_keeps_counters_on_device_and_copies_on_private_stream(tmp_path):
    header=Path(__file__).parents[1]/'scripts/morphx_cuda_copy_pool.hpp'
    assert header.is_file(),'copy-backed CUDA pool not implemented'
    prefix=r'''
#include <cassert>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <queue>
#include <thread>
#include <vector>
#include <mutex>
using CUdeviceptr=uintptr_t;using cudaStream_t=void*;using cudaMemcpyKind=int;
const int cudaHostAllocDefault=0,cudaStreamNonBlocking=1,cudaMemcpyHostToDevice=1,cudaMemcpyDeviceToHost=2;
int copies=0,syncs=0;void* private_stream=(void*)0x5678;
int cudaMalloc(void** p,size_t n){*p=std::malloc(n);return 0;}
int cudaHostAlloc(void** p,size_t n,unsigned){*p=std::malloc(n);return 0;}
int cudaStreamCreateWithFlags(cudaStream_t* s,unsigned flags){assert(flags==1);*s=private_stream;return 0;}
int cudaMemsetAsync(void* p,int v,size_t n,cudaStream_t s){assert(s==private_stream);std::memset(p,v,n);return 0;}
int cudaMemcpyAsync(void* d,const void* s,size_t n,int kind,cudaStream_t stream){assert(stream==private_stream && (kind==1||kind==2));std::memcpy(d,s,n);++copies;return 0;}
int cudaStreamSynchronize(cudaStream_t s){assert(s==private_stream);++syncs;return 0;}
int cudaStreamDestroy(cudaStream_t s){assert(s==private_stream);return 0;}
int cudaFree(void* p){std::free(p);return 0;}int cudaFreeHost(void* p){std::free(p);return 0;}
#define CHECK_CUDA_ERROR(call) assert((call)==0)
#define GDR_POOL_SIZE 4096
#define GDR_ENTRY_SIZE 8
#define GDR_BUFFER_SIZE 512
struct GdrEntry{CUdeviceptr d;uintptr_t h;GdrEntry half(){return {d+4,h+4};}};
namespace boost{namespace lockfree{template<class T> class queue{std::queue<T> q;public:explicit queue(size_t){}bool push(T x){q.push(x);return true;}bool pop(T& x){if(q.empty())return false;x=q.front();q.pop();return true;}};}}
'''
    main=r'''
int main(){
 GdrPool pool;auto e=pool.gdr_malloc();
 assert(e.d!=e.h);pool.set(e,17);assert(*(uint64_t*)e.d==17);
 *(uint64_t*)e.d=42;assert(*(uint64_t*)e.h==17);assert(pool.get(e)==42);
 std::vector<GdrEntry> entries;for(int i=0;i<4;++i)entries.push_back(pool.gdr_malloc());
 std::vector<std::thread> threads;
 for(int i=0;i<4;++i)threads.emplace_back([&,i]{for(int k=0;k<20;++k){pool.set(entries[i],i*100+k);assert(pool.get(entries[i])==(uint64_t)(i*100+k));}});
 for(auto& t:threads)t.join();
 uint64_t data[64]={};data[2]=99;pool.set_buffer(1,data);data[2]=0;pool.get_buffer(1,data);assert(data[2]==99);
 assert(copies==164 && syncs==copies+1);
 pool.gdr_free(e);for(auto entry:entries)pool.gdr_free(entry);
}
'''
    source=tmp_path/'pool.cpp';source.write_text(prefix+header.read_text()+main)
    subprocess.run(['g++','-std=c++17','-pthread',str(source),'-o',str(tmp_path/'probe')],check=True,capture_output=True,text=True)
    result=subprocess.run([str(tmp_path/'probe')],capture_output=True,text=True,timeout=10)
    assert result.returncode==0,result.stderr
