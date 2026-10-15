// Alternate transport without gdrdrv: counters stay in GPU memory. Each host
// thread owns its copy stream/staging, so a slow agent poll cannot serialize
// the launcher's independent initialization. DMA latency differs from GDRCopy.
#include <cstdlib>
#include <cstring>
class GdrPool {
    struct Transfer {
        cudaStream_t stream{};
        void* host{};
        Transfer() {
            CHECK_CUDA_ERROR(cudaHostAlloc(&host,GDR_BUFFER_SIZE,cudaHostAllocDefault));
            CHECK_CUDA_ERROR(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
        }
        ~Transfer() {
            // Only synchronized copies use these resources. CUDA teardown may
            // already be underway when the main thread's TLS is destroyed.
            cudaStreamDestroy(stream);
            cudaFreeHost(host);
        }
        void copy(void* destination,const void* source,size_t count,int kind) {
            CHECK_CUDA_ERROR(cudaMemcpyAsync(destination,source,count,static_cast<cudaMemcpyKind>(kind),stream));
            CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));
        }
    };
    static Transfer& transfer() {thread_local Transfer local;return local;}
    CUdeviceptr d_pool{};
    uintptr_t h_pool{};
    boost::lockfree::queue<GdrEntry> buffer{GDR_POOL_SIZE / GDR_ENTRY_SIZE};
public:
    GdrPool() {
        CHECK_CUDA_ERROR(cudaMalloc(reinterpret_cast<void**>(&d_pool),GDR_POOL_SIZE));
        // h is an identity/offset token retained for the upstream pool API;
        // actual transfers use thread-private pinned staging, never this shadow.
        h_pool=reinterpret_cast<uintptr_t>(std::malloc(GDR_POOL_SIZE));
        if(!h_pool) std::abort();
        auto& io=transfer();
        CHECK_CUDA_ERROR(cudaMemsetAsync(reinterpret_cast<void*>(d_pool),0,GDR_POOL_SIZE,io.stream));
        CHECK_CUDA_ERROR(cudaStreamSynchronize(io.stream));
        for(size_t i=0;i<GDR_POOL_SIZE;i+=GDR_ENTRY_SIZE)
            if(!buffer.push({d_pool+i,h_pool+i})) std::abort();
    }
    GdrEntry gdr_malloc() {GdrEntry entry;if(!buffer.pop(entry))std::abort();return entry;}
    void gdr_free(GdrEntry entry) {if(!buffer.push(entry))std::abort();}
    void set(GdrEntry& entry,uint64_t value) {
        auto& io=transfer();std::memcpy(io.host,&value,GDR_ENTRY_SIZE);
        io.copy(reinterpret_cast<void*>(entry.d),io.host,GDR_ENTRY_SIZE,cudaMemcpyHostToDevice);
    }
    uint64_t get(GdrEntry& entry) {
        auto& io=transfer();io.copy(io.host,reinterpret_cast<void*>(entry.d),GDR_ENTRY_SIZE,cudaMemcpyDeviceToHost);
        uint64_t value;std::memcpy(&value,io.host,GDR_ENTRY_SIZE);return value;
    }
    int get_index(GdrEntry& entry) {return ((entry.h-h_pool)%GDR_BUFFER_SIZE)/sizeof(int);}
    int get_bufferId(GdrEntry& entry) {return (entry.h-h_pool)/GDR_BUFFER_SIZE;}
    void get_buffer(int id,void* destination) {
        auto& io=transfer();io.copy(io.host,reinterpret_cast<void*>(d_pool+id*GDR_BUFFER_SIZE),GDR_BUFFER_SIZE,cudaMemcpyDeviceToHost);
        std::memcpy(destination,io.host,GDR_BUFFER_SIZE);
    }
    void set_buffer(int id,void* source) {
        auto& io=transfer();std::memcpy(io.host,source,GDR_BUFFER_SIZE);
        io.copy(reinterpret_cast<void*>(d_pool+id*GDR_BUFFER_SIZE),io.host,GDR_BUFFER_SIZE,cudaMemcpyHostToDevice);
    }
    ~GdrPool() {CHECK_CUDA_ERROR(cudaFree(reinterpret_cast<void*>(d_pool)));std::free(reinterpret_cast<void*>(h_pool));}
};
