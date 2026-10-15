// Alternate control transport for hosts without gdrdrv. GPU counters stay in
// device memory; host access uses a private nonblocking copy stream. This is
// explicitly slower control access than GDRCopy and must be measured separately.
#include <mutex>
class GdrPool {
    CUdeviceptr d_pool{};
    uintptr_t h_pool{};
    cudaStream_t copy_stream{};
    std::mutex copy_mutex;
    boost::lockfree::queue<GdrEntry> buffer{GDR_POOL_SIZE / GDR_ENTRY_SIZE};
    void copy(void* destination, const void* source, size_t count, int kind) {
        CHECK_CUDA_ERROR(cudaMemcpyAsync(destination, source, count,
            static_cast<cudaMemcpyKind>(kind), copy_stream));
        CHECK_CUDA_ERROR(cudaStreamSynchronize(copy_stream));
    }
public:
    GdrPool() {
        CHECK_CUDA_ERROR(cudaMalloc(reinterpret_cast<void**>(&d_pool), GDR_POOL_SIZE));
        void* host=nullptr;
        CHECK_CUDA_ERROR(cudaHostAlloc(&host, GDR_POOL_SIZE, cudaHostAllocDefault));
        h_pool=reinterpret_cast<uintptr_t>(host);
        CHECK_CUDA_ERROR(cudaStreamCreateWithFlags(&copy_stream, cudaStreamNonBlocking));
        CHECK_CUDA_ERROR(cudaMemsetAsync(reinterpret_cast<void*>(d_pool), 0, GDR_POOL_SIZE, copy_stream));
        CHECK_CUDA_ERROR(cudaStreamSynchronize(copy_stream));
        for(size_t i=0;i<GDR_POOL_SIZE;i+=GDR_ENTRY_SIZE) {
            if(!buffer.push({d_pool+i,h_pool+i})) std::abort();
        }
    }
    GdrEntry gdr_malloc() {
        GdrEntry entry;
        if(!buffer.pop(entry)) std::abort();
        return entry;
    }
    void gdr_free(GdrEntry entry) { if(!buffer.push(entry)) std::abort(); }
    void set(GdrEntry& entry, uint64_t value) {
        std::lock_guard<std::mutex> guard(copy_mutex);
        std::memcpy(reinterpret_cast<void*>(entry.h), &value, GDR_ENTRY_SIZE);
        copy(reinterpret_cast<void*>(entry.d), reinterpret_cast<void*>(entry.h), GDR_ENTRY_SIZE, cudaMemcpyHostToDevice);
    }
    uint64_t get(GdrEntry& entry) {
        std::lock_guard<std::mutex> guard(copy_mutex);
        copy(reinterpret_cast<void*>(entry.h), reinterpret_cast<void*>(entry.d), GDR_ENTRY_SIZE, cudaMemcpyDeviceToHost);
        uint64_t value;
        std::memcpy(&value,reinterpret_cast<void*>(entry.h),GDR_ENTRY_SIZE);
        return value;
    }
    int get_index(GdrEntry& entry) {return ((entry.h-h_pool)%GDR_BUFFER_SIZE)/sizeof(int);}
    int get_bufferId(GdrEntry& entry) {return (entry.h-h_pool)/GDR_BUFFER_SIZE;}
    void get_buffer(int id,void* destination) {
        std::lock_guard<std::mutex> guard(copy_mutex);
        void* host=reinterpret_cast<void*>(h_pool+id*GDR_BUFFER_SIZE);
        copy(host,reinterpret_cast<void*>(d_pool+id*GDR_BUFFER_SIZE),GDR_BUFFER_SIZE,cudaMemcpyDeviceToHost);
        std::memcpy(destination,host,GDR_BUFFER_SIZE);
    }
    void set_buffer(int id,void* source) {
        std::lock_guard<std::mutex> guard(copy_mutex);
        void* host=reinterpret_cast<void*>(h_pool+id*GDR_BUFFER_SIZE);
        std::memcpy(host,source,GDR_BUFFER_SIZE);
        copy(reinterpret_cast<void*>(d_pool+id*GDR_BUFFER_SIZE),host,GDR_BUFFER_SIZE,cudaMemcpyHostToDevice);
    }
    ~GdrPool() {
        CHECK_CUDA_ERROR(cudaStreamDestroy(copy_stream));
        CHECK_CUDA_ERROR(cudaFree(reinterpret_cast<void*>(d_pool)));
        CHECK_CUDA_ERROR(cudaFreeHost(reinterpret_cast<void*>(h_pool)));
    }
};
