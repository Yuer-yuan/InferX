// Driver-only control helpers; progress profiling and live SM writes stay upstream.
#include <cstddef>
#include <cstdint>
namespace inferx_control {
struct Driver {
    CUresult (*fill32)(CUdeviceptr,unsigned int,size_t,CUstream){};
    CUresult (*create_event)(CUevent*,unsigned int){};
    CUresult (*record_event)(CUevent,CUstream){};
    CUresult (*query_event)(CUevent){};
    CUresult (*destroy_event)(CUevent){};
    bool valid() const {return fill32 && create_event && record_event && query_event && destroy_event;}
};
inline CUresult initialize_agents(const Driver& driver,CUdeviceptr address,unsigned int value,CUstream stream) {
    static_assert(sizeof(unsigned int)==4,"CUDA D32 fill requires exactly32 bits");
    if(!driver.fill32) return CUDA_ERROR_NOT_SUPPORTED;
    return driver.fill32(address,value,1,stream);
}
class Completion {
    CUevent event{};
    bool recorded=false;
    bool completed=false;
public:
    Completion()=default;
    Completion(const Completion&)=delete;
    Completion& operator=(const Completion&)=delete;
    CUresult record_after_launch(const Driver& driver,CUstream stream,CUresult launch_result) {
        if(launch_result!=CUDA_SUCCESS) return launch_result;
        if(!driver.valid()) return CUDA_ERROR_NOT_SUPPORTED;
        if(event) return CUDA_ERROR_INVALID_HANDLE;
        auto result=driver.create_event(&event,CU_EVENT_DISABLE_TIMING);
        if(result!=CUDA_SUCCESS) return result;
        result=driver.record_event(event,stream);
        if(result!=CUDA_SUCCESS) {
            driver.destroy_event(event);event=nullptr;return result;
        }
        recorded=true;return CUDA_SUCCESS;
    }
    CUresult poll(const Driver& driver,bool& done) {
        done=false;
        if(!event || !recorded) return CUDA_ERROR_INVALID_HANDLE;
        auto result=driver.query_event(event);
        if(result==CUDA_ERROR_NOT_READY) return CUDA_SUCCESS;
        if(result!=CUDA_SUCCESS) return result;
        completed=done=true;return CUDA_SUCCESS;
    }
    CUresult release(const Driver& driver) {
        if(!event || !recorded) return CUDA_ERROR_INVALID_HANDLE;
        if(!completed) return CUDA_ERROR_NOT_READY;
        auto result=driver.destroy_event(event);
        if(result==CUDA_SUCCESS) {event=nullptr;recorded=false;completed=false;}
        return result;
    }
};
}
