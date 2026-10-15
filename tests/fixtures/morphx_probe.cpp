// Driver launch deliberately exercises the same _pk hook path as serving kernels.
#include <cuda.h>
#include <cstdio>
#include <cstring>
#include <vector>
#define CHECK(expr) do { CUresult r = (expr); if (r != CUDA_SUCCESS) { \
    const char *msg = nullptr; cuGetErrorString(r, &msg); \
    std::fprintf(stderr, "%s: %s\n", #expr, msg ? msg : "unknown"); return 1; } } while (0)
int main(int argc, char **argv) {
    if (argc != 2) return 2;
    CHECK(cuInit(0));
    CUdevice device; CHECK(cuDeviceGet(&device, 0));
    CUuuid uuid; CHECK(cuDeviceGetUuid(&uuid, device));
    const unsigned char expected[] = {0x4d,0x9b,0xef,0x9d,0x63,0x1c,0xbb,0x36,
                                     0x3f,0xf4,0x58,0x9d,0x6a,0x18,0x41,0x89};
    if (std::memcmp(uuid.bytes, expected, 16)) {
        std::fprintf(stderr, "refusing non-owned GPU UUID\n"); return 4;
    }
    CUcontext context; CHECK(cuCtxCreate(&context, 0, device));
    CUmodule module; CHECK(cuModuleLoad(&module, argv[1]));
    CUfunction function; CHECK(cuModuleGetFunction(&function, module, "morphx_probe"));
    constexpr unsigned count = 524288;
    CUdeviceptr output; CHECK(cuMemAlloc(&output, count * sizeof(unsigned)));
    CUstream stream; CHECK(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING));
    unsigned n = count; void *args[] = {&output, &n};
    CHECK(cuLaunchKernel(function, count/256, 1, 1, 256, 1, 1, 0, stream, args, nullptr));
    CHECK(cuStreamSynchronize(stream));
    std::vector<unsigned> host(count);
    CHECK(cuMemcpyDtoH(host.data(), output, count * sizeof(unsigned)));
    for (unsigned i=0; i<count; ++i) if (host[i] != i*3u+7u) {
        std::fprintf(stderr, "incorrect element %u: %u\n", i, host[i]); return 3;
    }
    std::printf("{\"correct\":true,\"elements\":%u}\n", count);
    CHECK(cuStreamDestroy(stream)); CHECK(cuMemFree(output));
    CHECK(cuModuleUnload(module)); CHECK(cuCtxDestroy(context));
    return 0;
}
