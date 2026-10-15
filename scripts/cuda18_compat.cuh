#pragma once
// CUDA12.8 declares this16-bit intrinsic, but Clang18 cannot lower it.
// PTX atom.cas.b16 supports generic global/shared addresses on SM70 and above:
// https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-atom
#if defined(__clang__) && __clang_major__ <= 18 && defined(__CUDA_ARCH__)
#include <cuda_runtime.h>
extern "C" __device__ __forceinline__ unsigned short __usAtomicCAS(
    unsigned short *address, unsigned short compare, unsigned short value) {
    unsigned short previous;
    asm volatile("atom.cas.b16 %0, [%1], %2, %3;"
                 : "=h"(previous) : "l"(address), "h"(compare), "h"(value) : "memory");
    return previous;
}
#endif
