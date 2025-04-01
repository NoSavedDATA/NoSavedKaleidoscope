#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <mma.h>


__device__ __forceinline__ float atomicMul(float* address, float val) {
    int *addr_as_int = (int *)address;
    int old = *addr_as_int, assumed;
    do {
        assumed = old;
        old = atomicCAS(addr_as_int, assumed,
                        __float_as_int(val * __int_as_float(assumed)));
    } while (assumed != old);
    return __int_as_float(old);
}