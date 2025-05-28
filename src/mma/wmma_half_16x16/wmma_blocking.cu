#include "../utils.h"
#include "../../nsk_cuda/include.h"
#include "wmma_blocking.h"


using namespace nvcuda;

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16





template __global__ void wmma_blocking<16,1,1,32>(const float *__restrict__ x, const float *__restrict__ w,
                        float *__restrict__ out, const int B, const int C, const int OC,
                        const int bx, const int by,
                        const int wx, const int wy,
                        const int bx_per_w,     const int by_per_w,
                        const int bx_per_wx,    const int by_per_wy);



// template __global__ void wmma_blocking<16,2,1,32>(const float *__restrict__ x, const float *__restrict__ w,
//                         float *__restrict__ out, const int B, const int C, const int OC,
//                         const int bx, const int by,
//                         const int wx, const int wy,
//                         const int bx_per_w,     const int by_per_w,
//                         const int bx_per_wx,    const int by_per_wy);


template __global__ void wmma_blocking<16,3,1,32>(const float *__restrict__ x, const float *__restrict__ w,
                        float *__restrict__ out, const int B, const int C, const int OC,
                        const int bx, const int by,
                        const int wx, const int wy,
                        const int bx_per_w,     const int by_per_w,
                        const int bx_per_wx,    const int by_per_wy);


template __global__ void wmma_blocking<16,4,1,32>(const float *__restrict__ x, const float *__restrict__ w,
                        float *__restrict__ out, const int B, const int C, const int OC,
                        const int bx, const int by,
                        const int wx, const int wy,
                        const int bx_per_w,     const int by_per_w,
                        const int bx_per_wx,    const int by_per_wy);


template __global__ void wmma_blocking<16,1,2,32>(const float *__restrict__ x, const float *__restrict__ w,
                        float *__restrict__ out, const int B, const int C, const int OC,
                        const int bx, const int by,
                        const int wx, const int wy,
                        const int bx_per_w,     const int by_per_w,
                        const int bx_per_wx,    const int by_per_wy);


template __global__ void wmma_blocking<16,2,2,32>(const float *__restrict__ x, const float *__restrict__ w,
                        float *__restrict__ out, const int B, const int C, const int OC,
                        const int bx, const int by,
                        const int wx, const int wy,
                        const int bx_per_w,     const int by_per_w,
                        const int bx_per_wx,    const int by_per_wy);


template __global__ void wmma_blocking<16,3,2,32>(const float *__restrict__ x, const float *__restrict__ w,
                        float *__restrict__ out, const int B, const int C, const int OC,
                        const int bx, const int by,
                        const int wx, const int wy,
                        const int bx_per_w,     const int by_per_w,
                        const int bx_per_wx,    const int by_per_wy);

template __global__ void wmma_blocking<16,4,2,32>(const float *__restrict__ x, const float *__restrict__ w,
                        float *__restrict__ out, const int B, const int C, const int OC,
                        const int bx, const int by,
                        const int wx, const int wy,
                        const int bx_per_w,     const int by_per_w,
                        const int bx_per_wx,    const int by_per_wy);


template __global__ void wmma_blocking<16,1,3,32>(const float *__restrict__ x, const float *__restrict__ w,
                        float *__restrict__ out, const int B, const int C, const int OC,
                        const int bx, const int by,
                        const int wx, const int wy,
                        const int bx_per_w,     const int by_per_w,
                        const int bx_per_wx,    const int by_per_wy);


template __global__ void wmma_blocking<16,2,3,32>(const float *__restrict__ x, const float *__restrict__ w,
                        float *__restrict__ out, const int B, const int C, const int OC,
                        const int bx, const int by,
                        const int wx, const int wy,
                        const int bx_per_w,     const int by_per_w,
                        const int bx_per_wx,    const int by_per_wy);


template __global__ void wmma_blocking<16,3,3,32>(const float *__restrict__ x, const float *__restrict__ w,
                        float *__restrict__ out, const int B, const int C, const int OC,
                        const int bx, const int by,
                        const int wx, const int wy,
                        const int bx_per_w,     const int by_per_w,
                        const int bx_per_wx,    const int by_per_wy);


template __global__ void wmma_blocking<16,4,3,32>(const float *__restrict__ x, const float *__restrict__ w,
                        float *__restrict__ out, const int B, const int C, const int OC,
                        const int bx, const int by,
                        const int wx, const int wy,
                        const int bx_per_w,     const int by_per_w,
                        const int bx_per_wx,    const int by_per_wy);


template __global__ void wmma_blocking<16,1,4,32>(const float *__restrict__ x, const float *__restrict__ w,
                        float *__restrict__ out, const int B, const int C, const int OC,
                        const int bx, const int by,
                        const int wx, const int wy,
                        const int bx_per_w,     const int by_per_w,
                        const int bx_per_wx,    const int by_per_wy);


template __global__ void wmma_blocking<16,2,4,32>(const float *__restrict__ x, const float *__restrict__ w,
                        float *__restrict__ out, const int B, const int C, const int OC,
                        const int bx, const int by,
                        const int wx, const int wy,
                        const int bx_per_w,     const int by_per_w,
                        const int bx_per_wx,    const int by_per_wy);


template __global__ void wmma_blocking<16,3,4,32>(const float *__restrict__ x, const float *__restrict__ w,
                        float *__restrict__ out, const int B, const int C, const int OC,
                        const int bx, const int by,
                        const int wx, const int wy,
                        const int bx_per_w,     const int by_per_w,
                        const int bx_per_wx,    const int by_per_wy);


template __global__ void wmma_blocking<16,4,4,32>(const float *__restrict__ x, const float *__restrict__ w,
                        float *__restrict__ out, const int B, const int C, const int OC,
                        const int bx, const int by,
                        const int wx, const int wy,
                        const int bx_per_w,     const int by_per_w,
                        const int bx_per_wx,    const int by_per_wy);