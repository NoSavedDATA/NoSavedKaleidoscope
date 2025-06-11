#pragma once

#include "../../nsk_cuda/include.h"

#include "wmma_blocking_i8.h"

using namespace nvcuda;

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16





// Each thread-block handles bx rows and by cols
// Each bx work is splitted accross wx warp loads

 

template __global__ void wmma_blocking_i8_mma<16, 1, 1, 32>(const int8_t *__restrict__ x, const int8_t *__restrict__ w,
                        float *__restrict__ out_tensor, const float *scale_M, const float *scale_N,
                        const int M, const int N, const int K,
                        const int num_warps,
                        const int bx, const int by,
                        const int wx, const int wy,
                        const int bx_per_w,     const int by_per_w,
                        const int bx_per_wx);
template __global__ void wmma_blocking_i8_mma<16, 1, 2, 32>(const int8_t *__restrict__ x, const int8_t *__restrict__ w,
                        float *__restrict__ out_tensor, const float *scale_M, const float *scale_N,
                        const int M, const int N, const int K,
                        const int num_warps,
                        const int bx, const int by,
                        const int wx, const int wy,
                        const int bx_per_w,     const int by_per_w,
                        const int bx_per_wx);
template __global__ void wmma_blocking_i8_mma<16, 1, 3, 32>(const int8_t *__restrict__ x, const int8_t *__restrict__ w,
                        float *__restrict__ out_tensor, const float *scale_M, const float *scale_N,
                        const int M, const int N, const int K,
                        const int num_warps,
                        const int bx, const int by,
                        const int wx, const int wy,
                        const int bx_per_w,     const int by_per_w,
                        const int bx_per_wx);
template __global__ void wmma_blocking_i8_mma<16, 1, 4, 32>(const int8_t *__restrict__ x, const int8_t *__restrict__ w,
                        float *__restrict__ out_tensor, const float *scale_M, const float *scale_N,
                        const int M, const int N, const int K,
                        const int num_warps,
                        const int bx, const int by,
                        const int wx, const int wy,
                        const int bx_per_w,     const int by_per_w,
                        const int bx_per_wx);
template __global__ void wmma_blocking_i8_mma<16, 2, 1, 32>(const int8_t *__restrict__ x, const int8_t *__restrict__ w,
                        float *__restrict__ out_tensor, const float *scale_M, const float *scale_N,
                        const int M, const int N, const int K,
                        const int num_warps,
                        const int bx, const int by,
                        const int wx, const int wy,
                        const int bx_per_w,     const int by_per_w,
                        const int bx_per_wx);
template __global__ void wmma_blocking_i8_mma<16, 2, 2, 32>(const int8_t *__restrict__ x, const int8_t *__restrict__ w,
                        float *__restrict__ out_tensor, const float *scale_M, const float *scale_N,
                        const int M, const int N, const int K,
                        const int num_warps,
                        const int bx, const int by,
                        const int wx, const int wy,
                        const int bx_per_w,     const int by_per_w,
                        const int bx_per_wx);
template __global__ void wmma_blocking_i8_mma<16, 2, 3, 32>(const int8_t *__restrict__ x, const int8_t *__restrict__ w,
                        float *__restrict__ out_tensor, const float *scale_M, const float *scale_N,
                        const int M, const int N, const int K,
                        const int num_warps,
                        const int bx, const int by,
                        const int wx, const int wy,
                        const int bx_per_w,     const int by_per_w,
                        const int bx_per_wx);
template __global__ void wmma_blocking_i8_mma<16, 2, 4, 32>(const int8_t *__restrict__ x, const int8_t *__restrict__ w,
                        float *__restrict__ out_tensor, const float *scale_M, const float *scale_N,
                        const int M, const int N, const int K,
                        const int num_warps,
                        const int bx, const int by,
                        const int wx, const int wy,
                        const int bx_per_w,     const int by_per_w,
                        const int bx_per_wx);
template __global__ void wmma_blocking_i8_mma<16, 3, 1, 32>(const int8_t *__restrict__ x, const int8_t *__restrict__ w,
                        float *__restrict__ out_tensor, const float *scale_M, const float *scale_N,
                        const int M, const int N, const int K,
                        const int num_warps,
                        const int bx, const int by,
                        const int wx, const int wy,
                        const int bx_per_w,     const int by_per_w,
                        const int bx_per_wx);
template __global__ void wmma_blocking_i8_mma<16, 3, 2, 32>(const int8_t *__restrict__ x, const int8_t *__restrict__ w,
                        float *__restrict__ out_tensor, const float *scale_M, const float *scale_N,
                        const int M, const int N, const int K,
                        const int num_warps,
                        const int bx, const int by,
                        const int wx, const int wy,
                        const int bx_per_w,     const int by_per_w,
                        const int bx_per_wx);
template __global__ void wmma_blocking_i8_mma<16, 3, 3, 32>(const int8_t *__restrict__ x, const int8_t *__restrict__ w,
                        float *__restrict__ out_tensor, const float *scale_M, const float *scale_N,
                        const int M, const int N, const int K,
                        const int num_warps,
                        const int bx, const int by,
                        const int wx, const int wy,
                        const int bx_per_w,     const int by_per_w,
                        const int bx_per_wx);
template __global__ void wmma_blocking_i8_mma<16, 3, 4, 32>(const int8_t *__restrict__ x, const int8_t *__restrict__ w,
                        float *__restrict__ out_tensor, const float *scale_M, const float *scale_N,
                        const int M, const int N, const int K,
                        const int num_warps,
                        const int bx, const int by,
                        const int wx, const int wy,
                        const int bx_per_w,     const int by_per_w,
                        const int bx_per_wx);
template __global__ void wmma_blocking_i8_mma<16, 4, 1, 32>(const int8_t *__restrict__ x, const int8_t *__restrict__ w,
                        float *__restrict__ out_tensor, const float *scale_M, const float *scale_N,
                        const int M, const int N, const int K,
                        const int num_warps,
                        const int bx, const int by,
                        const int wx, const int wy,
                        const int bx_per_w,     const int by_per_w,
                        const int bx_per_wx);
template __global__ void wmma_blocking_i8_mma<16, 4, 2, 32>(const int8_t *__restrict__ x, const int8_t *__restrict__ w,
                        float *__restrict__ out_tensor, const float *scale_M, const float *scale_N,
                        const int M, const int N, const int K,
                        const int num_warps,
                        const int bx, const int by,
                        const int wx, const int wy,
                        const int bx_per_w,     const int by_per_w,
                        const int bx_per_wx);
template __global__ void wmma_blocking_i8_mma<16, 4, 3, 32>(const int8_t *__restrict__ x, const int8_t *__restrict__ w,
                        float *__restrict__ out_tensor, const float *scale_M, const float *scale_N,
                        const int M, const int N, const int K,
                        const int num_warps,
                        const int bx, const int by,
                        const int wx, const int wy,
                        const int bx_per_w,     const int by_per_w,
                        const int bx_per_wx);
template __global__ void wmma_blocking_i8_mma<16, 4, 4, 32>(const int8_t *__restrict__ x, const int8_t *__restrict__ w,
                        float *__restrict__ out_tensor, const float *scale_M, const float *scale_N,
                        const int M, const int N, const int K,
                        const int num_warps,
                        const int bx, const int by,
                        const int wx, const int wy,
                        const int bx_per_w,     const int by_per_w,
                        const int bx_per_wx);


                        
template __global__ void wmma_blocking_i8<16,1,1,32>(const int8_t *__restrict__ x, const int8_t *__restrict__ w,
                        float *__restrict__ out, const float *scale_M, const float *scale_N,
                        const int M, const int N, const int K,
                        const int num_warps,
                        const int bx, const int by,
                        const int wx, const int wy,
                        const int bx_per_w,     const int by_per_w,
                        const int bx_per_wx);

template __global__ void wmma_blocking_i8<16,1,2,32>(const int8_t *__restrict__ x, const int8_t *__restrict__ w,
                        float *__restrict__ out, const float *scale_M, const float *scale_N,
                        const int M, const int N, const int K,
                        const int num_warps,
                        const int bx, const int by,
                        const int wx, const int wy,
                        const int bx_per_w,     const int by_per_w,
                        const int bx_per_wx);

template __global__ void wmma_blocking_i8<16,1,3,32>(const int8_t *__restrict__ x, const int8_t *__restrict__ w,
                        float *__restrict__ out, const float *scale_M, const float *scale_N,
                        const int M, const int N, const int K,
                        const int num_warps,
                        const int bx, const int by,
                        const int wx, const int wy,
                        const int bx_per_w,     const int by_per_w,
                        const int bx_per_wx);

template __global__ void wmma_blocking_i8<16,1,4,32>(const int8_t *__restrict__ x, const int8_t *__restrict__ w,
                        float *__restrict__ out, const float *scale_M, const float *scale_N,
                        const int M, const int N, const int K,
                        const int num_warps,
                        const int bx, const int by,
                        const int wx, const int wy,
                        const int bx_per_w,     const int by_per_w,
                        const int bx_per_wx);

template __global__ void wmma_blocking_i8<16,2,1,32>(const int8_t *__restrict__ x, const int8_t *__restrict__ w,
                        float *__restrict__ out, const float *scale_M, const float *scale_N,
                        const int M, const int N, const int K,
                        const int num_warps,
                        const int bx, const int by,
                        const int wx, const int wy,
                        const int bx_per_w,     const int by_per_w,
                        const int bx_per_wx);

template __global__ void wmma_blocking_i8<16,2,2,32>(const int8_t *__restrict__ x, const int8_t *__restrict__ w,
                        float *__restrict__ out, const float *scale_M, const float *scale_N,
                        const int M, const int N, const int K,
                        const int num_warps,
                        const int bx, const int by,
                        const int wx, const int wy,
                        const int bx_per_w,     const int by_per_w,
                        const int bx_per_wx);

template __global__ void wmma_blocking_i8<16,2,3,32>(const int8_t *__restrict__ x, const int8_t *__restrict__ w,
                        float *__restrict__ out, const float *scale_M, const float *scale_N,
                        const int M, const int N, const int K,
                        const int num_warps,
                        const int bx, const int by,
                        const int wx, const int wy,
                        const int bx_per_w,     const int by_per_w,
                        const int bx_per_wx);

template __global__ void wmma_blocking_i8<16,2,4,32>(const int8_t *__restrict__ x, const int8_t *__restrict__ w,
                        float *__restrict__ out, const float *scale_M, const float *scale_N,
                        const int M, const int N, const int K,
                        const int num_warps,
                        const int bx, const int by,
                        const int wx, const int wy,
                        const int bx_per_w,     const int by_per_w,
                        const int bx_per_wx);

template __global__ void wmma_blocking_i8<16,3,1,32>(const int8_t *__restrict__ x, const int8_t *__restrict__ w,
                        float *__restrict__ out, const float *scale_M, const float *scale_N,
                        const int M, const int N, const int K,
                        const int num_warps,
                        const int bx, const int by,
                        const int wx, const int wy,
                        const int bx_per_w,     const int by_per_w,
                        const int bx_per_wx);

template __global__ void wmma_blocking_i8<16,3,2,32>(const int8_t *__restrict__ x, const int8_t *__restrict__ w,
                        float *__restrict__ out, const float *scale_M, const float *scale_N,
                        const int M, const int N, const int K,
                        const int num_warps,
                        const int bx, const int by,
                        const int wx, const int wy,
                        const int bx_per_w,     const int by_per_w,
                        const int bx_per_wx);

template __global__ void wmma_blocking_i8<16,3,3,32>(const int8_t *__restrict__ x, const int8_t *__restrict__ w,
                        float *__restrict__ out, const float *scale_M, const float *scale_N,
                        const int M, const int N, const int K,
                        const int num_warps,
                        const int bx, const int by,
                        const int wx, const int wy,
                        const int bx_per_w,     const int by_per_w,
                        const int bx_per_wx);

template __global__ void wmma_blocking_i8<16,3,4,32>(const int8_t *__restrict__ x, const int8_t *__restrict__ w,
                        float *__restrict__ out, const float *scale_M, const float *scale_N,
                        const int M, const int N, const int K,
                        const int num_warps,
                        const int bx, const int by,
                        const int wx, const int wy,
                        const int bx_per_w,     const int by_per_w,
                        const int bx_per_wx);

template __global__ void wmma_blocking_i8<16,4,1,32>(const int8_t *__restrict__ x, const int8_t *__restrict__ w,
                        float *__restrict__ out, const float *scale_M, const float *scale_N,
                        const int M, const int N, const int K,
                        const int num_warps,
                        const int bx, const int by,
                        const int wx, const int wy,
                        const int bx_per_w,     const int by_per_w,
                        const int bx_per_wx);

template __global__ void wmma_blocking_i8<16,4,2,32>(const int8_t *__restrict__ x, const int8_t *__restrict__ w,
                        float *__restrict__ out, const float *scale_M, const float *scale_N,
                        const int M, const int N, const int K,
                        const int num_warps,
                        const int bx, const int by,
                        const int wx, const int wy,
                        const int bx_per_w,     const int by_per_w,
                        const int bx_per_wx);

template __global__ void wmma_blocking_i8<16,4,3,32>(const int8_t *__restrict__ x, const int8_t *__restrict__ w,
                        float *__restrict__ out, const float *scale_M, const float *scale_N,
                        const int M, const int N, const int K,
                        const int num_warps,
                        const int bx, const int by,
                        const int wx, const int wy,
                        const int bx_per_w,     const int by_per_w,
                        const int bx_per_wx);

template __global__ void wmma_blocking_i8<16,4,4,32>(const int8_t *__restrict__ x, const int8_t *__restrict__ w,
                        float *__restrict__ out, const float *scale_M, const float *scale_N,
                        const int M, const int N, const int K,
                        const int num_warps,
                        const int bx, const int by,
                        const int wx, const int wy,
                        const int bx_per_w,     const int by_per_w,
                        const int bx_per_wx);
