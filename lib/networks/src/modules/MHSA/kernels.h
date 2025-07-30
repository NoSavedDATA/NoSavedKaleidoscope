#pragma once

#include <cuda_runtime.h>
#include <cublasLt.h>
#include <cublas_v2.h>
#include <curand_kernel.h>
#include <cuda_fp16.h>
#include <cudnn.h>
#include <mma.h>


__global__ void flash_attn_kernel(float *o, const float *qkv, float *l,
                                  const int B, const int nh, const int T, const int d, const int C, const float d_scale, const int Bc, const int Br,
                                  const int Tc, const int Tr, const int tile_size, const float warps_per_block, const int threads_per_block);


__global__ void flash_attn_backward_kernel(float *d_qkv, const float *d_o, const float *qkv, const float *o, const float *l, float *D,
                                           const int B, const int nh, const int T, const int d, const int C, const float d_scale,
                                           const int Bc, const int Br, const int Tc, const int Tr,
                                           const int warps_per_block, const int threads_per_block);

