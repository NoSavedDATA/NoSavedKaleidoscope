#pragma once

#include <cuda_runtime.h>
#include <cublasLt.h>
#include <cublas_v2.h>
#include <curand_kernel.h>
#include <cuda_fp16.h>
#include <cudnn.h>
#include <mma.h>




__global__ void lstm_single_step_kernel(float *fused_out, const float *x_out, const float *W, const float *ht, const float *b,
                      const int t, const int T, const int tile_size, const int tile_offset,
                      const int B, const int OC, const int fourX_OC, const int tanh_offset); 



__global__ void lstm_elementwise_ops_kernel(const float *fused_out,
                      float *ht, float *ct,
                      const int tile_size, const int tile_offset,
                      const int t, const int T,
                      const int B, const int OC, const int fourX_OC,
                      const int f_offset, const int o_offset, const int c_offset); 


__global__ void lstm_single_step_backward_dht_kernel(const float *d_ifoc,
                      float *d_ht, const float *w,
                      const int t, const int _t, const int T,
                      const int tile_size, const int tile_offset,
                      const int B, const int C, const int OC); 



__global__ void lstm_backward_dx_kernel(const float *d_ifoc,
                      float *dx, const float *w,
                      const int tile_size, const int tile_offset,
                      const int B, const int T, const int C, const int OC); 




__global__ void lstm_elementwise_ops_backward_kernel(const float *fused_out,
                      const float *ct,
                      float *d_ht, float *d_ct, float *d_ifoc, float *dB,
                      const float *w,
                      const int tile_size, const int tile_offset,
                      const int t, const int _t, const int T,
                      const int B, const int OC, const int fourX_OC,
                      const int f_offset, const int o_offset, const int c_offset);