#pragma once


#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <mma.h>



__global__ void repeat_interleave_kernel_last_dim(const float *tensor,
                           float *probs,
                           int B, int C); 


__global__ void idx_last_dim_kernel(float *tgt,
                           const float *tensor, const float *idx_tensor, 
                           int dims_prod, int last_dim_size); 

__global__ void idx_attr_semi_last_dim_kernel(float *tgt,
                           const float *tensor, const float *idx_tensor, 
                           int dims_prod, int last_dim_size); 


__global__ void idx_attr_simple_single_dim_kernel(float *tensor, const float *idx, const float *x, const int dims_prod);
