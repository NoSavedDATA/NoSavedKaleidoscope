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


__global__ void broadcast_lastdim_add(float *y, const float *x,
    const float *w, int dims_prod, int C); 

__global__ void sum_over_last_dim_kernel(const float *tensor,
                           float *summed,
                           int dims_prod, int summed_dim_size); 

__global__ void mean_over_semilast_dim_kernel(const float *x, float *y, const int dims_prod, const int T, const int C, const int warps_per_block);



__global__ void mean_over_semilast_dim_backward_kernel(float *dx, const float *dy, const int dims_prod, const int T, const int C);


__global__ void sum_single_dim_kernel(const float *tensor,
                           float *summed,
                           int dims_prod); 


__global__ void sum_over_semilast_dim_kernel(const float *tensor,
                           float *summed,
                           int dims_prod, int last_dim_size, int summed_dim_size);


__global__ void prod_single_dim_kernel(const float *tensor,
                           float *summed,
                           int dims_prod); 

__global__ void prod_over_last_dim_kernel(const float *tensor,
                           float *summed,
                           int dims_prod, int summed_dim_size); 

__global__ void prod_over_semilast_dim_kernel(const float *tensor,
                           float *summed,
                           int dims_prod, int last_dim_size, int summed_dim_size); 

__global__ void gather_last_dim_kernel(float* y, const float* tensor, const float *tensor_idx,
                                      const int leading_dim, float dims_prod); 



__global__ void gather_last_dim_backward_kernel(float* dx, const float* dy, const float *tensor_idx,
                                      const int leading_dim, float dims_prod); 