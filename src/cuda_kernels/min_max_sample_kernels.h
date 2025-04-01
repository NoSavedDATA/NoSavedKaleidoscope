#pragma once


#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <mma.h>


// Parallelizes over B, C
__global__ void onehot_kernel(const float *tensor,
                           float *probs,
                           int B, int C); 

                        
__global__ void warped_to_probs_single_dim(float *y, const float *x, int C); 


__global__ void sample_val_from_probs(float *tensor, float *sampled_value, int n, unsigned long long seed); 

__global__ void sample_from_probs(float *tensor, float *sampled_value, int n, unsigned long long seed);


__global__ void warped_to_probs_single_dim_pow(float *y, const float *x, float alpha, int C); 


__global__ void is_w_kernel(float *is_w_ptr, const float *probs, const float *idx, float beta, float max_idx);



__global__ void max_over_last_dim_kernel(const float *tensor,
                           float *maxed,
                           int dims_prod, int maxed_dim_size); 

__global__ void max_over_semilast_dim_kernel(const float *tensor,
                           float *maxed,
                           int dims_prod, int last_dim_size, int maxed_dim_size); 

__global__ void argmax_over_last_dim_kernel(const float *tensor,
                           float *maxed, float *argmaxed,
                           int dims_prod, int maxed_dim_size); 



__global__ void topk_kernel(const float *tensor, float *topk,
                           float *maxed, float *argmaxed,
                           int dims_prod, int maxed_dim_size,
                           int j, int k);

                           
__global__ void topk_erase_argmax_aux_kernel(float *tensor,
                           float *argmaxed, int dims_prod, int maxed_dim_size); 