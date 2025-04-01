#pragma once


#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <mma.h>



__global__ void rl_discounted_return_kernel(float *G, const float *rewards, const float *terminated,
                                      const int T, const float gamma, const float dims_prod); 