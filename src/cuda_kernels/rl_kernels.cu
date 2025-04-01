#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <mma.h>

#include "atomic_kernels.cu"
#include "handles.h"



__global__ void rl_discounted_return_kernel(float *G, const float *rewards, const float *terminated,
                                      const int T, const float gamma, const float dims_prod) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;

    float g=0;
    if (b < dims_prod) {
        for(int t=T-1; t>=0; t--)
        {
          g += rewards[b*T+t] * powf(gamma, t);
          g = g * (1-terminated[b*T+t]);
        }
        G[b] = g;
    }
}