#pragma once


#include <cuda_runtime.h>
#include <curand_kernel.h>

__global__ void dropout_mask_kernel(float *y, float *m, const float *x, float rate, float scale,
                               int dims_prod,
                               unsigned long long seed) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < dims_prod)
    {
      curandState state;
      curand_init(seed, i, 0, &state);

      float r = curand_uniform(&state);
      if(r<rate)
        m[i] = 0;
      else
        m[i] = scale;
      
      y[i] = m[i]*x[i];
    }
}


__global__ void dropout_backward_kernel(float *dx, float *m, const float *dy,
                               int dims_prod) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < dims_prod)
      dx[i] = m[i]*dy[i];
}