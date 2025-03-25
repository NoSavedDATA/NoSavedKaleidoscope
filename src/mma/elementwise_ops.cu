#pragma once


#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <mma.h>



__inline__ __device__ float cuda_clip(float val, float min_val, float max_val) {
    return fmaxf(min_val, fminf(val, max_val));
}



__inline__ __global__ void set_to_zero_kernel(float *y, int dims_prod) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < dims_prod)
        y[i] = 0;
}
__inline__ __global__ void set_to_one_kernel(float *y, int dims_prod) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < dims_prod)
        y[i] = 1;
}

__inline__ __global__ void set_to_minus_inf_kernel(float *y, int dims_prod) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < dims_prod)
        y[i] = -INFINITY;
}

__inline__ __global__ void copy_tensor_kernel(float *y, const float *x, int dims_prod) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < dims_prod)
        y[i] = x[i];
}
__inline__ __global__ void ema_tensor_kernel(float *y, const float *x, const float factor, int dims_prod) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < dims_prod)
        y[i] = factor*y[i] + (1-factor)*x[i];
}


__inline__ __global__ void vec_log(const float* x, float* y, int dims_prod) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < dims_prod) {
    y[idx] = logf(x[idx]);
  }
}
__inline__ __global__ void vec_log2(const float* x, float* y, int dims_prod) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < dims_prod) {
    y[idx] = log2f(x[idx]);
  }
}

__inline__ __global__ void tensor_div(float *w, float *x, float *y, int dims_prod) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < dims_prod) {
    y[idx] = x[idx] / w[idx];
  }
}

__inline__ __global__ void tensor_clip(float* x, float *y, float _min, float _max, int dims_prod) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < dims_prod) {
    if (x[idx]>_max)
      y[idx] = _max;
    else if (x[idx]<_min)
      y[idx] = _min;
    else
      y[idx] = x[idx];
  }
}
