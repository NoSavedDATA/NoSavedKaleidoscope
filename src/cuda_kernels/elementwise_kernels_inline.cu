#include <cuda_runtime.h>

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <mma.h>


__device__ __forceinline__ float cuda_clip(float val, float min_val, float max_val) {
    return fmaxf(min_val, fminf(val, max_val));
}


__global__ __forceinline__ void set_to_zero_kernel(float *y, int dims_prod) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < dims_prod)
        y[i] = 0;
}
__global__ __forceinline__ void set_to_one_kernel(float *y, int dims_prod) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < dims_prod)
        y[i] = 1;
}

__global__ __forceinline__ void set_to_minus_inf_kernel(float *y, int dims_prod) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < dims_prod)
        y[i] = -INFINITY;
}

__global__ __forceinline__ void copy_tensor_kernel(float *y, const float *x, int dims_prod) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < dims_prod)
        y[i] = x[i];
}
__global__ __forceinline__ void ema_tensor_kernel(float *y, const float *x, const float factor, int dims_prod) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < dims_prod)
        y[i] = factor*y[i] + (1-factor)*x[i];
}


__global__ __forceinline__ void vec_log(const float* x, float* y, int dims_prod) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < dims_prod) {
    y[idx] = logf(x[idx]);
  }
}
__global__ __forceinline__ void vec_log2(const float* x, float* y, int dims_prod) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < dims_prod) {
    y[idx] = log2f(x[idx]);
  }
}

__global__ __forceinline__ void tensor_div(float *w, float *x, float *y, int dims_prod) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < dims_prod) {
    y[idx] = x[idx] / w[idx];
  }
}

__global__ __forceinline__ void tensor_clip(float* x, float *y, float _min, float _max, int dims_prod) {
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

__device__ inline float lerp(float start, float end, float weight) {
  return fma(weight, end, fma(-weight, start, start));
}