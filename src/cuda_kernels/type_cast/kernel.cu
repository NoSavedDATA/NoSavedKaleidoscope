
#include <cuda_fp16.h>

__global__ void float_to_half_kernel(const float *__restrict__ x, __half *__restrict__ y, const int dims_prod)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx>dims_prod)
    return;

  y[idx] = __float2half(x[idx]);
}




__global__ void half_to_float_kernel(const __half *__restrict__ x, float *__restrict__ y, const int dims_prod)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx>dims_prod)
    return;

  y[idx] = __half2float(x[idx]);
}