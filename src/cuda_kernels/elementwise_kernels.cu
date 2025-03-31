#include <cuda_runtime.h>

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <mma.h>



__global__ void to_half(__half *y, const float *x, int dims_prod)
{
  int idx = blockIdx.x*blockDim.x + threadIdx.x;

  if(idx>=dims_prod)
    return;

  y[idx] = __float2half(x[idx]);
}