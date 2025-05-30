#pragma once

#include <cuda_runtime.h>
#include <cublasLt.h>
#include <cublas_v2.h>
#include <curand_kernel.h>
#include <cuda_fp16.h>
#include <cudnn.h>
#include <mma.h>

#include "../../../nsk_cuda/include.h"

using namespace nvcuda;






__global__ void embeddingln_backward_kernel(const float *x,
                      float *dw, const float *dy, const int tile_size,
                      int B, int C, int OC) {

  int row = blockIdx.y * blockDim.y + threadIdx.y; // B
  int col = blockIdx.x * blockDim.x + threadIdx.x; // OC

  int tx = threadIdx.x;
  int ty = threadIdx.y;
  

  if(row<B && col<OC)
  {    
    float *_dw = dw + ((int)x[row])*OC + col;
    //float _dy = dy[row*OC + col];

    atomicAdd(_dw, dy[row*OC + col]);
    
    //dw[row*C + col] = tmp;
  }
}