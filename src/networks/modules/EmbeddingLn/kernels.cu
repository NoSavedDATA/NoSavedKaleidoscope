#pragma once

#include <cuda_runtime.h>
#include <cublasLt.h>
#include <cublas_v2.h>
#include <curand_kernel.h>
#include <cuda_fp16.h>
#include <cudnn.h>
#include <mma.h>




__global__ void embeddingln_forward_kernel(const float *x, const float *w,
                      float *out, const int tile_size, const int B, const int batches_per_block, const int C, const int OC) {
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int x_block = blockIdx.x;
  int y_block = blockIdx.y;

  
  int col = x_block*tile_size + tx; // OC


  // w e [V, OC]
  for (int i=0; i<batches_per_block; ++i)
  {
    int row = (y_block*batches_per_block+i)*tile_size + ty; // B

    if(row<B && col<OC)
      out[row*OC + col] = w[((int)x[row])*OC + col];
  }
}




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