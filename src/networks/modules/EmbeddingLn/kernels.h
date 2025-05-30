#pragma once

#include <cuda_runtime.h>
#include <cublasLt.h>
#include <cublas_v2.h>
#include <curand_kernel.h>
#include <cuda_fp16.h>
#include <cudnn.h>
#include <mma.h>


#include "../../../nsk_cuda/include.h"



template<int wx_per_wmma_m, int wy_per_wmma_n>
__global__ void embeddingln_forward_kernel(const float *idxs, const float *embedding_book, const float *w,
                      float *out, const int B, const int OC, const int C) {
  // int tx = threadIdx.x;
  // int ty = threadIdx.y;
  // int x_block = blockIdx.x;
  // int y_block = blockIdx.y;

  
  // int col = x_block*tile_size + tx; // OC


  // // w e [V, OC]
  // for (int i=0; i<batches_per_block; ++i)
  // {
  //   int row = (y_block*batches_per_block+i)*tile_size + ty; // B

  //   if(row<B && col<OC)
  //     out[row*OC + col] = embedding_book[((int)idxs[row])*OC + col];
  // }

  fp16_wmma_frags<wx_per_wmma_m, wy_per_wmma_n> frag_loader;
  extern __shared__ float smem[];

//   wmma_indexes<wx_per_wmma_m, wy_per_wmma_n> wmma_idx(bx_per_w, by_per_w, bx_per_wx, bx, by, wx, wy, wk);
}



__global__ void embeddingln_backward_kernel(const float *x,
                      float *dw, const float *dy, const int tile_size,
                      int B, int C, int OC); 