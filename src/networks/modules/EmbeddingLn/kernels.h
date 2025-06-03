#pragma once

#include <cuda_runtime.h>
#include <cublasLt.h>
#include <cublas_v2.h>
#include <curand_kernel.h>
#include <cuda_fp16.h>
#include <cudnn.h>
#include <mma.h>


#include "../../../nsk_cuda/include.h"



#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16




template<int wx_per_wmma_m, int wy_per_wmma_n>
__global__ void embeddingln_forward_kernel(const float *embedding_book, const float *idxs, const float *w,
                      float *out, 
                      int bx_per_w, int by_per_w, int bx_per_wx, int bx, int by, int wx, int wy, int wk,
                      const int M, const int N, const int K) {
  // B, OC, C

  fp16_wmma_frags<wx_per_wmma_m, wy_per_wmma_n, __half> frag_loader;
  extern __shared__ float smem[];


  wmma_indexes<wx_per_wmma_m, wy_per_wmma_n> wmma_idx(bx_per_w, by_per_w, bx_per_wx, bx, by, wx, wy, wk);

  smem_cpasync_wmma_loader<wx_per_wmma_m, wy_per_wmma_n, float> smem_loader(smem, wmma_idx, (bx+by)*wk);
  float *x_smem = smem_loader.smem_malloc(smem, by*wk);
  float *w_smem = smem_loader.smem_malloc(smem);


  blocking_tiled_wmma_fp16_16x16x16_L_index(frag_loader, wmma_idx, smem_loader,
                                    embedding_book, idxs, w, x_smem, w_smem,
                                    M, N, K, WMMA_M, WMMA_N);


  smem_loader.blocking_tiled_store_C(out, frag_loader, M, N, WMMA_M, WMMA_N, WMMA_K);
}




template<int wx_per_wmma_m, int wy_per_wmma_n>
__global__ void embeddingln_backward_dw(const float *x,
                                        const float *embedding_book, const float *idxs,
                                        float *out,
                                        int bx_per_w, int by_per_w, int bx_per_wx, int bx, int by, int wx, int wy, int wk,
                                        const int M, const int N, const int K)
{
  fp16_wmma_frags<wx_per_wmma_m, wy_per_wmma_n, __half> frag_loader;
  extern __shared__ float smem[];


  wmma_indexes<wx_per_wmma_m, wy_per_wmma_n> wmma_idx(bx_per_w, by_per_w, bx_per_wx, bx, by, wx, wy, wk);

  smem_cpasync_wmma_loader<wx_per_wmma_m, wy_per_wmma_n, float> smem_loader(smem, wmma_idx, (bx+by)*wk);
  float *x_smem = smem_loader.smem_malloc(smem, by*wk);
  float *w_smem = smem_loader.smem_malloc(smem);
                        

  blocking_tiled_wmma_fp16_16x16x16_dw_L_index(frag_loader, wmma_idx, smem_loader,
                                    x, embedding_book, idxs, x_smem, w_smem,
                                    M, N, K, WMMA_M, WMMA_N);



  smem_loader.blocking_tiled_store_C(out, frag_loader, M, N, WMMA_M, WMMA_N, WMMA_K);
}




template<int wx_per_wmma_m, int wy_per_wmma_n>
__global__ void embeddingln_backward_dx(const float *x, const float *w, float *out, const float *idxs,
                      int bx_per_w, int by_per_w, int bx_per_wx, int bx, int by, int wx, int wy, int wk,
                      const int M, const int N, const int K) {

  fp16_wmma_frags<wx_per_wmma_m, wy_per_wmma_n, __half> frag_loader;
  extern __shared__ float smem[];


  wmma_indexes<wx_per_wmma_m, wy_per_wmma_n> wmma_idx(bx_per_w, by_per_w, bx_per_wx, bx, by, wx, wy, wk);

  smem_cpasync_wmma_loader<wx_per_wmma_m, wy_per_wmma_n, float> smem_loader(smem, wmma_idx, (bx+by)*wk);
  float *x_smem = smem_loader.smem_malloc(smem, by*wk);
  float *w_smem = smem_loader.smem_malloc(smem);


  blocking_tiled_wmma_fp16_16x16x16_dx(frag_loader, wmma_idx, smem_loader,
                                    x, w, x_smem, w_smem,
                                    M, N, K, WMMA_M, WMMA_N);


  smem_loader.blocking_tiled_store_C_indexed(out, frag_loader, idxs, M, N, WMMA_M, WMMA_N, WMMA_K);
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