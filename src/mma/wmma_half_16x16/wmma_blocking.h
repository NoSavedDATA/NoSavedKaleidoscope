#pragma once

#include "../utils.h"
#include "../../nsk_cuda/include.h"

using namespace nvcuda;

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16





// Each thread-block handles bx rows and by cols
// Each bx work is splitted accross wx warp loads

template<int WMMA_T, int wx_per_wmma_m, int wy_per_wmma_n, int wk>
__global__ void wmma_blocking(const float *__restrict__ x, const float *__restrict__ w,
                        float *__restrict__ out, const int M, const int K, const int N,
                        const int bx, const int by,
                        const int wx, const int wy,
                        const int bx_per_w,     const int by_per_w,
                        const int bx_per_wx)
 {

  // bx: 128, by: 64
  // wx: 32,  wy: 32
  // wmma_t: 16
  // wk: 32

  
  fp16_wmma_frags<wx_per_wmma_m, wy_per_wmma_n, __half> frag_loader;
    
  extern __shared__ float smem[];


  wmma_indexes<wx_per_wmma_m, wy_per_wmma_n> wmma_idx(bx_per_w, by_per_w, bx_per_wx, bx, by, wx, wy, wk);
  
  smem_cpasync_wmma_loader<wx_per_wmma_m, wy_per_wmma_n, float> smem_loader(smem, wmma_idx, (bx+by)*wk);
  float *x_smem = smem_loader.smem_malloc(smem, by*wk);
  float *w_smem = smem_loader.smem_malloc(smem);



  blocking_tiled_wmma_fp16_16x16x16(frag_loader, wmma_idx, smem_loader,
                                    x, w, x_smem, w_smem,
                                    M, N, K, WMMA_M, WMMA_N);



  smem_loader.blocking_tiled_store_C(out, frag_loader, M, N, WMMA_M, WMMA_N, WMMA_T);

}