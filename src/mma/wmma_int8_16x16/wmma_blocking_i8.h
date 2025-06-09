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
__global__ void wmma_blocking_i8(const int8_t *__restrict__ x, const int8_t *__restrict__ w,
                        float *__restrict__ out, const float *scale_M, const float *scale_N,
                        const int M, const int N, const int K,
                        const int num_warps,
                        const int bx, const int by,
                        const int wx, const int wy,
                        const int bx_per_w,     const int by_per_w,
                        const int bx_per_wx)
 {

  // bx: 128, by: 64
  // wx: 32,  wy: 32
  // wmma_t: 16
  // wk: 128 to handle 1B


  // if(blockIdx.x==0&&blockIdx.y==0&&threadIdx.x==0)
  //   printf("WXS AND WYS IS %d/%d of %d/%d\n", wx_per_wmma_m, wy_per_wmma_n, wx, wy);
  
  i8_wmma_frags<wx_per_wmma_m, wy_per_wmma_n, int8_t> frag_loader;

  
  extern __shared__ float smem[];
  // frag_loader.investigate_mapping(smem);


  wmma_indexes<wx_per_wmma_m, wy_per_wmma_n> wmma_idx(bx_per_w, by_per_w, bx_per_wx, bx, by, wx, wy, 32, 2, num_warps);
  // Load 2 columns of 16, because the maximum supported number of rows per ptx is wmma_m==16
  
  smem_cpasync_wmma_loader<wx_per_wmma_m, wy_per_wmma_n, float> smem_loader(smem, wmma_idx, (bx+by)*8);


  // if(blockIdx.x==0&&blockIdx.y==0&&threadIdx.x==0)
  //   printf("by %d by*32 %d\n", by, by*32);
  float *x_smem = smem_loader.smem_malloc(smem, by*8); // 8 floats that store 2 rows of 16 columns of int8
  float *w_smem = smem_loader.smem_malloc(smem);



  blocking_tiled_wmma_i8_16x16x16(frag_loader, wmma_idx, smem_loader,
                                    x, w, x_smem, w_smem,
                                    M, N, K, WMMA_M, WMMA_N);


  __syncthreads();

  smem_loader.blocking_tiled_store_C(out, scale_M, scale_N, frag_loader, M, N, WMMA_M, WMMA_N, WMMA_T);
}