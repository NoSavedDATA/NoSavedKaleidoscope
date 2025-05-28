#pragma once

#include "wmma_indexes.h"


template<int warp_rows_per_m, int warp_cols_per_n>
struct fp16_wmma_frags {

    
  wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> x_frag[warp_rows_per_m];
  wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> w_frag[warp_cols_per_n];

  float acc_frag[warp_rows_per_m*warp_cols_per_n*8];


  __device__ fp16_wmma_frags() {
    for (int i=0; i<warp_rows_per_m*warp_cols_per_n*8; ++i)
        acc_frag[i] = 0.0f;
  }

  // __device__ void load_frag_A(smem_cpasync_wmma_loader<warp_rows_per_m, warp_cols_per_n> &smem_loader,
  //                             wmma_indexes<warp_rows_per_m, warp_cols_per_n> &wmma_idx, float *x_smem,
  //                             const int WMMA_N, int k_stride)
  // {
  //   for (int w_tile=0; w_tile<warp_cols_per_n; ++w_tile)
  //   {

  //     // if ((block_x+block_y+laneId+warp_x)==0)
  //       // printf("wy tile %d, wy_per_wmma_n: %d, warp y: %d, row: %d\n", w_tile, warp_cols_per_n, wmma_idx.warp_y, (wmma_idx.warp_y*wmma_idx.wy + w_tile*WMMA_N));
  //     // __syncthreads();
  //     smem_xor_to_reg_A(x_frag[w_tile], x_smem + smem_loader.xor_load_offset + (wmma_idx.warp_y*wmma_idx.wy + w_tile*WMMA_N)*wmma_idx.wk, wmma_idx.wk, k_stride);

  //   }
  // }
};