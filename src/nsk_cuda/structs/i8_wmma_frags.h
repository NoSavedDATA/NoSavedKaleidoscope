#pragma once

#include "wmma_indexes.h"


template<int warp_rows_per_m, int warp_cols_per_n, typename T>
struct i8_wmma_frags {

    
  wmma::fragment<wmma::matrix_a, 16, 16, 16, T, wmma::row_major> x_frag[warp_rows_per_m];
  wmma::fragment<wmma::matrix_b, 16, 16, 16, T, wmma::col_major> w_frag[warp_cols_per_n];

  int acc_frag[warp_rows_per_m*warp_cols_per_n*8];


  __device__ i8_wmma_frags() {
    for (int i=0; i<warp_rows_per_m*warp_cols_per_n*8; ++i)
        acc_frag[i] = 0;
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


  __device__ void investigate_mapping(float *smem) {
    if(threadIdx.x==0&&blockIdx.x==0&&blockIdx.y==0)
      printf("Investigate w_frag mapping:\n");
    if(blockIdx.x==0&&blockIdx.y==0)
    {
      T *h_smem = (T *)smem;
      for (int i=0;i<16*16;++i)
        h_smem[i] = i;

      wmma::load_matrix_sync(x_frag[0], h_smem, 16);
      for (int i=0; i<32; ++i)
      {
        if (threadIdx.x==i)
        {
          for (int j=0; j<x_frag[0].num_elements; ++j)
            printf("%d, ", (uint8_t)((int)x_frag[0].x[j]));
          printf("\n");
        }
        __syncwarp();
      }
    }
  }
  
};