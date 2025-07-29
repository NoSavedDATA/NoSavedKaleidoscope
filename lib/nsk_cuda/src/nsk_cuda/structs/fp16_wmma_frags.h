#pragma once

#include "wmma_indexes.h"


template<int warp_rows_per_m, int warp_cols_per_n, typename T>
struct fp16_wmma_frags {

    
  wmma::fragment<wmma::matrix_a, 16, 16, 16, T, wmma::row_major> x_frag[warp_rows_per_m];
  wmma::fragment<wmma::matrix_b, 16, 16, 16, T, wmma::col_major> w_frag[warp_cols_per_n];

  float acc_frag[warp_rows_per_m*warp_cols_per_n*8];


  __device__ fp16_wmma_frags() {
    for (int i=0; i<warp_rows_per_m*warp_cols_per_n*8; ++i)
        acc_frag[i] = 0.0f;
  }



  __device__ void investigate_mapping(float *smem) {
    if(threadIdx.x==0&&blockIdx.x==0&&blockIdx.y==0)
      printf("Investigate x_frag mapping:\n");
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