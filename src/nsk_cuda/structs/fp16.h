#pragma once


template<int warp_rows_per_m, int warp_cols_per_n>
struct fp16_mma_info {

    
  wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> x_frag[warp_rows_per_m];
  wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> w_frag[warp_cols_per_n];

  float acc_frag[warp_rows_per_m*warp_cols_per_n*8];

  __device__ fp16_mma_info() {

    // int blockidx = blockIdx.x;

    // printf("Block is %d", blockidx);

    for (int i=0; i<warp_rows_per_m*warp_cols_per_n*8; ++i)
        acc_frag[i] = 0.0f;

  }
};