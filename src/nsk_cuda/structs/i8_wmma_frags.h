#pragma once

#include "wmma_indexes.h"



// struct __align__(16) {
//     int8_t x_frag[8 * 4];      // 32B (already 16B-aligned)
//     int8_t w_frag[8 * 2];      // 16B
//     uint8_t _pad[16];          // Explicit padding (optional, but safe)
//     int acc_frag[4 * 2 * 8];   // 64B
// } frags;


template<int warp_rows_per_m, int warp_cols_per_n, typename T>
struct alignas(16) i8_wmma_frags {

    
  // wmma::fragment<wmma::matrix_a, 16, 16, 16, T, wmma::row_major> x_frag[warp_rows_per_m];
  // wmma::fragment<wmma::matrix_b, 16, 16, 16, T, wmma::col_major> w_frag[warp_cols_per_n];
    alignas(16) int8_t x_frag[8 * warp_rows_per_m];
    // int8_t _pad_x[16 - (8 * warp_rows_per_m) % 16];
    // int8_t _pad_x[16 - (8 * warp_rows_per_m) % 16];  // Pad to 16B
    alignas(16) int8_t w_frag[8 * warp_cols_per_n];
    // int8_t _pad_w[16 - (8 * warp_cols_per_n) % 16];
    // int8_t _pad_w[16 - (8 * warp_cols_per_n) % 16];  // Pad to 16B
    alignas(16) int padding_placeholder;
    alignas(16) int acc_frag[warp_rows_per_m * warp_cols_per_n * 8];


  __device__ i8_wmma_frags() {
    zero_frag();
    // printf("x_frag: %p, w_frag: %p, acc_frag: %p\n", x_frag, w_frag, acc_frag);
    // printf("wm %d, wn%d\n", warp_rows_per_m, warp_cols_per_n);
  }

  __device__ void zero_frag() {
    for (int i=0; i<warp_rows_per_m*warp_cols_per_n*8; ++i)
        acc_frag[i] = 0;
  }



  // __device__ void investigate_mapping(float *smem) {
  //   if(threadIdx.x==0&&blockIdx.x==0&&blockIdx.y==0)
  //     printf("Investigate w_frag mapping:\n");
  //   if(blockIdx.x==0&&blockIdx.y==0)
  //   {
  //     T *h_smem = (T *)smem;
  //     for (int i=0;i<16*16;++i)
  //       h_smem[i] = i;

  //     wmma::load_matrix_sync(x_frag[0], h_smem, 16);
  //     for (int i=0; i<32; ++i)
  //     {
  //       if (threadIdx.x==i)
  //       {
  //         for (int j=0; j<x_frag[0].num_elements; ++j)
  //           printf("%d, ", (uint8_t)((int)x_frag[0].x[j]));
  //         printf("\n");
  //       }
  //       __syncwarp();
  //     }
  //   }
  // }
  
};