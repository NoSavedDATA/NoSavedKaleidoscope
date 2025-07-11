#pragma once


#include "../../smem/include.h"
#include "../smem_cpasync_loader.h"
#include "../wmma_indexes.h"
#include "../fp16_wmma_frags.h"




template<int warp_rows_per_m, int warp_cols_per_n, typename T>
__device__ void smem_cpasync_wmma_loader<warp_rows_per_m, warp_cols_per_n, T>::load_A_transposed(T *x_smem, const float *x, int next_tile, int M, int N) {
    for(int block_tile=0; block_tile<wmma_idx.by_per_w/4; ++block_tile) 
    {
        int col = wmma_idx.block_y*wmma_idx.blocking_size_y + wmma_idx.by_warp_offset + block_tile*4 + wmma_idx.ml; 
        float *smem_out = x_smem + xor_store_offset + (wmma_idx.by_warp_offset + block_tile*4)*wmma_idx.wk + xor_addr;

        #pragma unroll
        for(int i=0;i<4;++i)
        {
            int row = next_tile+wmma_idx.mw*4 + i; 
            
            if (row<M && col<N)
                smem_out[i] = x[row*N + col];
            else
                smem_out[i] = 0.0f;
        } 
    }
} 

template<int warp_rows_per_m, int warp_cols_per_n, typename T>
__device__ void smem_cpasync_wmma_loader<warp_rows_per_m, warp_cols_per_n, T>::load_B_transposed(T *x_smem, const float *x, int next_tile, int M, int N) {
    for(int block_tile=0; block_tile<wmma_idx.bx_per_w/4; ++block_tile)
    {
      int col = wmma_idx.block_x*wmma_idx.blocking_size_x + wmma_idx.bx_warp_offset + block_tile*4 + wmma_idx.ml;
      float *smem_out  = x_smem + xor_store_offset + (wmma_idx.bx_warp_offset + block_tile*4)*wmma_idx.wk + xor_addr;

      #pragma unroll
      for(int i=0; i<4; i++) // Simulate copy 4 floats of cp.async
      { 
        int row = next_tile+wmma_idx.mw*4 + i;


        if (row<M && col<N)
          smem_out[i] = x[row*N + col];
        else
          smem_out[i] = 0.0f;
      }
    }
} 







template<int warp_rows_per_m, int warp_cols_per_n, typename T>
__device__ void smem_cpasync_wmma_loader<warp_rows_per_m, warp_cols_per_n, T>::load_A_transposed(T *x_smem, const int8_t *x, int next_tile, int M, int N) {
    for(int block_tile=0; block_tile<wmma_idx.by_per_w/4; ++block_tile) 
    {
      int col = wmma_idx.block_y*wmma_idx.blocking_size_y + wmma_idx.by_warp_offset + block_tile*4 + wmma_idx.ml; 
      float *smem_out = x_smem + xor_store_offset + (wmma_idx.by_warp_offset + block_tile*4)*wmma_idx.wk + xor_addr;
      int8_t *smem_i8 = (int8_t *)smem_out; 

      // const int8_t *x_shift = x + col;

      #pragma unroll
      for(int i=0;i<16;++i)
      {
          int row = next_tile+wmma_idx.mw*16 + i; 
          
          if (row<M && col<N)
              smem_i8[i] = x[row*N + col];
          else
              smem_i8[i] = 0.0f;
      } 
    }
} 

template<int warp_rows_per_m, int warp_cols_per_n, typename T>
__device__ void smem_cpasync_wmma_loader<warp_rows_per_m, warp_cols_per_n, T>::load_B_transposed(T *x_smem, const int8_t *x, int next_tile, int M, int N) {
    for(int block_tile=0; block_tile<wmma_idx.bx_per_w/4; ++block_tile)
    {
      int col = wmma_idx.block_x*wmma_idx.blocking_size_x + wmma_idx.bx_warp_offset + block_tile*4 + wmma_idx.ml;
      float *smem_out  = x_smem + xor_store_offset + (wmma_idx.bx_warp_offset + block_tile*4)*wmma_idx.wk + xor_addr;
      int8_t *smem_i8 = (int8_t *)smem_out; 

      #pragma unroll
      for(int i=0; i<16; i++) // Simulate copy 4 floats of cp.async
      { 
        int row = next_tile+wmma_idx.mw*16 + i;


        if (row<M && col<N)
          smem_i8[i] = x[row*N + col];
        else
          smem_i8[i] = 0.0f;
      }
    }
} 