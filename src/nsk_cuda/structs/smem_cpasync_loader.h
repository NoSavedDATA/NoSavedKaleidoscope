#pragma once

#include "../smem/include.h"
#include "wmma_indexes.h"
#include "fp16_wmma_frags.h"
#include "i8_wmma_frags.h"


template<int warp_rows_per_m, int warp_cols_per_n, typename T>
struct smem_cpasync_wmma_loader {

  T *smem;
  int xor_addr;
  int xor_store_offset=0;
  int xor_load_offset, xor_swap;
  wmma_indexes<warp_rows_per_m, warp_cols_per_n>& wmma_idx;

  int smem_offset=0;

  __device__ smem_cpasync_wmma_loader(T *smem, wmma_indexes<warp_rows_per_m, warp_cols_per_n>& wmma_idx, int xor_swap)
        : smem(smem), wmma_idx(wmma_idx), xor_swap(xor_swap) {
    // printf("laneId is %d\n", wmma_idx.laneId);
    xor_addr = smem_xor_cp_async(wmma_idx.laneId);
    xor_load_offset = xor_swap;
  }



  __device__ void swap() {
    xor_store_offset ^= xor_swap;
    xor_load_offset ^= xor_swap;
  }


  __device__ T *smem_malloc(T *smem, int size) {
    T *ret_ptr = smem + smem_offset;
    smem_offset += size;
    return ret_ptr;
  }
  __device__ T *smem_malloc(T *smem) {
    return smem + smem_offset;
  }



  __device__ void print(T *x_smem, const int M, const int N) {
    if(threadIdx.x==0&&blockIdx.x==0&&blockIdx.y==0)
    {
      for(int i=0; i<M*N; ++i)
      {
        for(int j=0;j<N;++j)
          printf("%f, ", x_smem[(i/N)*N + i]);
        printf("\n");
      }
      printf("\n\n");
    }
    __syncthreads();
  }

  __device__ void print_i8(T *x_smem, int M, int N) {
    if(threadIdx.x==0&&blockIdx.x==0&&blockIdx.y==0)
    {
      int8_t *i_smem = (int8_t*)x_smem;

      for(int i=0; i<M; ++i)
      {
        for(int j=0;j<N;++j)
          printf("%d, ", (int)i_smem[i*N + j]);
        printf("\n");
      }
      printf("\n\n");
    }
    __syncthreads();
  }


  __device__ void load_A(T *x_smem, const float *x, int next_tile, int M, int N); 
  __device__ void load_B(T *x_smem, const float *x, int next_tile, int M, int N); 

  __device__ void load_A(T *x_smem, const int8_t *x, int next_tile, int M, int N); 
  __device__ void load_B(T *x_smem, const int8_t *x, int next_tile, int M, int N); 


  __device__ void load_A_transposed(T *x_smem, const float *x, int next_tile, int M, int N); 
  __device__ void load_B_transposed(T *x_smem, const float *x, int next_tile, int M, int N); 
  

  __device__ void load_A_transposed(T *x_smem, const int8_t *x, int next_tile, int M, int N);
  __device__ void load_B_transposed(T *x_smem, const int8_t *x, int next_tile, int M, int N);

  __device__ void load_A_indexed(T *x_smem, const float *embedding_book, const float *idxs, int next_tile, int M, int N);
  __device__ void load_B_indexed(T *x_smem, const float *embedding_book, const float *idxs, int next_tile, int M, int N);

  __device__ void load_A_transposed_indexed(T *x_smem, const float *embedding_book, const float *idxs, int next_tile, int M, int N);
  __device__ void load_B_transposed_indexed(T *x_smem, const float *embedding_book, const float *idxs, int next_tile, int M, int N);




  __device__ void store_frag_A(fp16_wmma_frags<warp_rows_per_m, warp_cols_per_n, __half> &frag_loader,
                              float *x_smem,
                              const int WMMA_N, int k_stride)
  {
    for (int w_tile=0; w_tile<warp_rows_per_m; ++w_tile) // Each warp tile handles 16 rows
      smem_xor_to_reg_A(frag_loader.x_frag[w_tile], x_smem + xor_load_offset + (wmma_idx.warp_y*wmma_idx.wy + w_tile*WMMA_N)*wmma_idx.wk, wmma_idx.wk, k_stride);
  }

  __device__ void store_frag_B(fp16_wmma_frags<warp_rows_per_m, warp_cols_per_n, __half> &frag_loader,
                              float *x_smem,
                              const int WMMA_M, int k_stride)
  {
    for (int w_tile=0; w_tile<warp_cols_per_n; ++w_tile)
        smem_xor_to_reg_B(frag_loader.w_frag[w_tile], x_smem + xor_load_offset + (wmma_idx.warp_x*wmma_idx.wx + w_tile*WMMA_M)*wmma_idx.wk, wmma_idx.wk, k_stride);
  }




  __device__ void store_frag_A(i8_wmma_frags<warp_rows_per_m, warp_cols_per_n, int8_t> &frag_loader,
                              float *x_smem,
                              const int WMMA_N, int k_stride)
  {
    for (int w_tile=0; w_tile<warp_rows_per_m; ++w_tile)
      smem_xor_to_reg_A(frag_loader.x_frag[w_tile], x_smem + xor_load_offset + (wmma_idx.warp_y*wmma_idx.wy + w_tile*WMMA_N)*32, 32, k_stride);
  }

  __device__ void store_frag_B(i8_wmma_frags<warp_rows_per_m, warp_cols_per_n, int8_t> &frag_loader,
                              float *x_smem,
                              const int WMMA_M, int k_stride)
  {
    for (int w_tile=0; w_tile<warp_cols_per_n; ++w_tile)
      smem_xor_to_reg_B(frag_loader.w_frag[w_tile], x_smem + xor_load_offset + (wmma_idx.warp_x*wmma_idx.wx + w_tile*WMMA_M)*32, 32, k_stride);
  }




  __device__ void store_C(float *out, float *out_smem, int threaded_row, int threaded_col,
                          int M, int N,
                          int WMMA_M, int WMMA_N, int WMMA_K) {
  #pragma unroll
    for (int tile=0; tile<std::ceil((WMMA_N*WMMA_M)/(float)(warpSize)); ++tile)
    {
      int tile_idx = tile*warpSize + wmma_idx.laneId;

      int row =  tile_idx / WMMA_M;
      int col = (tile_idx % WMMA_M);


      if((threaded_row+row)<M  &&  (threaded_col+col)<N && row<WMMA_K)
        out[(threaded_row+row)*N + threaded_col+col] = out_smem[row*(wmma_idx.bx_per_wx*WMMA_M)+col];
    }
  }
  

  __device__ void blocking_tiled_store_C(float *out,
                                         fp16_wmma_frags<warp_rows_per_m, warp_cols_per_n, __half> &frag_loader,
                                         int M, int N, const int WMMA_M, const int WMMA_N, const int WMMA_K)
  {

    float *out_smem = smem + wmma_idx.warp_y*WMMA_M*(wmma_idx.bx_per_wx*WMMA_K) + wmma_idx.warp_x*WMMA_N; // todo: is this correct?

    for (int wx_tile=0; wx_tile<warp_rows_per_m; ++wx_tile)
    {
      for (int wy_tile=0; wy_tile<warp_cols_per_n; ++wy_tile)
      {
        __syncthreads();

        int threaded_row = wmma_idx.block_y*wmma_idx.blocking_size_y + wmma_idx.warp_y*wmma_idx.wy + wy_tile*WMMA_M;
        int threaded_col = wmma_idx.block_x*wmma_idx.blocking_size_x + wmma_idx.warp_x*wmma_idx.wx + wx_tile*WMMA_N;

        if (threaded_row<M && threaded_col<N && (wmma_idx.warp_y*wmma_idx.wy)<M && (wmma_idx.warp_x*wmma_idx.wx)<N)
        {
          
          
          frag_to_mem(frag_loader.acc_frag+(wx_tile*warp_cols_per_n + wy_tile)*8, out_smem, wmma_idx.bx_per_wx*WMMA_K);
                  
          
          store_C(out, out_smem, threaded_row, threaded_col, M, N, WMMA_M, WMMA_N, WMMA_K);

        }
      }
    }
  }



  __device__ void store_C(float *out, float *out_smem, const float *scale_M, const float *scale_N, int threaded_row, int threaded_col,
                          int M, int N,
                          int WMMA_M, int WMMA_N, int WMMA_K) {
  #pragma unroll
    for (int tile=0; tile<std::ceil((WMMA_N*WMMA_M)/(float)(warpSize)); ++tile)
    {
      int tile_idx = tile*warpSize + wmma_idx.laneId;

      int row =  tile_idx / WMMA_M;
      int col = (tile_idx % WMMA_M);


      if((threaded_row+row)<M  &&  (threaded_col+col)<N && row<WMMA_K)
      {
        // if (blockIdx.x==0&&threadIdx.x==0)
        // {
        //   printf("Storing %f - %f\n", out_smem[row*(wmma_idx.bx_per_wx*WMMA_M)+col], out_smem[row*(wmma_idx.bx_per_wx*WMMA_M)+col] / (scale_M[threaded_row+row] * scale_N[threaded_col+col]));
        //   printf("Scale is: %f - %f\n", scale_M[threaded_row+row], scale_N[threaded_col+col]);
        // }
        
        out[(threaded_row+row)*N + threaded_col+col] = out_smem[row*(wmma_idx.bx_per_wx*WMMA_M)+col] / (scale_M[threaded_row+row] * scale_N[threaded_col+col]);
      }
    }
  }
  __device__ void blocking_tiled_store_C(float *out, const float *scale_M, const float *scale_N,
                                         i8_wmma_frags<warp_rows_per_m, warp_cols_per_n, int8_t> &frag_loader,
                                         int M, int N, const int WMMA_M, const int WMMA_N, const int WMMA_K)
  {

    // float *out_smem = smem + wmma_idx.warp_y*WMMA_M*(wmma_idx.bx_per_wx*WMMA_K) + wmma_idx.warp_x*WMMA_N; // todo: is this correct?
    
    float *out_smem = smem + wmma_idx.warpId*WMMA_K;


    for (int wx_tile=0; wx_tile<warp_rows_per_m; ++wx_tile)
    {
      for (int wy_tile=0; wy_tile<warp_cols_per_n; ++wy_tile)
      {
        __syncthreads();

        int threaded_row = wmma_idx.block_y*wmma_idx.blocking_size_y + wmma_idx.warp_y*wmma_idx.wy + wy_tile*WMMA_M;
        int threaded_col = wmma_idx.block_x*wmma_idx.blocking_size_x + wmma_idx.warp_x*wmma_idx.wx + wx_tile*WMMA_N;

        if (threaded_row<M && threaded_col<N && (wmma_idx.warp_y*wmma_idx.wy)<M && (wmma_idx.warp_x*wmma_idx.wx)<N)
        {
          
          frag_to_mem(frag_loader.acc_frag+(wx_tile*warp_cols_per_n + wy_tile)*8, out_smem, 64);
          
          
          store_C(out, out_smem, scale_M, scale_N, threaded_row, threaded_col, M, N, WMMA_M, WMMA_N, WMMA_K);
        }
      }
    }
  }





  __device__ void store_C_indexed(float *out, const float *idxs, float *out_smem, int threaded_row, int threaded_col,
                          int M, int N,
                          int WMMA_M, int WMMA_N, int WMMA_K) {
  #pragma unroll
    for (int tile=0; tile<std::ceil((WMMA_N*WMMA_M)/(float)(warpSize)); ++tile)
    {
      int tile_idx = tile*warpSize + wmma_idx.laneId;

      int row =  tile_idx / WMMA_M;
      int col = (tile_idx % WMMA_M);


      if((threaded_row+row)<M  &&  (threaded_col+col)<N && row<WMMA_K)
      {
        int idx = (int)idxs[threaded_row+row];
        float *_out = out + idx*N + threaded_col+col;
        // out[(threaded_row+row)*N + threaded_col+col] = out_smem[row*(wmma_idx.bx_per_wx*WMMA_M)+col];
        // printf("row %d - idx %d\n", threaded_row+row, idx);
        atomicAdd(_out, out_smem[row*(wmma_idx.bx_per_wx*WMMA_M)+col]);
      }
    }
  }
  
  __device__ void blocking_tiled_store_C_indexed(float *out,
                                         fp16_wmma_frags<warp_rows_per_m, warp_cols_per_n, __half> &frag_loader,
                                         const float *idxs,
                                         int M, int N, const int WMMA_M, const int WMMA_N, const int WMMA_K)
  {

    float *out_smem = smem + wmma_idx.warp_y*WMMA_M*(wmma_idx.bx_per_wx*WMMA_K) + wmma_idx.warp_x*WMMA_N; // todo: is this correct?

    for (int wx_tile=0; wx_tile<warp_rows_per_m; ++wx_tile)
    {
      for (int wy_tile=0; wy_tile<warp_cols_per_n; ++wy_tile)
      {
        __syncthreads();

        int threaded_row = wmma_idx.block_y*wmma_idx.blocking_size_y + wmma_idx.warp_y*wmma_idx.wy + wy_tile*WMMA_M;
        int threaded_col = wmma_idx.block_x*wmma_idx.blocking_size_x + wmma_idx.warp_x*wmma_idx.wx + wx_tile*WMMA_N;

        if (threaded_row<M && threaded_col<N && (wmma_idx.warp_y*wmma_idx.wy)<M && (wmma_idx.warp_x*wmma_idx.wx)<N)
        {
          
          
          frag_to_mem(frag_loader.acc_frag+(wx_tile*warp_cols_per_n + wy_tile)*8, out_smem, wmma_idx.bx_per_wx*WMMA_K);
                  
          
          store_C_indexed(out, idxs, out_smem, threaded_row, threaded_col, M, N, WMMA_M, WMMA_N, WMMA_K);

        }
      }
    }
  }

};