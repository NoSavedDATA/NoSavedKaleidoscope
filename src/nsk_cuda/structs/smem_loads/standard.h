#pragma once


#include "../../smem/include.h"
#include "../smem_cpasync_loader.h"
#include "../wmma_indexes.h"
#include "../fp16_wmma_frags.h"




template<int warp_rows_per_m, int warp_cols_per_n, typename T>
__device__ void smem_cpasync_wmma_loader<warp_rows_per_m, warp_cols_per_n, T>::load_A(T *x_smem, const float *x, int next_tile, int M, int N) {
    for(int block_tile=0; block_tile<wmma_idx.by_per_w/4; ++block_tile) // 4 from 4 jumped rows
    {
        int row = wmma_idx.block_y*wmma_idx.blocking_size_y + wmma_idx.by_warp_offset + block_tile*4 + wmma_idx.ml; // 4 from 4 ml loaded rows
        int col = next_tile+wmma_idx.mw*4;
        float const *gmem_ptr = x + row*N + col; // 4 from 4 loaded floats of cp.async
                                                                    // mw goes up to 8, so it completes a tile of 32 floats.

        // Each thread copies 1 row  per 4  columns of floats / 16 Bytes.
        // Each warp   copies 4 rows per 32 columns of floats.
        
        int trunc_to = std::max(std::min(( (N-col))*4, 16), 0);
        if (trunc_to==16)
            gmem_to_smem_xor(gmem_ptr,  *(x_smem + xor_store_offset + (wmma_idx.by_warp_offset + block_tile*4)*wmma_idx.wk + xor_addr), // 4 from 4 ml loaded rows, wk=32
                        (row<M) ? trunc_to : 0); // last *4 tells that sizeof float is 4
        else
        {
            // printf("Truncating to %d | N: %d\n", trunc_to, N);
            float *smem_ptr = x_smem + xor_store_offset + (wmma_idx.by_warp_offset + block_tile*4)*wmma_idx.wk + xor_addr;
            for(int i=0; i<4; ++i)
            {
                if (row<M && (col+i)<N)
                    smem_ptr[i] = gmem_ptr[i];
                else
                    smem_ptr[i] = 0.0f;
            }
        }
        // xor addr(laneId): up to 128, jump by 4 
    }
}
template<int warp_rows_per_m, int warp_cols_per_n, typename T>
__device__ void smem_cpasync_wmma_loader<warp_rows_per_m, warp_cols_per_n, T>::load_B(T *x_smem, const float *x, int next_tile, int M, int N) {
    for(int block_tile=0; block_tile<wmma_idx.bx_per_w/4; ++block_tile)
    {
        int row = wmma_idx.block_x*wmma_idx.blocking_size_x + wmma_idx.bx_warp_offset + block_tile*4 + wmma_idx.ml;
        int col = next_tile+wmma_idx.mw*4; 
        float const *gmem_ptr = x + row*N + col;

        
        int trunc_to = std::max(std::min(( (N-col))*4, 16), 0);
        if (trunc_to==16)
            gmem_to_smem_xor(gmem_ptr,  *(x_smem + xor_store_offset + (wmma_idx.bx_warp_offset + block_tile*4)*wmma_idx.wk + xor_addr),
                        (row<M) ? std::max(std::min(( (N-(next_tile+wmma_idx.mw*4)))*4, 16), 0) : 0);
        else {
            float *smem_ptr = x_smem + xor_store_offset + (wmma_idx.bx_warp_offset + block_tile*4)*wmma_idx.wk + xor_addr;
            for(int i=0;i<4;++i)
            {
                if (row<M && (col+i)<N)
                    smem_ptr[i] = gmem_ptr[i];
                else
                    smem_ptr[i] = 0.0f;
            }
        }

    }
}












template<int warp_rows_per_m, int warp_cols_per_n, typename T>
__device__ void smem_cpasync_wmma_loader<warp_rows_per_m, warp_cols_per_n, T>::load_A(T *x_smem, const int8_t *x, int next_tile, int M, int N) {

    int copy_chunks = floor((wmma_idx.blocking_size_y + (wmma_idx.cp_rows*wmma_idx.num_warps-1) ) / ((float)(wmma_idx.cp_rows*wmma_idx.num_warps)) );
        
    // if(blockIdx.x==0&&blockIdx.y==0&&threadIdx.x==0)
    //     printf("Copy chunks: %d - mem used: %d\n", copy_chunks, copy_chunks*wmma_idx.cp_rows*wmma_idx.num_warps*4);
    
    // if(wmma_idx.warpId!=0)
    for(int block_tile=0; block_tile<copy_chunks; ++block_tile)
    {
        int row = wmma_idx.block_y*wmma_idx.blocking_size_y + (wmma_idx.warpId*copy_chunks + block_tile)*wmma_idx.cp_rows + wmma_idx.ml;
        // int row = wmma_idx.block_y*wmma_idx.blocking_size_y + wmma_idx.by_warp_offset + block_tile*wmma_idx.cp_rows + wmma_idx.ml; // 4 from 4 ml loaded rows
        int col = next_tile+wmma_idx.mw*16;
        const int8_t *gmem_ptr = x + row*N + col;

        // if(blockIdx.x==0&&blockIdx.y==0)
        //     printf("row %d - col %d - M %d - N %d\n", row, col, M, N);


        // Each thread copies 1 row  per 16  columns of int8 / 16 Bytes.
        // Each warp   copies 4 rows per 128 columns of int8.
        int trunc_to = std::max(std::min(( (N-col)), 16), 0);
        if (trunc_to==16)
        {
            gmem_to_smem_xor(gmem_ptr,  *(x_smem + xor_store_offset + (wmma_idx.warpId*copy_chunks + block_tile)*128 + wmma_idx.laneId*4), // 4 * 32 == jumps of 128
                (row<M) ? trunc_to : 0); // last *4 tells that sizeof float is 4
        }
        else
        {
            // printf("Truncating to %d | N: %d\n", trunc_to, N);
            float *smem_ptr = x_smem + xor_store_offset + (wmma_idx.warpId*copy_chunks + block_tile)*128 + wmma_idx.laneId*4;
            int8_t *smem_i8 = (int8_t *)smem_ptr; 
            for(int i=0; i<16; ++i)
            {
                if (row<M && (col+i)<N)
                    smem_i8[i] = gmem_ptr[i];
                else
                    smem_i8[i] = 0.0f;
            }
        }
    }
}

template<int warp_rows_per_m, int warp_cols_per_n, typename T>
__device__ void smem_cpasync_wmma_loader<warp_rows_per_m, warp_cols_per_n, T>::load_B(T *x_smem, const int8_t *x, int next_tile, int M, int N) {

    int copy_chunks = floor((wmma_idx.blocking_size_x + (wmma_idx.cp_rows*wmma_idx.num_warps-1) ) / ((float)(wmma_idx.cp_rows*wmma_idx.num_warps)) );

    // if(wmma_idx.warpId!=0)
    for(int block_tile=0; block_tile<copy_chunks; ++block_tile)
    {
        int row = wmma_idx.block_x*wmma_idx.blocking_size_x + (wmma_idx.warpId*copy_chunks + block_tile)*wmma_idx.cp_rows + wmma_idx.ml;
        // int row = wmma_idx.block_x*wmma_idx.blocking_size_x + wmma_idx.bx_warp_offset + block_tile*wmma_idx.cp_rows + wmma_idx.ml;
        int col = next_tile+wmma_idx.mw*16; 
        const int8_t *gmem_ptr = x + row*N + col;

        
        int trunc_to = std::max(std::min(( (N-col)), 16), 0);
        if (trunc_to==16)
            gmem_to_smem_xor(gmem_ptr,  *(x_smem + xor_store_offset + (wmma_idx.warpId*copy_chunks + block_tile)*128 + wmma_idx.laneId*4), 
                        (row<M) ? trunc_to : 0);
        else {
            float *smem_ptr = x_smem + xor_store_offset + (wmma_idx.warpId*copy_chunks + block_tile)*128 + wmma_idx.laneId*4;
            int8_t *smem_i8 = (int8_t *)smem_ptr;
            for(int i=0; i<16; ++i)
            {
                if (row<M && (col+i)<N)
                    smem_i8[i] = gmem_ptr[i];
                else
                    smem_i8[i] = 0;
            }
        }

    }
}
