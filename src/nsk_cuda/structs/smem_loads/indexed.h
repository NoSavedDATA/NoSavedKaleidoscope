#pragma once


#include "../../smem/include.h"
#include "../smem_cpasync_loader.h"
#include "../wmma_indexes.h"
#include "../fp16_wmma_frags.h"




template<int warp_rows_per_m, int warp_cols_per_n>
__device__ void smem_cpasync_wmma_loader<warp_rows_per_m, warp_cols_per_n>::load_A_indexed(float *x_smem, const float *embedding_book, const float *idxs, int next_tile, int M, int N) {
    
    // printf("smem_cpasync_wmma_loader<>::load_A_indexed.");
    for(int block_tile=0; block_tile<wmma_idx.by_per_w/4; ++block_tile) 
    {
        int row = wmma_idx.block_y*wmma_idx.blocking_size_y + wmma_idx.by_warp_offset + block_tile*4 + wmma_idx.ml; 
        int idx = (int)idxs[row];
        // printf("M %d - row %d - idx %d \n", M, row, idx);
        int col = next_tile+wmma_idx.mw*4;
        float const *gmem_ptr = embedding_book + idx*N + col; 

        
        int trunc_to = std::max(std::min(( (N-col))*4, 16), 0);
        if (trunc_to==16)
            gmem_to_smem_xor(gmem_ptr,  *(x_smem + xor_store_offset + (wmma_idx.by_warp_offset + block_tile*4)*wmma_idx.wk + xor_addr), 
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
    }
}
template<int warp_rows_per_m, int warp_cols_per_n>
__device__ void smem_cpasync_wmma_loader<warp_rows_per_m, warp_cols_per_n>::load_B_indexed(float *x_smem, const float *embedding_book, const float *idxs, int next_tile, int M, int N) {
    // for(int block_tile=0; block_tile<wmma_idx.bx_per_w/4; ++block_tile)
    // {
    //     int row = wmma_idx.block_x*wmma_idx.blocking_size_x + wmma_idx.bx_warp_offset + block_tile*4 + wmma_idx.ml;
    //     int col = next_tile+wmma_idx.mw*4; 
    //     float const *gmem_ptr = embedding_book + row*N + col;

        
    //     int trunc_to = std::max(std::min(( (N-col))*4, 16), 0);
    //     if (trunc_to==16)
    //         gmem_to_smem_xor(gmem_ptr,  *(x_smem + xor_store_offset + (wmma_idx.bx_warp_offset + block_tile*4)*wmma_idx.wk + xor_addr),
    //                     (row<M) ? std::max(std::min(( (N-(next_tile+wmma_idx.mw*4)))*4, 16), 0) : 0);
    //     else {
    //         float *smem_ptr = x_smem + xor_store_offset + (wmma_idx.bx_warp_offset + block_tile*4)*wmma_idx.wk + xor_addr;
    //         for(int i=0;i<4;++i)
    //         {
    //             if (row<M && (col+i)<N)
    //             smem_ptr[i] = gmem_ptr[i];
    //             else
    //             smem_ptr[i] = 0.0f;
    //         }
    //     }

    // }
    printf("smem_cpasync_wmma_loader<>::load_B_indexed is not implemented yet.");
}
