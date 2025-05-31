#pragma once

#include <mma.h>

#include "../../structs/fp16_wmma_frags.h"
#include "../fp16_16x16x16_warp_tile.h"

using namespace nvcuda;




template<int warp_rows_per_m, int warp_cols_per_n>
__device__ void blocking_tiled_wmma_fp16_16x16x16_L_index(fp16_wmma_frags<warp_rows_per_m, warp_cols_per_n> &frag_loader,
                                              wmma_indexes<warp_rows_per_m, warp_cols_per_n>& wmma_idx,
                                              smem_cpasync_wmma_loader<warp_rows_per_m, warp_cols_per_n>& smem_loader,
                                              const float *embedding_book, const float *idxs, const float *w, float *x_smem, float *w_smem,
                                              const int M, const int N, const int K, const int WMMA_M, const int WMMA_N)
{
    smem_loader.load_A_indexed(x_smem, embedding_book, idxs, 0, M, K);
    smem_loader.load_B(w_smem, w, 0, N, K);

    asm volatile("cp.async.commit_group;\n" ::);
    // asm volatile("cp.async.wait_all;");
    // __syncthreads();

    #pragma unroll
    for (int tile=0; tile<K; tile+=wmma_idx.wk)
    {
        // warp * mw_size * i_size + mw*i_size + i
        smem_loader.swap();

        // Each iter processes 4/ml rows and 8/mw*(4 floats) | 32 cols.
        // So, we jump 4|ml rows per iter.

        int next_tile = tile + wmma_idx.wk;

        if (next_tile<K)
        {
            
            smem_loader.load_A_indexed(x_smem, embedding_book, idxs, next_tile, M, K);
            smem_loader.load_B(w_smem, w, next_tile, N, K);


            asm volatile("cp.async.commit_group;\n" ::);
            asm volatile("cp.async.wait_group %0;" ::"n"(1));
        } else
            asm volatile("cp.async.wait_all;");
        

        __syncthreads();

        for (int k_stride=0; k_stride<2; ++k_stride)
        {
            smem_loader.store_frag_A(frag_loader, x_smem, WMMA_M, k_stride);
            smem_loader.store_frag_B(frag_loader, w_smem, WMMA_N, k_stride);

            warp_tiled_wmma_fp16_16x16x16(frag_loader, wmma_idx, M, N, WMMA_M, WMMA_N);
        }
        // asm volatile("cp.async.wait_all;");
        // __syncthreads();
    }
}