#pragma once

#include <mma.h>

#include "../../structs/i8_wmma_frags.h"
#include "i8_16x16x16_warp_tile_mma.h"

using namespace nvcuda;




template<int warp_rows_per_m, int warp_cols_per_n, typename T>
__device__ void blocking_tiled_wmma_i8_16x16x16_mma(float *out_tensor, const float *scale_M, const float *scale_N, int *out,
                                              wmma_indexes<warp_rows_per_m, warp_cols_per_n>& wmma_idx,
                                              smem_cpasync_wmma_loader<warp_rows_per_m, warp_cols_per_n, T>& smem_loader,
                                              const int8_t *x, const int8_t *w, float *x_smem, float *w_smem,
                                              const int M, const int N, const int K, const int WMMA_M, const int WMMA_N)
{

    smem_loader.load_A(x_smem, x, 0, M, K);
    smem_loader.load_B(w_smem, w, 0, N, K);

    asm volatile("cp.async.commit_group;\n" ::);
    

    size_t reg_store_idx = 0;
    size_t reg_load_idx = 0;

    int O[warp_rows_per_m][warp_cols_per_n][8];
#pragma unroll
    for (int i=0; i<warp_rows_per_m; ++i)
#pragma unroll
        for (int j=0; j<warp_cols_per_n; ++j)
#pragma unroll
            for (int k=0; k<8; ++k)
                O[i][j][k] = 0;


    int reg_A[2][warp_cols_per_n][4];
    int reg_B[2][warp_rows_per_m][4];




    #pragma unroll
    for (int tile=0; tile<K; tile+=wmma_idx.wk)
    {   

        // warp * mw_size * i_size + mw*i_size + i
        smem_loader.swap();


        int next_tile = tile + wmma_idx.wk;

        if (next_tile<K)
        {
            
            smem_loader.load_A(x_smem, x, next_tile, M, K);
            smem_loader.load_B(w_smem, w, next_tile, N, K);


            asm volatile("cp.async.commit_group;\n" ::);
            asm volatile("cp.async.wait_group %0;" ::"n"(1));
        } else {
            asm volatile("cp.async.wait_all;");
        }

        __syncthreads();

        // smem_loader.print_i8(x_smem+smem_loader.xor_load_offset, 64, 16);
        // smem_loader.print_i8(w_smem+smem_loader.xor_load_offset, 128, 16);



        
#pragma unroll
        for (int k_stride=0; k_stride<2; ++k_stride)
        {

#pragma unroll
            for (int i=0; i<warp_cols_per_n; ++i)
            {
                uint32_t x_smem_casted = __cvta_generic_to_shared(x_smem +\
                    smem_loader.xor_load_offset + \
                    (wmma_idx.warp_y*wmma_idx.wy + i*WMMA_N)*8 + \
                    k_stride*64 + \ 
                    ((wmma_idx.laneId/16)*16 + wmma_idx.laneId%16)*4);
       
                asm volatile("ldmatrix.sync.aligned.x2.m8n8.shared.b16 {%0, %1}, [%2];\n" \
                        : "=r"(reg_A[reg_store_idx][i][0]), "=r"(reg_A[reg_store_idx][i][1])
                        : "r"(x_smem_casted));
            }
                    

#pragma unroll
            for (int i=0; i<warp_rows_per_m; ++i)
            {
                uint32_t w_smem_casted = __cvta_generic_to_shared(w_smem +\
                    smem_loader.xor_load_offset + \
                    (wmma_idx.warp_x*wmma_idx.wx + i*WMMA_M)*8 + \
                    k_stride*64 + \
                    ((wmma_idx.laneId/16)*16 + wmma_idx.laneId%16)*4);
                    //   (((wmma_idx.laneId/8) %2)*16 + wmma_idx.laneId%8)*4);
                    
                asm volatile("ldmatrix.sync.aligned.x2.m8n8.shared.b16 {%0, %1}, [%2];\n" \
                        : "=r"(reg_B[reg_store_idx][i][0]), "=r"(reg_B[reg_store_idx][i][1])
                        : "r"(w_smem_casted));
            }

            reg_store_idx ^= 1;
            reg_load_idx ^= 1;



#pragma unroll
            for (int wy_tile=0; wy_tile<warp_cols_per_n; ++wy_tile)
            { 
#pragma unroll
                for (int wx_tile=0; wx_tile<warp_rows_per_m; ++wx_tile)
                {
                    size_t j_s = (wy_tile % 2) ? (warp_rows_per_m - wx_tile - 1) : wx_tile;

                    asm volatile(
                            "mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 "
                            "{%0,%1,%2,%3}, "    // D matrix
                            "{%4,%5}, "                     // A matrix
                            "{%6}, "                   // B matrix
                            "{%7, %8, %9, %10};\n"
                            : "=r"(O[j_s][wy_tile][0]), "=r"(O[j_s][wy_tile][1]), "=r"(O[j_s][wy_tile][2]) , "=r"(O[j_s][wy_tile][3]) 
                            : "r"(reg_A[reg_load_idx][wy_tile][0]), "r"(reg_A[reg_load_idx][wy_tile][1]),
                            "r"(reg_B[reg_load_idx][j_s][0]),
                            "r"(O[j_s][wy_tile][0]), "r"(O[j_s][wy_tile][1]), "r"(O[j_s][wy_tile][2]) , "r"(O[j_s][wy_tile][3]));
                    asm volatile(
                            "mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 "
                            "{%0,%1,%2,%3}, "    // D matrix
                            "{%4,%5}, "                     // A matrix
                            "{%6}, "                   // B matrix
                            "{%7, %8, %9, %10};\n"
                            : "=r"(O[j_s][wy_tile][4]), "=r"(O[j_s][wy_tile][5]), "=r"(O[j_s][wy_tile][6]) , "=r"(O[j_s][wy_tile][7]) 
                            : "r"(reg_A[reg_load_idx][wy_tile][0]), "r"(reg_A[reg_load_idx][wy_tile][1]),
                            "r"(reg_B[reg_load_idx][j_s][1]),
                            "r"(O[j_s][wy_tile][4]), "r"(O[j_s][wy_tile][5]), "r"(O[j_s][wy_tile][6]) , "r"(O[j_s][wy_tile][7]));
                }
            }
        }
    }
    

#pragma unroll
    for (int wy_tile=0; wy_tile<warp_cols_per_n; ++wy_tile)
#pragma unroll
        for (int wx_tile=0; wx_tile<warp_rows_per_m; ++wx_tile)
            for (int i=0;i<8;++i)
                out[(wx_tile*warp_cols_per_n + wy_tile)*8 + i] = O[wx_tile][wy_tile][i];

    // __syncthreads();

//     const int C_SMEM_OFFSET = 128;
//     const int C_SMEM_STRIDE = 64;

// #pragma unroll
//     for (int wy_tile=0; wy_tile<warp_cols_per_n; ++wy_tile)
//     {
// #pragma unroll
//         for (int wx_tile=0; wx_tile<warp_rows_per_m; ++wx_tile)
//         {
//             half *lane_ptr0 =
//                 smem_warp_tile_row_ptr + (i * MMA_M + lane_id / 4) * C_SMEM_STRIDE +
//                 ((warp_id % BLOCK_ROW_WARPS) * C_SMEM_OFFSET + j * MMA_N +
//                  (lane_id % 4) * sizeof(uint32_t) / sizeof(half) + ((lane_id / 4) % 8) * PERMUTED_OFFSET) %
//                     C_SMEM_STRIDE;
//             half *lane_ptr1 =
//                 smem_warp_tile_row_ptr + (i * MMA_M + lane_id / 4 + 8) * C_SMEM_STRIDE +
//                 ((warp_id % BLOCK_ROW_WARPS) * C_SMEM_OFFSET + j * MMA_N +
//                  (lane_id % 4) * sizeof(uint32_t) / sizeof(half) + ((lane_id / 4 + 8) % 8) * PERMUTED_OFFSET) %
//                     C_SMEM_STRIDE;

//             *((uint32_t *)(lane_ptr0)) = O[wx_tile][wy_tile][0];
//             *((uint32_t *)(lane_ptr1)) = O[wx_tile][wy_tile][1];
//         }
//     }

//     __syncthreads();

}