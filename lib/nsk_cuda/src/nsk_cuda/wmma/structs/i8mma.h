#pragma once

#include "../../structs/smem_cpasync_loader.h"
#include "../../structs/wmma_indexes.h"



// TOO SLOW. REGISTERS BUG


template<int warp_rows_per_m, int warp_cols_per_n, typename T>
struct INT8MMA {

    int reg_A[2][warp_cols_per_n][4];
    int reg_B[2][warp_rows_per_m][4];
    int O[warp_rows_per_m][warp_cols_per_n][8];

    size_t reg_store_idx = 0;
    size_t reg_load_idx = 0;

    __device__ INT8MMA() {

#pragma unroll
    for (int i=0; i<warp_rows_per_m; ++i)
#pragma unroll
        for (int j=0; j<warp_cols_per_n; ++j)
#pragma unroll
            for (int k=0; k<8; ++k)
                O[i][j][k] = 0;
    }


    __device__ void swap() {
        reg_store_idx ^= 1;
        reg_load_idx ^= 1;
    }




    __device__ void load_A(float *x_smem,
                           wmma_indexes<warp_rows_per_m, warp_cols_per_n>& wmma_idx,
                           smem_cpasync_wmma_loader<warp_rows_per_m, warp_cols_per_n, T>& smem_loader,
                           const int WMMA_N, int k_stride) 
                                        
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
    }




    __device__ void load_B(float *w_smem,
                           wmma_indexes<warp_rows_per_m, warp_cols_per_n>& wmma_idx,
                           smem_cpasync_wmma_loader<warp_rows_per_m, warp_cols_per_n, T>& smem_loader,
                           const int WMMA_M, int k_stride) 
                                        
    {
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
    }




    __device__ void mma() {

#pragma unroll
        for (int wy_tile=0; wy_tile<warp_cols_per_n; ++wy_tile)
        { 
            // for (int wy_tile=0; wy_tile<1; ++wy_tile)
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


                // asm volatile(
                //     "mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 "
                //     "{%0,%1,%2,%3}, "    // D matrix
                //     "{%4,%5}, "                     // A matrix
                //     "{%6}, "                   // B matrix
                //     "{%7, %8, %9, %10};\n"
                //     : "=r"(O[(j_s*warp_cols_per_n + wy_tile)*8]), "=r"(O[(j_s*warp_cols_per_n + wy_tile)*8+1]), "=r"(O[(j_s*warp_cols_per_n + wy_tile)*8+2]) , "=r"(O[+(j_s*warp_cols_per_n + wy_tile)*8+3]) 
                //     : "r"(reg_A[reg_load_idx][wy_tile][0]), "r"(reg_A[reg_load_idx][wy_tile][1]),
                //       "r"(reg_B[reg_load_idx][j_s][0]),
                //       "r"(O[(j_s*warp_cols_per_n + wy_tile)*8]), "r"(O[(j_s*warp_cols_per_n + wy_tile)*8+1]), "r"(O[(j_s*warp_cols_per_n + wy_tile)*8+2]) , "r"(O[+(j_s*warp_cols_per_n + wy_tile)*8+3]) );
                // asm volatile(
                //     "mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 "
                //     "{%0,%1,%2,%3}, "    // D matrix
                //     "{%4,%5}, "                     // A matrix
                //     "{%6}, "                   // B matrix
                //     "{%7, %8, %9, %10};\n"
                //     : "=r"(O[(j_s*warp_cols_per_n + wy_tile)*8+4]), "=r"(O[(j_s*warp_cols_per_n + wy_tile)*8+5]), "=r"(O[(j_s*warp_cols_per_n + wy_tile)*8+6]) , "=r"(O[+(j_s*warp_cols_per_n + wy_tile)*8+7]) 
                //     : "r"(reg_A[reg_load_idx][wy_tile][0]), "r"(reg_A[reg_load_idx][wy_tile][1]),
                //       "r"(reg_B[reg_load_idx][j_s][1]),
                //       "r"(O[(j_s*warp_cols_per_n + wy_tile)*8+4]), "r"(O[(j_s*warp_cols_per_n + wy_tile)*8+5]), "r"(O[(j_s*warp_cols_per_n + wy_tile)*8+6]) , "r"(O[+(j_s*warp_cols_per_n + wy_tile)*8+7]));

            }
        }
    }
    

};