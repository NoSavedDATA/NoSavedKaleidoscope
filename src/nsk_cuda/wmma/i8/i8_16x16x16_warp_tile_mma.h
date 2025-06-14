// #pragma once

// #include <mma.h>

// #include "../../structs/i8_wmma_frags.h"
// #include "ptx.h"

// using namespace nvcuda;



// #define LDMATRIX_X2(R0, R1, addr) \
//     asm volatile("ldmatrix.sync.aligned.x2.m8n8.shared.b16 {%0, %1}, [%2];\n" : "=r"(R0), "=r"(R1) : "r"(addr))

// #define LDMATRIX_X4(R0, R1, R2, R3, addr)                                             \
//     asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n" \
//                  : "=r"(R0), "=r"(R1), "=r"(R2), "=r"(R3)                             \
//                  : "r"(addr))


// template<int warp_rows_per_m, int warp_cols_per_n>
// __device__ inline void warp_tiled_wmma_i8_16x16x16_mma(int *out, int *reg_A, int *reg_B,
//                                               wmma_indexes<warp_rows_per_m, warp_cols_per_n>& wmma_idx,
//                                               const int M, const int N, const int WMMA_M, const int WMMA_N)
// {


//     // for (int wx_tile=0; wx_tile<1; ++wx_tile)
//     for (int wy_tile=0; wy_tile<warp_cols_per_n; ++wy_tile)
//     { 
//         // for (int wy_tile=0; wy_tile<1; ++wy_tile)
//         for (int wx_tile=0; wx_tile<warp_rows_per_m; ++wx_tile)
//         {
//             size_t j_s = (wy_tile % 2) ? (warp_rows_per_m - wx_tile - 1) : wx_tile;
//             // size_t j_s = wx_tile;

//             if ((wmma_idx.block_y*wmma_idx.blocking_size_y + wy_tile*WMMA_M)<M && (wmma_idx.block_x*wmma_idx.blocking_size_x + j_s*WMMA_N)<N)
//                 wmma16x16x16_i8_mma(out+(j_s*warp_cols_per_n + wy_tile)*8, reg_A+wy_tile*4, reg_B+j_s*4); // 8 is the frag ld  
//         }
//     }
//     // __syncthreads();
    
//     // if(threadIdx.x==0)
//     // {
//     //     printf("out is: ");
//     //     for (int i=0;i<8;++i)
//     //         printf("%d, ", out[i]);
//     //     printf("\n");

//     // }
    

// }






// template<int warp_rows_per_m, int warp_cols_per_n, typename T>
// __device__ inline void wmma_i8_m16n16k16_mma(int *out,
//                                         wmma_indexes<warp_rows_per_m, warp_cols_per_n>& wmma_idx,
//                                         smem_cpasync_wmma_loader<warp_rows_per_m, warp_cols_per_n, T>& smem_loader,
//                                         float *x_smem, float *w_smem,
//                                         const int M, const int N, const int K,
//                                         const int WMMA_M, const int WMMA_N,
//                                         int chunks)
// {
//     // chunks = 1;
//     size_t reg_store_idx = 0;
//     size_t reg_load_idx = 0;

//     int O[warp_rows_per_m][warp_cols_per_n][8];
// #pragma unroll
//     for (int i=0; i<warp_rows_per_m; ++i)
// #pragma unroll
//         for (int j=0; j<warp_cols_per_n; ++j)
// #pragma unroll
//             for (int k=0; k<8; ++k)
//                 O[i][j][k] = 0;


//     int reg_A[2][warp_cols_per_n][4];
//     int reg_B[2][warp_rows_per_m][4];


// #pragma unroll
//     for (int k_stride=0; k_stride<chunks; ++k_stride)
//     {
//         // smem_loader.store_frag_A(frag_loader, x_smem, WMMA_M, k_stride);
//         // smem_loader.store_frag_B(frag_loader, w_smem, WMMA_N, k_stride);


//         // if (threadIdx.x<32)
//         // {
            
// #pragma unroll
//         for (int i=0; i<warp_cols_per_n; ++i)
//         {

//             uint32_t x_smem_casted = __cvta_generic_to_shared(x_smem +\
//                 smem_loader.xor_load_offset + \
//                 (wmma_idx.warp_y*wmma_idx.wy + i*WMMA_N)*8 + \
//                 k_stride*64 + \ 
//                 ((wmma_idx.laneId/16)*16 + wmma_idx.laneId%16)*4);
                
//             asm volatile("ldmatrix.sync.aligned.x2.m8n8.shared.b16 {%0, %1}, [%2];\n" \
//                     : "=r"(reg_A[reg_store_idx][i][0]), "=r"(reg_A[reg_store_idx][i][1])
//                     : "r"(x_smem_casted));
//         }
                    

// #pragma unroll
//         for (int i=0; i<warp_rows_per_m; ++i)
//         {

//             uint32_t w_smem_casted = __cvta_generic_to_shared(w_smem +\
//                 smem_loader.xor_load_offset + \
//                 (wmma_idx.warp_x*wmma_idx.wx + i*WMMA_M)*8 + \
//                 k_stride*64 + \
//                 ((wmma_idx.laneId/16)*16 + wmma_idx.laneId%16)*4);
//                 //   (((wmma_idx.laneId/8) %2)*16 + wmma_idx.laneId%8)*4);
                
//             asm volatile("ldmatrix.sync.aligned.x2.m8n8.shared.b16 {%0, %1}, [%2];\n" \
//                     : "=r"(reg_B[reg_store_idx][i][0]), "=r"(reg_B[reg_store_idx][i][1])
//                     : "r"(w_smem_casted));
//         }

//         reg_store_idx ^= 1;
//         reg_load_idx ^= 1;





// #pragma unroll
//         for (int wy_tile=0; wy_tile<warp_cols_per_n; ++wy_tile)
//         { 
// #pragma unroll
//             for (int wx_tile=0; wx_tile<warp_rows_per_m; ++wx_tile)
//             {
//                 size_t j_s = (wy_tile % 2) ? (warp_rows_per_m - wx_tile - 1) : wx_tile;

//                 asm volatile(
//                     "mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 "
//                     "{%0,%1,%2,%3}, "    // D matrix
//                     "{%4,%5}, "                     // A matrix
//                     "{%6}, "                   // B matrix
//                     "{%7, %8, %9, %10};\n"
//                     : "=r"(O[j_s][wy_tile][0]), "=r"(O[j_s][wy_tile][1]), "=r"(O[j_s][wy_tile][2]) , "=r"(O[j_s][wy_tile][3]) 
//                     : "r"(reg_A[reg_load_idx][wy_tile][0]), "r"(reg_A[reg_load_idx][wy_tile][1]),
//                       "r"(reg_B[reg_load_idx][j_s][0]),
//                       "r"(O[j_s][wy_tile][0]), "r"(O[j_s][wy_tile][1]), "r"(O[j_s][wy_tile][2]) , "r"(O[j_s][wy_tile][3]));
//                 asm volatile(
//                     "mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 "
//                     "{%0,%1,%2,%3}, "    // D matrix
//                     "{%4,%5}, "                     // A matrix
//                     "{%6}, "                   // B matrix
//                     "{%7, %8, %9, %10};\n"
//                     : "=r"(O[j_s][wy_tile][4]), "=r"(O[j_s][wy_tile][5]), "=r"(O[j_s][wy_tile][6]) , "=r"(O[j_s][wy_tile][7]) 
//                     : "r"(reg_A[reg_load_idx][wy_tile][0]), "r"(reg_A[reg_load_idx][wy_tile][1]),
//                       "r"(reg_B[reg_load_idx][j_s][1]),
//                       "r"(O[j_s][wy_tile][4]), "r"(O[j_s][wy_tile][5]), "r"(O[j_s][wy_tile][6]) , "r"(O[j_s][wy_tile][7]));


//                 // asm volatile(
//                 //     "mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 "
//                 //     "{%0,%1,%2,%3}, "    // D matrix
//                 //     "{%4,%5}, "                     // A matrix
//                 //     "{%6}, "                   // B matrix
//                 //     "{%7, %8, %9, %10};\n"
//                 //     : "=r"(O[(j_s*warp_cols_per_n + wy_tile)*8]), "=r"(O[(j_s*warp_cols_per_n + wy_tile)*8+1]), "=r"(O[(j_s*warp_cols_per_n + wy_tile)*8+2]) , "=r"(O[+(j_s*warp_cols_per_n + wy_tile)*8+3]) 
//                 //     : "r"(reg_A[reg_load_idx][wy_tile][0]), "r"(reg_A[reg_load_idx][wy_tile][1]),
//                 //       "r"(reg_B[reg_load_idx][j_s][0]),
//                 //       "r"(O[(j_s*warp_cols_per_n + wy_tile)*8]), "r"(O[(j_s*warp_cols_per_n + wy_tile)*8+1]), "r"(O[(j_s*warp_cols_per_n + wy_tile)*8+2]) , "r"(O[+(j_s*warp_cols_per_n + wy_tile)*8+3]) );
//                 // asm volatile(
//                 //     "mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 "
//                 //     "{%0,%1,%2,%3}, "    // D matrix
//                 //     "{%4,%5}, "                     // A matrix
//                 //     "{%6}, "                   // B matrix
//                 //     "{%7, %8, %9, %10};\n"
//                 //     : "=r"(O[(j_s*warp_cols_per_n + wy_tile)*8+4]), "=r"(O[(j_s*warp_cols_per_n + wy_tile)*8+5]), "=r"(O[(j_s*warp_cols_per_n + wy_tile)*8+6]) , "=r"(O[+(j_s*warp_cols_per_n + wy_tile)*8+7]) 
//                 //     : "r"(reg_A[reg_load_idx][wy_tile][0]), "r"(reg_A[reg_load_idx][wy_tile][1]),
//                 //       "r"(reg_B[reg_load_idx][j_s][1]),
//                 //       "r"(O[(j_s*warp_cols_per_n + wy_tile)*8+4]), "r"(O[(j_s*warp_cols_per_n + wy_tile)*8+5]), "r"(O[(j_s*warp_cols_per_n + wy_tile)*8+6]) , "r"(O[+(j_s*warp_cols_per_n + wy_tile)*8+7]));

//             }
//         }
//     }

// #pragma unroll
//         for (int wy_tile=0; wy_tile<warp_cols_per_n; ++wy_tile)
// #pragma unroll
//             for (int wx_tile=0; wx_tile<warp_rows_per_m; ++wx_tile)
//                 for (int i=0;i<8;++i)
//                     out[(wx_tile*warp_cols_per_n + wy_tile)*8 + i] = O[wx_tile][wy_tile][i];
// }
