#pragma once

#include <mma.h>

#include "../../structs/i8_wmma_frags.h"
#include "ptx.h"

using namespace nvcuda;



// template<int warp_rows_per_m, int warp_cols_per_n>
// __device__ inline void warp_tiled_wmma_i8_16x16x16(i8_wmma_frags<warp_rows_per_m, warp_cols_per_n, int8_t> &frag_loader,
//                                               wmma_indexes<warp_rows_per_m, warp_cols_per_n>& wmma_idx,
//                                               const int M, const int N, const int WMMA_M, const int WMMA_N) {
  

//     for (int wx_tile=0; wx_tile<warp_rows_per_m; ++wx_tile)
//     { 
//         for (int wy_tile=0; wy_tile<warp_cols_per_n; ++wy_tile)
//         {
//             if ((wmma_idx.block_y*wmma_idx.blocking_size_y + wy_tile*WMMA_M)<M && (wmma_idx.block_x*wmma_idx.blocking_size_x + wx_tile*WMMA_N)<N)
//             wmma16x16x16_i8(frag_loader.acc_frag+(wx_tile*warp_cols_per_n + wy_tile)*8, frag_loader.x_frag+wy_tile*8, frag_loader.w_frag+wx_tile*8); // 8 is the frag ld  
//         }
//     }
//     __syncthreads();
// }






// template<int warp_rows_per_m, int warp_cols_per_n, typename T>
// __device__ inline void wmma_i8_m16n16k16(i8_wmma_frags<warp_rows_per_m, warp_cols_per_n, int8_t> &frag_loader,
//                                         wmma_indexes<warp_rows_per_m, warp_cols_per_n>& wmma_idx,
//                                         smem_cpasync_wmma_loader<warp_rows_per_m, warp_cols_per_n, T>& smem_loader,
//                                         float *x_smem, float *w_smem,
//                                         const int M, const int N, const int K,
//                                         const int WMMA_M, const int WMMA_N,
//                                         const int chunks)
// {

//     for (int k_stride=0; k_stride<chunks; ++k_stride)
//     {
//         smem_loader.store_frag_A(frag_loader, x_smem, WMMA_M, k_stride);
//         smem_loader.store_frag_B(frag_loader, w_smem, WMMA_N, k_stride);

//         warp_tiled_wmma_i8_16x16x16(frag_loader, wmma_idx, M, N, WMMA_M, WMMA_N);
//     }
// }
