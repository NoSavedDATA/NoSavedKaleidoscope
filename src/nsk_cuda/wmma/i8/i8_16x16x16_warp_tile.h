#pragma once

#include <mma.h>

#include "../../structs/i8_wmma_frags.h"
#include "ptx.h"

using namespace nvcuda;



template<int warp_rows_per_m, int warp_cols_per_n>
__device__ void warp_tiled_wmma_i8_16x16x16(i8_wmma_frags<warp_rows_per_m, warp_cols_per_n, int8_t> &frag_loader,
                                              wmma_indexes<warp_rows_per_m, warp_cols_per_n>& wmma_idx,
                                              const int M, const int N, const int WMMA_M, const int WMMA_N) {
  

    for (int wx_tile=0; wx_tile<warp_rows_per_m; ++wx_tile)
    { 
        for (int wy_tile=0; wy_tile<warp_cols_per_n; ++wy_tile)
        {
            if ((wmma_idx.block_y*wmma_idx.blocking_size_y + wy_tile*WMMA_M)<M && (wmma_idx.block_x*wmma_idx.blocking_size_x + wx_tile*WMMA_N)<N)
            wmma16x16x16_i8(frag_loader.acc_frag+(wx_tile*warp_cols_per_n + wy_tile)*8, frag_loader.x_frag[wy_tile], frag_loader.w_frag[wx_tile]); // 8 is the frag ld  
        }
    }
    __syncthreads();
}

