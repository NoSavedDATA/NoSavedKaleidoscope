#pragma once


template<int warp_rows_per_m, int warp_cols_per_n>
struct wmma_indexes {

  int block_x, block_y, bx_per_w, by_per_w, bx_per_wx, warpId, laneId, warp_x, warp_y, mw, ml, tid, blocking_size_x, blocking_size_y, wx, wy, wk, cp_rows, cp_cols, num_warps;
  int by_warp_offset, bx_warp_offset;
  __device__ wmma_indexes(int bx_per_w, int by_per_w, int bx_per_wx, int bx, int by, int wx, int wy, int wk, int cp_cols=8, int num_warps=8)
             : bx_per_w(bx_per_w), by_per_w(by_per_w), bx_per_wx(bx_per_wx), block_x(block_x), blocking_size_x(bx), blocking_size_y(by),
               wx(wx), wy(wy), wk(wk), cp_rows(cp_rows), num_warps(num_warps) {

    block_x = blockIdx.x;
    block_y = blockIdx.y;
    tid = threadIdx.x;

    warpId = threadIdx.x / warpSize;
    laneId = (threadIdx.x) % warpSize;

    warp_y = warpId / 4; // BLOCK_ROW_WARPS: 2
    warp_x = warpId % 4; // BLOCK_COL_WARPS: 4
    
    by_warp_offset = by_per_w*warpId;
    bx_warp_offset = bx_per_w*warpId;

    cp_rows = 32/cp_cols;

    mw = laneId / cp_rows;
    ml = laneId % cp_rows;


  }

};