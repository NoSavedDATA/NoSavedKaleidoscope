#pragma once


template<int warp_rows_per_m, int warp_cols_per_n>
struct wmma_indexes {

  int block_x, block_y, bx_per_w, by_per_w, bx_per_wx, warpId, laneId, warp_x, warp_y, mw, ml, tid, blocking_size_x, blocking_size_y, wx, wy, wk;
  int by_warp_offset, bx_warp_offset;
  __device__ wmma_indexes(int bx_per_w, int by_per_w, int bx_per_wx, int bx, int by, int wx, int wy, int wk)
             : bx_per_w(bx_per_w), by_per_w(by_per_w), bx_per_wx(bx_per_wx), block_x(block_x), blocking_size_x(bx), blocking_size_y(by),
               wx(wx), wy(wy), wk(wk) {

    block_x = blockIdx.x;
    block_y = blockIdx.y;
    tid = threadIdx.x;

    warpId = threadIdx.x / warpSize;
    laneId = (threadIdx.x) % warpSize;

    warp_y = warpId / bx_per_wx;
    warp_x = warpId % bx_per_wx;
    
    by_warp_offset = by_per_w*warpId;
    bx_warp_offset = bx_per_w*warpId;

    mw = laneId / 4;
    ml = laneId % 4;

  }

};