#pragma once

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <mma.h>






struct Wmma_Grid {

  
  dim3 g;
  dim3 w;


  int smem;

  int bx_per_w;
  int by_per_w;

  int wx, wy, bx, by, bx_per_wx, by_per_wy, wx_per_wmma_m, wy_per_wmma_n, warps;

  Wmma_Grid(int gx, int gy, int warps, int bx, int by, int wx, int wy, int wmma_m, int wmma_n);
};


Wmma_Grid CalculateBlockingSize(int M, int N,
                                int warps,
                                int block_size_x, int block_size_y,
                                int wx, int wy,
                                int wmma_m, int wmma_n);





