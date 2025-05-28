#pragma once

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <mma.h>



struct Grid {

  dim3 g;
  dim3 b;
  dim3 w;


  int smem;

  int wx_per_bx;
  int wy_per_by;

  void NewGrid(int gx, int gy, int bx, int by);
  void SetWarpSize(int wx, int wy);
};


Grid CalculateBlockingSize(int M, int N);




struct Grid2 {

  
  dim3 g;
  dim3 b;
  dim3 w;


  int smem;

  int bx_per_w;
  int by_per_w;

  int wx, wy, bx_per_wx, by_per_wy, wx_per_wmma_m, wy_per_wmma_n;

  void NewGrid(int gx, int gy, int bx, int by);

  void SetWarpSize(int warps, int wx, int wy);
};


Grid2 CalculateBlockingSize2(int M, int N);





