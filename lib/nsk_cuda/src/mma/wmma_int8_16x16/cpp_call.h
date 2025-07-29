#pragma once

#include <iostream>

#include "../util.h"

#include "wmma_blocking_i8.h"



template<int WMMA_T, int WX, int WY>
void launch_kernel_i8(Wmma_Grid grid, const int8_t* x, const int8_t* w, float* o, const float *, const float *,
                   int B, int C, int OC, cudaStream_t stream); 


template<int WMMA_T>
void blocking_mma_i8(const int8_t *x, const int8_t *w, float *o, const float *scale_M, const float *scale_N, int M, int N, int K, cudaStream_t stream)
{

  Wmma_Grid grid = CalculateBlockingSize(N, M,
                                         8,
                                         256, 128,
                                         64, 64,
                                         16, 16);

  // Wmma_Grid grid = CalculateBlockingSize(N, M,
  //                                        8,
  //                                        128, 64,
  //                                        32, 32,
  //                                        16, 16);

                                         
  // std::cout << "OC: " << N << ", B: " << M << \
  //   "\ngx: " << grid.g.x << ", gy: " << grid.g.y <<  \
  //   "\nblocking warps per block x: " << grid.bx_per_w << ", y: " << grid.by_per_w << \
  //   "\nx warps: " << grid.w.x/32 << ", y warps: " << grid.w.y << \
  //   "\nwx_per_wmma_m " << grid.wx_per_wmma_m << " wy_per_wmma_n " << grid.wy_per_wmma_n << "\n\n";


  using LaunchFn = void(*)(Wmma_Grid, const int8_t*, const int8_t *, float*, const float *, const float *, int, int, int, cudaStream_t);
  static constexpr LaunchFn dispatch_table[5][5] = {
      {nullptr}, // 0 is unused
      {nullptr, launch_kernel_i8<WMMA_T,1,1>, launch_kernel_i8<WMMA_T,1,2>, launch_kernel_i8<WMMA_T,1,3>, launch_kernel_i8<WMMA_T,1,4>},
      {nullptr, launch_kernel_i8<WMMA_T,2,1>, launch_kernel_i8<WMMA_T,2,2>, launch_kernel_i8<WMMA_T,2,3>, launch_kernel_i8<WMMA_T,2,4>},
      {nullptr, launch_kernel_i8<WMMA_T,3,1>, launch_kernel_i8<WMMA_T,3,2>, launch_kernel_i8<WMMA_T,3,3>, launch_kernel_i8<WMMA_T,3,4>},
      {nullptr, launch_kernel_i8<WMMA_T,4,1>, launch_kernel_i8<WMMA_T,4,2>, launch_kernel_i8<WMMA_T,4,3>, launch_kernel_i8<WMMA_T,4,4>},
  };


  auto launcher = dispatch_table[grid.wx_per_wmma_m][grid.wy_per_wmma_n];
  launcher(grid, x, w, o, scale_M, scale_N, M, N, K, stream);
}



template void blocking_mma_i8<16>(const int8_t*, const int8_t*, float*, const float *, const float *, int, int, int, cudaStream_t);