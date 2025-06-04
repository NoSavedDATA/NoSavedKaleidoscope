#pragma once

#include <iostream>

#include "../../util.h"




template<int WMMA_T, int WX, int WY>
void launch_i8dw_kernel(Wmma_Grid grid, const int8_t* x, const int8_t* w, float* o,
                   int, int, int, cudaStream_t stream); 

template<int WMMA_T, int WX, int WY>
void launch_i8dx_kernel(Wmma_Grid grid, const int8_t* x, const int8_t* w, float* o,
                   int, int, int, cudaStream_t stream); 


template<int WMMA_T>
void blocking_mma_i8_dw(const int8_t *x, const int8_t *w, float *o, int M, int N, int K, cudaStream_t stream)
{
  Wmma_Grid grid = CalculateBlockingSize(N, M,
                                         8,
                                         128, 64,
                                         32, 32,
                                         16, 16);

  using LaunchFn = void(*)(Wmma_Grid, const int8_t*, const int8_t*, float*, int, int, int, cudaStream_t);
  static constexpr LaunchFn dispatch_table[5][5] = {
      {nullptr}, // 0 is unused
      {nullptr, launch_i8dw_kernel<WMMA_T,1,1>, launch_i8dw_kernel<WMMA_T,1,2>, launch_i8dw_kernel<WMMA_T,1,3>, launch_i8dw_kernel<WMMA_T,1,4>},
      {nullptr, launch_i8dw_kernel<WMMA_T,2,1>, launch_i8dw_kernel<WMMA_T,2,2>, launch_i8dw_kernel<WMMA_T,2,3>, launch_i8dw_kernel<WMMA_T,2,4>},
      {nullptr, launch_i8dw_kernel<WMMA_T,3,1>, launch_i8dw_kernel<WMMA_T,3,2>, launch_i8dw_kernel<WMMA_T,3,3>, launch_i8dw_kernel<WMMA_T,3,4>},
      {nullptr, launch_i8dw_kernel<WMMA_T,4,1>, launch_i8dw_kernel<WMMA_T,4,2>, launch_i8dw_kernel<WMMA_T,4,3>, launch_i8dw_kernel<WMMA_T,4,4>},
  };


  auto launcher = dispatch_table[grid.wx_per_wmma_m][grid.wy_per_wmma_n];
  launcher(grid, x, w, o, M, N, K, stream);
}

template<int WMMA_T>
void blocking_mma_i8_dx(const int8_t *x, const int8_t *w, float *o, int M, int N, int K, cudaStream_t stream)
{
  Wmma_Grid grid = CalculateBlockingSize(N, M,
                                         8,
                                         128, 64,
                                         32, 32,
                                         16, 16);

  using LaunchFn = void(*)(Wmma_Grid, const int8_t*, const int8_t*, float*, int, int, int, cudaStream_t);
  static constexpr LaunchFn dispatch_table[5][5] = {
      {nullptr}, // 0 is unused
      {nullptr, launch_i8dx_kernel<WMMA_T,1,1>, launch_i8dx_kernel<WMMA_T,1,2>, launch_i8dx_kernel<WMMA_T,1,3>, launch_i8dx_kernel<WMMA_T,1,4>},
      {nullptr, launch_i8dx_kernel<WMMA_T,2,1>, launch_i8dx_kernel<WMMA_T,2,2>, launch_i8dx_kernel<WMMA_T,2,3>, launch_i8dx_kernel<WMMA_T,2,4>},
      {nullptr, launch_i8dx_kernel<WMMA_T,3,1>, launch_i8dx_kernel<WMMA_T,3,2>, launch_i8dx_kernel<WMMA_T,3,3>, launch_i8dx_kernel<WMMA_T,3,4>},
      {nullptr, launch_i8dx_kernel<WMMA_T,4,1>, launch_i8dx_kernel<WMMA_T,4,2>, launch_i8dx_kernel<WMMA_T,4,3>, launch_i8dx_kernel<WMMA_T,4,4>},
  };


  auto launcher = dispatch_table[grid.wx_per_wmma_m][grid.wy_per_wmma_n];
  launcher(grid, x, w, o, M, N, K, stream);
}


template void blocking_mma_i8_dw<16>(const int8_t*, const int8_t*, float*, int, int, int, cudaStream_t);
template void blocking_mma_i8_dx<16>(const int8_t*, const int8_t*, float*, int, int, int, cudaStream_t);