#include "../include.h"

#include "wmma_blocking_i8_dx.h"
#include "wmma_blocking_i8_dx.h"
#include "cpp_call.h"


template<int WMMA_T, int WX, int WY>
void launch_i8dw_kernel(Wmma_Grid grid, const int8_t* x, const int8_t* w, float* o,
                   int M, int N, int K, cudaStream_t stream) {
    wmma_blocking_i8_dw<WMMA_T, WX, WY, 32><<<grid.g, grid.w, grid.smem, stream>>>(
        x, w, o, M, N, K, grid.bx, grid.by, grid.wx, grid.wy,
        grid.bx_per_w, grid.by_per_w,
        grid.bx_per_wx);
}

template<int WMMA_T, int WX, int WY>
void launch_i8dx_kernel(Wmma_Grid grid, const int8_t* x, const int8_t* w, float* o,
                   int M, int N, int K, cudaStream_t stream) {
    wmma_blocking_i8_dx<WMMA_T, WX, WY, 32><<<grid.g, grid.w, grid.smem, stream>>>(
        x, w, o, M, N, K, grid.bx, grid.by, grid.wx, grid.wy,
        grid.bx_per_w, grid.by_per_w,
        grid.bx_per_wx);
}



