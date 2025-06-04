#include "../include.h"

#include "wmma_blocking_i8.h"
#include "cpp_call.h"

template<int WMMA_T, int WX, int WY>
void launch_kernel_i8(Wmma_Grid grid, const int8_t* x, const int8_t* w, float* o,
                   int M, int N, int K, cudaStream_t stream) {
    wmma_blocking_i8<WMMA_T, WX, WY, 128><<<grid.g, grid.w, grid.smem, stream>>>( // 128 because float has 4B and int8 has 1B
        x, w, o, M, N, K, grid.bx, grid.by, grid.wx, grid.wy,
        grid.bx_per_w, grid.by_per_w,
        grid.bx_per_wx);
}



