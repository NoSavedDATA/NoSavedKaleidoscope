#include "../include.h"

#include "wmma_blocking.h"
#include "cpp_call.h"

template<int WMMA_T, int WX, int WY>
void launch_kernel(Grid2 grid, const float* x, const float* w, float* o,
                   int B, int C, int OC, cudaStream_t stream) {
    wmma_blocking<WMMA_T, WX, WY, 32><<<grid.g, grid.w, grid.smem, stream>>>(
        x, w, o, B, C, OC, grid.b.x, grid.b.y, grid.wx, grid.wy,
        grid.bx_per_w, grid.by_per_w,
        grid.bx_per_wx);
}




// template void blocking_mma<16>(const float*, const float*, float*, int, int, int, cudaStream_t);