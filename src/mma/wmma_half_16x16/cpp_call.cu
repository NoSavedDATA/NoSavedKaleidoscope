#include "../include.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>  // if using bfloat16

#include "wmma_blocking.h"
#include "cpp_call.h"

template<int WMMA_T, int WX, int WY>
void launch_kernel(Wmma_Grid grid, const float* x, const float* w, float* o,
                   int B, int C, int OC, cudaStream_t stream) {
    wmma_blocking<WMMA_T, WX, WY, 32><<<grid.g, grid.w, grid.smem, stream>>>(
        x, w, o, B, C, OC, grid.bx, grid.by, grid.wx, grid.wy,
        grid.bx_per_w, grid.by_per_w,
        grid.bx_per_wx);
}



