
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>  // if using bfloat16

#include "wmma_blocking_i4.h"
#include "cpp_call.h"

template<int WMMA_T, int WX, int WY>
void launch_kernel_i4(Wmma_Grid grid, const int8_t* x, const int8_t* w, float* o, const float *scale_M, const float *scale_N,
                   int M, int N, int K, cudaStream_t stream) {

    wmma_blocking_i4_mma<WMMA_T, WX, WY, 128><<<grid.g, grid.w, grid.smem, stream>>>( // 128 because float has 4B and int8 has 1B
        x, w, o, scale_M, scale_N, M, N, K, grid.warps, grid.bx, grid.by, grid.wx, grid.wy,
        grid.bx_per_w, grid.by_per_w,
        grid.bx_per_wx);
}



