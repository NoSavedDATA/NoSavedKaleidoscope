#pragma once


#include "transpose_kernel.h"


void transpose_tensor(float *y, const float *x, const int M, const int N, cudaStream_t stream)
{
    int num_warps = 8;

    dim3 grid_size(std::floor((M + (num_warps - 1)) / (float)(num_warps)));
    
    int smem = num_warps*sizeof(float)*128*2; // 128 from 32*4 loads of 1 warp of cp.async, 2 from xor smem


    transpose_kernel<<<grid_size, num_warps*32, smem, stream>>>(y, x, M, N, 128, num_warps);
}