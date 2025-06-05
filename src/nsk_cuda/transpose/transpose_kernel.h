#pragma once


__global__ void transpose_kernel(float *y, const float *x, const int M, const int N, const int smem_N, const int);