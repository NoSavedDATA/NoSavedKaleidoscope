#pragma once




__global__ void quantize_f32_i4_kernel(int8_t *x8, const float *x, float *scale, const float fraction, const int lower, const int upper, const int M, const int N, const int, const int, const int num_warps, const int dims_prod);
