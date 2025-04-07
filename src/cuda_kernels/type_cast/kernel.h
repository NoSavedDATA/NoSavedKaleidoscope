#pragma once


__global__ void float_to_half_kernel(const float *__restrict__ x, __half *__restrict__ y, const int dims_prod);


__global__ void half_to_float_kernel(const __half *__restrict__ x, float *__restrict__ y, const int dims_prod);
