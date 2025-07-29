#pragma once



void transpose_tensor(float *y, const float *x, const int M, const int N, cudaStream_t);