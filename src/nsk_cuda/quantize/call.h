#pragma once





void quantize_f32_to_i8(int8_t *x8, float *x, float quant, int M, int N, cudaStream_t stream);