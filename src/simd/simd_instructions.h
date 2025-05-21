#pragma once



void simd_fill_float(float* dst, float value, int n);
void simd_fill_float_aligned(float* dst, float value, int n);

void simd_copy_ints(int* dst, const int* src, size_t count);

void simd_copy_floats(float* dst, const float* src, size_t count);

void simd_copy_floats_aligned(float* dst, const float* src, size_t count);
