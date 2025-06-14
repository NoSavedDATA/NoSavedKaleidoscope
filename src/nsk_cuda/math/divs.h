#pragma once

__host__ __device__ __forceinline__ constexpr size_t div_ceil(size_t a, size_t b) {
    return (a + b - 1) / b;
}