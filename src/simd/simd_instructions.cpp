#include <immintrin.h>

void simd_fill_float(float* dst, float value, int n) {
    __m256 vec_val = _mm256_set1_ps(value);  // Broadcast PAD_TOK to all 8 lanes

    int i = 0;
    int simd_width = 8;

    // Bulk fill using SIMD
    for (; i <= n - simd_width; i += simd_width) {
        _mm256_storeu_ps(dst + i, vec_val);  // Use _mm256_store_ps if dst is 32-byte aligned
    }

    // Handle remaining elements
    for (; i < n; ++i)
        dst[i] = value;
}


void simd_fill_float_aligned(float* dst, float value, int n) {
    __m256 vec_val = _mm256_set1_ps(value);  // Broadcast PAD_TOK to all 8 lanes

    int i = 0;
    int simd_width = 8;

    // Bulk fill using SIMD
    for (; i <= n - simd_width; i += simd_width) {
        _mm256_store_ps(dst + i, vec_val);
    }

    // Handle remaining elements
    for (; i < n; ++i)
        dst[i] = value;
}



void simd_copy_ints(int* dst, const int* src, size_t count) {
    size_t i = 0;
    size_t simd_width = 8; // AVX2 = 8 x 32-bit ints

    // SIMD copy in chunks of 8 ints (256 bits)
    for (; i + simd_width <= count; i += simd_width) {
        __m256i vec = _mm256_loadu_si256((__m256i const*)(src + i));
        _mm256_storeu_si256((__m256i*)(dst + i), vec);
    }

    // Remaining scalar copy
    for (; i < count; ++i)
        dst[i] = src[i];
}



void simd_copy_floats(float* dst, const float* src, size_t count) {
    size_t i = 0;
    size_t simd_width = 8; // AVX2 processes 8 floats at a time

    // SIMD copy
    for (; i + simd_width <= count; i += simd_width) {
        __m256 vec = _mm256_loadu_ps(src + i);   // unaligned load
        _mm256_storeu_ps(dst + i, vec);          // unaligned store
    }

    // Tail copy (scalar)
    for (; i < count; ++i)
        dst[i] = src[i];
}


void simd_copy_floats_aligned(float* dst, const float* src, size_t count) {
    size_t i = 0;
    size_t simd_width = 8; // AVX2 processes 8 floats at a time

    // SIMD copy
    for (; i + simd_width <= count; i += simd_width) {
        __m256 vec = _mm256_load_ps(src + i);
        _mm256_store_ps(dst + i, vec);
    }

    // Tail copy (scalar)
    for (; i < count; ++i)
        dst[i] = src[i];
}
