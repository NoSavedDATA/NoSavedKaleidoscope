#pragma once



#define MANTISSA_BITS 3
#define EXP_BITS 4
#define SIGN_BIT 1
#define MAX_EXP ((1 << EXP_BITS) - 1)
#define MAX_MANT ((1 << MANTISSA_BITS) - 1)


__device__ inline int8_t quantize_float_to_int8(float x) {
    if (x == 0.0f) return 0;

    int sign = x < 0.0f;
    float abs_x = fabsf(x);
    int exp = __float2int_rn(log2f(abs_x));
    exp = max(0, min(exp, MAX_EXP));
    float scaled = abs_x / powf(2.0f, exp);
    int mant = __float2int_rn(scaled * MAX_MANT);
    mant = min(mant, MAX_MANT);

    return (int8_t)((sign << 7) | (exp << MANTISSA_BITS) | mant);
}



__device__ inline float dequantize_int8_to_float(int8_t q) {
    int uq = (uint8_t)q;
    int sign = (uq >> 7) & 0x1;
    int exp = (uq >> MANTISSA_BITS) & MAX_EXP;
    int mant = uq & MAX_MANT;
    float val = ((float)mant / MAX_MANT) * powf(2.0f, exp);
    return sign ? -val : val;
}




__device__ inline int8_t quantize_scaled_float(float x, float scale) {
    // Apply scale first
    float scaled = x * scale;

    // Clamp to avoid overflow
    if (scaled == 0.0f) return 0;
    int sign = scaled < 0.0f;
    float abs_val = fabsf(scaled);

    // Compute exponent
    int exp = __float2int_rn(log2f(abs_val));
    exp = max(0, min(exp, MAX_EXP));

    // Compute mantissa
    float norm = abs_val / powf(2.0f, exp);
    int mant = __float2int_rn(norm * MAX_MANT);
    mant = min(mant, MAX_MANT);

    // Pack into int8
    return (int8_t)((sign << 7) | (exp << MANTISSA_BITS) | mant);
}


__device__ inline float dequantize_scaled_int8(int8_t q, float scale) {
    int uq = (uint8_t)q;
    int sign = (uq >> 7) & 0x1;
    int exp = (uq >> MANTISSA_BITS) & MAX_EXP;
    int mant = uq & MAX_MANT;

    float val = ((float)mant / MAX_MANT) * powf(2.0f, exp);
    val = sign ? -val : val;
    return val / scale;
}
