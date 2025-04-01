#pragma once


__global__ void mse_kernel(float *dy, const float* y_hat, const float* y,
                            const float scale, const float dims_prod);

                            
__global__ void online_mse(float *out, const float *y_hat, const float *y_true, int N, int C); 


__global__ void mse_with_priorities_kernel(float *dy, const float* y_hat, const float* y, const float *is_w,
    const float scale, const float dims_prod); 