#pragma once



__global__ void dropout_mask_kernel(float *y, float *m, const float *x, float rate, float scale,
                               int dims_prod,
                               unsigned long long seed); 

__global__ void dropout_backward_kernel(float *dx, float *m, const float *dy,
                               int dims_prod); 