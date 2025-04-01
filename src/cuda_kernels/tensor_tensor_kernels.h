#include <cuda_runtime.h>


__global__ void add_forward(float *y, const float *x,
                            const float *w, int dims_prod);
                             
__global__ void add_inplace(float *y, const float *x,
                                    int dims_prod); 

__global__ void sub_forward(float *y, const float *x,
                            const float *w, int dims_prod); 


__global__ void equal_forward(float *y, const float *x,
                            const float *w, int dims_prod); 

__global__ void hadamard_kernel(float *y, const float *x,
                            const float *w, int dims_prod); 