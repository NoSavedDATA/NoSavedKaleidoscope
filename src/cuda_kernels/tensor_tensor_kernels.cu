#include <cuda_runtime.h>


__global__ void add_forward(float *y, const float *x,
                            const float *w, int dims_prod) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < dims_prod)
        y[i] = x[i] + w[i];
}
__global__ void add_inplace(float *y, const float *x,
                                    int dims_prod) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < dims_prod)
        y[i] = y[i] + x[i];
}

__global__ void sub_forward(float *y, const float *x,
                            const float *w, int dims_prod) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < dims_prod)
        y[i] = x[i] - w[i];
}


__global__ void equal_forward(float *y, const float *x,
                            const float *w, int dims_prod) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < dims_prod)
        y[i] = (x[i]==w[i]) ? 1.0f : 0.0f;
}

__global__ void hadamard_kernel(float *y, const float *x,
                            const float *w, int dims_prod) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < dims_prod)
        y[i] = x[i] * w[i];
}


__global__ void hadamard_backward_kernel(const float *x, const float *w,
                                         float *dx, float *dw, const float *dy,
                                         int dims_prod) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < dims_prod)
    {
      dx[i] = w[i] * dy[i];
      dw[i] = x[i] * dy[i];
    }
}