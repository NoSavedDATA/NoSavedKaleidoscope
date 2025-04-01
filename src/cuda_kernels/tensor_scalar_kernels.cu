

__global__ void scalarmult_backward_kernel(float *dx, const float *dy,
                                           const float scalar,
                                           int dims_prod) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < dims_prod)
    {
      dx[i] = scalar * dy[i];
    }
}