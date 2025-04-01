__inline__ __global__ void vec_mult(const float a, float* x, float* y, int dims_prod) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < dims_prod) {
    y[idx] = x[idx] * a;
  }
}
__inline__ __global__ void vec_div(const float a, float* x, float* y, int dims_prod) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < dims_prod) {
    y[idx] = x[idx] / a;
  }
}
__inline__ __global__ void vec_reverse_div(const float a, float* x, float* y, int dims_prod) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < dims_prod) {
    y[idx] = a / x[idx];
  }
}
__inline__ __global__ void vec_add(const float a, float* x, float* y, int dims_prod) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < dims_prod) {
    y[idx] = x[idx] + a;
  }
}
__inline__ __global__ void vec_sub(const float a, float* x, float* y, int dims_prod) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < dims_prod) {
    y[idx] = x[idx] - a;
  }
}
__inline__ __global__ void vec_equal(const float a, float* x, float* y, int dims_prod) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < dims_prod) {
    y[idx] = (x[idx]==a) ? 1.0f : 0.0f;
  }
}
__inline__ __global__ void vec_diff(const float a, float* x, float* y, int dims_prod) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < dims_prod) {
    y[idx] = (x[idx]!=a) ? 1.0f : 0.0f;
  }
}
__inline__ __global__ void vec_higher(const float a, float* x, float* y, int dims_prod) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < dims_prod) {
    y[idx] = (x[idx]>a) ? 1.0f : 0.0f;
  }
}
__inline__ __global__ void vec_higher_eq(const float a, float* x, float* y, int dims_prod) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < dims_prod) {
    y[idx] = (x[idx]>=a) ? 1.0f : 0.0f;
  }
}
__inline__ __global__ void vec_minor(const float a, float* x, float* y, int dims_prod) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < dims_prod) {
    y[idx] = (x[idx]<a) ? 1.0f : 0.0f;
  }
}
__inline__ __global__ void vec_minor_eq(const float a, float* x, float* y, int dims_prod) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < dims_prod) {
    y[idx] = (x[idx]<=a) ? 1.0f : 0.0f;
  }
}

