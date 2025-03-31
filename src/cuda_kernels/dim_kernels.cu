#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <mma.h>



__global__ void repeat_interleave_kernel_last_dim(const float *tensor,
                           float *probs,
                           int B, int C) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    //int i = threadIdx.x;
    
    if (i < B * C) {
        int b = i / (C);
        int v = i % C;

        float *probs_b = probs + b * C;
        float ix = tensor[b];

        probs_b[v] = ix;
    }
}


__global__ void idx_last_dim_kernel(float *tgt,
                           const float *tensor, const float *idx_tensor, 
                           int dims_prod, int last_dim_size) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int C = last_dim_size;
    
    if (i < dims_prod) {
        int b = i / C;
        int v = i % C;
        // i = b * C + v

        float *tgt_b = tgt + b;
        float idx_b = idx_tensor[b];

        if (v==idx_b)
        {
          float ix = tensor[i];
          tgt[b] = ix;
        }
    }
}


__global__ void idx_attr_semi_last_dim_kernel(float *tgt,
                           const float *tensor, const float *idx_tensor, 
                           int dims_prod, int last_dim_size) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int C = last_dim_size;
    
    if (i < dims_prod) {
        int b = i / C;
        int v = i % C;
        // i = b * C + v

        float *tgt_b = tgt + b;
        float idx_b = idx_tensor[b];

        if (v==idx_b)
        {
          float ix = tensor[b];
          tgt[i] = ix;
        }
    }
}


__global__ void idx_attr_simple_single_dim_kernel(float *tensor, const float *idx, const float *x, const int dims_prod)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  if (tid>=dims_prod)
    return; 

  tensor[(int)idx[tid]] = x[tid];
}