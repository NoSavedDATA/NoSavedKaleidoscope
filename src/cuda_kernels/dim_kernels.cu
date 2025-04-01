#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <mma.h>

#include "atomic_kernels.cu"
#include "handles.h"



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

__global__ void broadcast_lastdim_add(float *y, const float *x,
                            const float *w, int dims_prod, int C) {

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int w_id = tid / C;

    if (tid >= dims_prod)
      return;



    y[tid] = x[tid] + w[w_id];
}

__global__ void sum_over_last_dim_kernel(const float *tensor,
                           float *summed,
                           int dims_prod, int summed_dim_size) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    int C = summed_dim_size;
    
    if (i < dims_prod) {
        int b = i / (C); // b updates only when v reaches it's maximum value
        

        float *summed_b = summed + b;

        float ix = tensor[i];

        atomicAdd(summed_b, ix);        
    }
}





__global__ void mean_over_semilast_dim_kernel(const float *x, float *y, const int dims_prod, const int T, const int C, const int warps_per_block)
{
  int tid = threadIdx.x;
  int b = blockIdx.x;

  if (b>=dims_prod)
    return;

  int warpId = tid / warpSize;
  int laneId = tid % warpSize;

  for (int warp_tile=0; warp_tile<ceilf(C/(float)warps_per_block); ++warp_tile)
  {
    int c = warp_tile*warps_per_block + warpId;
    __syncwarp();
    float sumval=0.0f;
    if (c<C)
    {
      for (int lane_tile=laneId; lane_tile<T; lane_tile+=warpSize)
        sumval += x[b*T*C + lane_tile*T + c];

      float mask_sumval;
      for(int mask=warpSize/2; mask>0; mask>>=1)
      {
        __syncwarp();
        mask_sumval = __shfl_down_sync(0xFFFFFFFF, sumval, mask);
        sumval+=mask_sumval;
      }
      sumval = __shfl_sync(0xFFFFFFFF, sumval, 0);

      y[b*C + c] = sumval/T;
    }
  }
}



__global__ void mean_over_semilast_dim_backward_kernel(float *dx, const float *dy, const int dims_prod, const int T, const int C)
{
  int idx = blockIdx.x*blockDim.x + threadIdx.x;

  if(idx>=dims_prod)
    return;



  int b = idx / (T*C);
  int t = (idx/C) % T;
  int c = idx % C;


  dx[b*T*C + t*C + c] = dy[b*C + c];
}


__global__ void sum_single_dim_kernel(const float *tensor,
                           float *summed,
                           int dims_prod) {

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  
  int C = dims_prod;
  
  if (i < dims_prod) {
    int b = i / (C); // b updates only when v reaches it's maximum value
    int v = i % C;


    float ix = tensor[i];

    atomicAdd(summed, ix);
  }
}


__global__ void sum_over_semilast_dim_kernel(const float *tensor,
                           float *summed,
                           int dims_prod, int last_dim_size, int summed_dim_size) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    int C = last_dim_size;
    int D = summed_dim_size*last_dim_size;
    
    if (i < dims_prod) {
        int b = i / C; // b updates only when v reaches it's maximum value
        int d = i / D;
        int v = i % C;
        // i = b*C + v

        float *summed_b = summed + v + d*C;

        float ix = tensor[i];

        atomicAdd(summed_b, ix);        
    }
}



__global__ void prod_single_dim_kernel(const float *tensor,
                           float *summed,
                           int dims_prod) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    int C = dims_prod;
    
    if (i < dims_prod) {
        int b = i / (C); // b updates only when v reaches it's maximum value
        int v = i % C;
        // i = b*C + v


        float ix = tensor[i];

        atomicMul(summed, ix);        
    }
}

__global__ void prod_over_last_dim_kernel(const float *tensor,
                           float *summed,
                           int dims_prod, int summed_dim_size) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    int C = summed_dim_size;
    
    if (i < dims_prod) {
        int b = i / (C); // b updates only when v reaches it's maximum value
        int v = i % C;
        // i = b*C + v

        float *summed_b = summed + b;

        float ix = tensor[i];

        atomicMul(summed_b, ix);        
    }
}

__global__ void prod_over_semilast_dim_kernel(const float *tensor,
                           float *summed,
                           int dims_prod, int last_dim_size, int summed_dim_size) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    int C = last_dim_size;
    int D = summed_dim_size*last_dim_size;
    
    if (i < dims_prod) {
        int b = i / C; // b updates only when v reaches it's maximum value
        int d = i / D;
        int v = i % C;
        // i = b*C + v

        float *summed_b = summed + v + d*C;

        float ix = tensor[i];

        atomicMul(summed_b, ix);        
    }
}