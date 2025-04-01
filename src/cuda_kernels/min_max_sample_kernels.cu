#pragma once


#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <curand_kernel.h>

// Parallelizes over B, C
__global__ void onehot_kernel(const float *tensor,
                           float *probs,
                           int B, int C) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    //int i = threadIdx.x;
    
    if (i < B * C) {
        int b = i / (C);
        int v = i % C;

        float *probs_b = probs + b * C;
        int ix = tensor[b];

        float indicator = (v==ix) ? 1.0f : 0.0f;
        probs_b[v] = indicator;
    }
}


__global__ void warped_to_probs_single_dim(float *y, const float *x, int C) {
  
    const int warpsPerBlock = blockDim.x / warpSize;
    int tid = threadIdx.x;

    

    int warpId = tid / warpSize;
    int laneId = tid % warpSize;
    // one warp one row
    //int row = blockIdx.x * warpsPerBlock + warpId;
    
    if (laneId >= C)
        return;


    
    float sumval = 0.0f;

#pragma unroll
    for (int i = laneId; i < C; i += warpSize)
        sumval += x[i];
    

    float offsetSumval;

#pragma unroll
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        __syncwarp();
        offsetSumval = __shfl_down_sync(0xFFFFFFFF, sumval, offset);
        
        sumval += offsetSumval;
    }


    sumval = __shfl_sync(0xFFFFFFFF, sumval, 0);


#pragma unroll
    for (int i = laneId; i < C; i += warpSize)
        y[i] = x[i] / sumval;
}


__global__ void sample_val_from_probs(float *tensor, float *sampled_value, int n, unsigned long long seed) {
    // Get the thread ID
    int idx = blockIdx.x * blockDim.x + threadIdx.x;


    // Only one thread needs to sample a value
    if (idx == 0) {
        // Setup random generator
        curandState state;
        curand_init(seed, idx, 0, &state);

        // Generate a random float in the range [0, 1)
        float rand_value = curand_uniform(&state);

        // Perform sampling
        float cumulative_sum = 0.0f;
        for (int i = 0; i < n; ++i) {
            cumulative_sum += tensor[i];
            if (rand_value < cumulative_sum) {
                sampled_value[0] = tensor[i];  // Sampled value
                return;
            }
        }
    }
}

__global__ void sample_from_probs(float *tensor, float *sampled_value, int n, unsigned long long seed) {
    // Get the thread ID
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    

    // Only one thread needs to sample a value
    if (idx == 0) {

        // Setup random generator
        curandState state;
        curand_init(seed, idx, 0, &state);

        // Generate a random float in the range [0, 1)
        float rand_value = curand_uniform(&state);

        // Perform sampling
        float cumulative_sum = 0.0f;
        for (int i = 0; i < n; ++i) {
            cumulative_sum += tensor[i];
            if (rand_value < cumulative_sum) {
                sampled_value[0] = i;  // Sampled value
                return;
            }
        }
    }
}


__global__ void warped_to_probs_single_dim_pow(float *y, const float *x, float alpha, int C) {
  
    const int warpsPerBlock = blockDim.x / warpSize;
    int tid = threadIdx.x;

    
    
    int warpId = tid / warpSize;
    int laneId = tid % warpSize;
    // one warp one row
    //int row = blockIdx.x * warpsPerBlock + warpId;
    
    if (laneId >= C)
        return;


    
    float sumval = 0.0f;

#pragma unroll
    for (int i = laneId; i < C; i += warpSize)
        sumval += pow(x[i], alpha);
    

    float offsetSumval;

#pragma unroll
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        __syncwarp();
        offsetSumval = __shfl_down_sync(0xFFFFFFFF, sumval, offset);
        
        sumval += offsetSumval;
    }


    sumval = __shfl_sync(0xFFFFFFFF, sumval, 0);


#pragma unroll
    for (int i = laneId; i < C; i += warpSize)
        y[i] = x[i] / sumval;
    
}


__global__ void is_w_kernel(float *is_w_ptr, const float *probs, const float *idx, float beta, float max_idx)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int lid = tid % warpSize;

  if (lid>=max_idx)
    return;

  float eps = 1e-6;

  float is_w = pow(1/(probs[(int)idx[0]]*max_idx + eps), beta);
  float iter_is_w;



  float max_is_w = -INFINITY;

#pragma unroll
  for (int i=lid; i<max_idx; i+=warpSize)
  {

    iter_is_w = pow(1/(probs[i]*max_idx + eps), beta);
    max_is_w = fmaxf(iter_is_w, max_is_w);
  }

  
  float warp_is_w;
#pragma unroll
  for (int mask=warpSize/2; mask > 0; mask>>=1)
  {
    __syncwarp();
    warp_is_w = __shfl_down_sync(0xFFFFFFFF, max_is_w, mask);
    max_is_w = fmaxf(max_is_w, warp_is_w);
  }
  max_is_w = __shfl_sync(0xFFFFFFFF, max_is_w, 0);


  is_w_ptr[0] = is_w / max_is_w;
}

__global__ void max_over_last_dim_kernel(const float *tensor,
                           float *maxed,
                           int dims_prod, int maxed_dim_size) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    int C = maxed_dim_size;
    
    if (i < dims_prod) {
        int b = i / C;
        int v = i % C;
        // i = b * C + v

        float *max_b = maxed + b;

        float ix = tensor[i];


        unsigned int *const addr_as_ui = (unsigned int *)max_b;
        unsigned int old = *addr_as_ui, assumed;
        do {
          assumed = old;
          if (__uint_as_float(assumed) >= ix) break;
          old = atomicCAS(addr_as_ui, assumed, __float_as_uint(ix));
        } while (assumed != old);
    }
}

__global__ void max_over_semilast_dim_kernel(const float *tensor,
                           float *maxed,
                           int dims_prod, int last_dim_size, int maxed_dim_size) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    int C = last_dim_size;
    int D = maxed_dim_size*last_dim_size;
    
    
    if (i < dims_prod) {
        int b = i / C;
        int d = i / D;
        int v = i % C;
        // i = b * C + v

        float *max_b = maxed + v + d*C;

        float ix = tensor[i];

        unsigned int *const addr_as_ui = (unsigned int *)max_b;
        unsigned int old = *addr_as_ui, assumed;
        do {
          assumed = old;
          if (__uint_as_float(assumed) >= ix) break;
          old = atomicCAS(addr_as_ui, assumed, __float_as_uint(ix));
        } while (assumed != old);
    }
}



__global__ void argmax_over_last_dim_kernel(const float *tensor,
                           float *maxed, float *argmaxed,
                           int dims_prod, int maxed_dim_size) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    int C = maxed_dim_size;
    
    if (i < dims_prod) {
        int b = i / C;
        int v = i % C;
        // i = b * C + v

        float *max_b = maxed + b;
        float *argmax_b = argmaxed + b;

        float ix = tensor[i];

        // max
        int *addr_as_int = (int *)max_b;
        int old_int = *addr_as_int, assumed_int;
        float old_val;
        do {
            assumed_int = old_int;
            old_val = __int_as_float(assumed_int);
            if (old_val >= ix) break;
            old_int = atomicCAS(addr_as_int, assumed_int, __float_as_int(ix));
        } while (assumed_int != old_int);

        // argmax
        if (__int_as_float(old_int) < ix) {
            int *addr_as_int_argmax = (int *)argmax_b;
            atomicExch(addr_as_int_argmax, __float_as_int((float)v));
        }
      }
}


__global__ void topk_kernel(const float *tensor, float *topk,
                           float *maxed, float *argmaxed,
                           int dims_prod, int maxed_dim_size,
                           int j, int k) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    int C = maxed_dim_size;
    
    if (i < dims_prod) {
        int b = i / C;
        int v = i % C;
        // i = b * C + v

        float *max_b = maxed + b;
        float *argmax_b = argmaxed + b;
        float *topk_b = topk + b*k + j;

        float ix = tensor[i];

        // max
        int *addr_as_int = (int *)max_b;
        int old_int = *addr_as_int, assumed_int;
        float old_val;
        do {
            assumed_int = old_int;
            old_val = __int_as_float(assumed_int);
            if (old_val >= ix) break;
            old_int = atomicCAS(addr_as_int, assumed_int, __float_as_int(ix));
        } while (assumed_int != old_int);

        // argmax & topk
        if (__int_as_float(old_int) < ix) {
            int *addr_as_int_argmax = (int *)argmax_b;
            atomicExch(addr_as_int_argmax, __float_as_int((float)v));

            int *addr_as_int_topk = (int *)topk_b;
            atomicExch(addr_as_int_topk, __float_as_int((float)v));
        }
      }
}


__global__ void topk_erase_argmax_aux_kernel(float *tensor,
                           float *argmaxed, int dims_prod, int maxed_dim_size) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    int C = maxed_dim_size;
    
    if (i < dims_prod) {
        int b = i / C;
        int v = i % C;
        // i = b * C + v

        float *tensor_b = tensor + b * C;

        float ix = argmaxed[b];

        float indicator = (v==ix) ? 0 : ix;
        if (v==ix)
          tensor_b[v] = 0;
      }
}