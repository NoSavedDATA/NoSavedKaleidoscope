

#include <algorithm>


#include "../structs/smem_cpasync_loader.h"
#include "../smem/gmem_to_smem.h"


#include "quantize.cuh"





__global__ void quantize_f32_i4_kernel(int8_t *x8, const float *x, float *scale_tensor, const float fraction, const int lower, const int upper, const int M, const int N, const int max_N, const int smem_N, const int num_warps, const int dims_prod)
{
  int tid = threadIdx.x;
  int warpId = threadIdx.x / warpSize;
  int laneId = threadIdx.x % warpSize;

  int cpasync_jump = num_warps / 4; //4 is from 4 jumped floats

  int block_x = blockIdx.x;

  int B_idx = block_x*num_warps + warpId;

  int xor_swap = 0, xor_swap_val = 128; // jumps of 4 per warpSize floats

  

  extern __shared__ float all_smem[];
  float *smem_i = all_smem + warpId*smem_N;
  

  const float *x_i = x + B_idx*N;
  int8_t *x8_i = x8 + B_idx*N/2;




  float top_k[6];
  #pragma unroll
  for (int i = 0; i < 6; ++i)
      top_k[i] = -INFINITY;




  // --- Load First Smem Tile and Get Quantization Statistics --- //
  int col = laneId*4;

  // if(laneId==0&&block_x==0)
  //   printf("block_x %d - B_idx %d - M %d - col %d - N %d \n", block_x, B_idx, M, col, N);

  if(B_idx<M)
  {
    gmem_to_smem_safe(x_i+col, *(smem_i+col), (N-col)*4);
  }
  asm volatile("cp.async.commit_group;");

  __syncthreads();


  float maxval = -INFINITY;
  int tile=0;

  #pragma unroll
  for(; tile<std::min(N, max_N); tile+=128)
  {
    col = tile + laneId*4;
    float *smem_ij = smem_i+col;
    int next_tile = tile+128;


    if (next_tile<max_N) {

      int next_col = next_tile + laneId*4; // jump by 4 copied floats of cp.async (16B)
      const float *x_ij = x_i + next_col;
      
      
      float *smem_ij_next_tile = smem_i + next_col;
      
      if(B_idx<M)
        gmem_to_smem_safe(x_ij, *smem_ij_next_tile, (N-next_col)*4);
      asm volatile("cp.async.commit_group;\n" ::);
      asm volatile("cp.async.wait_group %0;" ::"n"(1));
    } else
      asm volatile("cp.async.wait_all;");

    __syncthreads();
    


      
    #pragma unroll
    for(int i=0; i<4; ++i)
    {
      float _maxval;
      if(col+i<N)
      {
        _maxval = smem_ij[i]; 
        _maxval = abs(_maxval);
      }
      else 
        _maxval = -INFINITY;


      if (_maxval>top_k[5])
      {
        top_k[5] = _maxval;
        #pragma unroll
        for (int j=5; j>0 && top_k[j]>top_k[j-1]; --j) {
            float tmp = top_k[j];
            top_k[j] = top_k[j - 1];
            top_k[j - 1] = tmp;
        }
      }

      float mask__maxval;
    #pragma unroll
      for (int mask=warpSize/2; mask>0; mask>>=1)
      {
        __syncwarp();
        mask__maxval = __shfl_down_sync(0xFFFFFFFF, _maxval, mask);

        if (mask__maxval > _maxval)
            _maxval = mask__maxval;
        
      }
      _maxval = __shfl_sync(0xFFFFFFFF, _maxval, 0);

     if (maxval<_maxval&&laneId==0)
        maxval=_maxval;
      
      
    }
    __syncthreads();
  }



  #pragma unroll
  for (int k = 0; k < 6; ++k) {
    float v = top_k[k];
    #pragma unroll
    for (int mask = warpSize / 2; mask > 0; mask >>= 1) {
        __syncwarp();
        float shuffled = __shfl_down_sync(0xFFFFFFFF, v, mask);
        if (shuffled > top_k[k]) {
            top_k[k] = shuffled;
        }
    }
  }


  float quantile_clamp;
  // if(N>512)
  // {
  //   quantile_clamp = (1-fraction)*top_k[5]+fraction*top_k[4];
  //   maxval = quantile_clamp;
  // } else
    quantile_clamp = INFINITY;

  // if(block_x==0&&threadIdx.x==0)
  // {
  //   for(int i=0; i<6; ++i)
  //     printf("%f, ", top_k[i]);
  //   printf("\n");

  //   printf("max %f, 6 %f, 5 %f, fraction %f, clamp %f\n", top_k[0], top_k[5], top_k[4], fraction, quantile_clamp);
  // }





  // maxval = min(maxval, 30.0f);
  float scale = 8/maxval;


  if (std::isinf(scale)||scale<=0.0f)
    scale = 1.0f;

  // scale = min(scale, 1000000.0f);


  if (B_idx<M && laneId==0)
    scale_tensor[B_idx] = scale;
  scale = __shfl_sync(0xFFFFFFFF, scale, 0);




  __syncthreads();

  #pragma unroll
  for(tile=0; tile<std::min(N, max_N); tile+=128)
  {
    int col = tile+laneId*4;
    int col2 = col/2;
    float *smem_ij = smem_i+col;

    #pragma unroll
    for(int i=0; i<2; ++i) // 4 floats of cp_async jumped by 2 for packing
    {
      if(B_idx<M && col+i*2<std::min(N, max_N))
      {
        float val0 = min(max(smem_ij[i*2], -quantile_clamp), quantile_clamp) * scale;
        float val1;
        if(B_idx<M && col+i*2+1<std::min(N, max_N))
          val1 = min(max(smem_ij[i*2+1], -quantile_clamp), quantile_clamp) * scale;
        else
          val1 = 0.0f;

        // Quantize to [-8, 7]
        int8_t q0 = max(-8, min(7, __float2int_rn(val0)));
        int8_t q1 = max(-8, min(7, __float2int_rn(val1)));

        // if(blockIdx.x==0&&blockIdx.y==0&&threadIdx.x==0)
        //   printf("packing %d and %d from %f - %f, converted from %f - %f\n", (int)q0, (int)q1, val0, val1, smem_ij[i*2], smem_ij[i*2+1]);

        // Pack into 1 byte (2x int4)
        int8_t packed = (q1 << 4) | (q0 & 0xF);
        x8_i[col2+i] = packed;
      }
    }
  }

 



  __syncthreads();
  // --- Load Remanescent Tiles --- //
  if(N<=max_N)
    return;

  
  col = xor_swap + laneId*4;
  if(B_idx<M)
    gmem_to_smem_safe(x_i+col, *(smem_i+col), (N-col)*4);
  asm volatile("cp.async.commit_group;");

  __syncthreads();

  #pragma unroll
  for(; tile<N; tile+=128)
  {
    col = tile + laneId*4;
    int col2 = col/2;
    float *smem_ij = smem_i + xor_swap + laneId*4;
    
    xor_swap ^= xor_swap_val;

    int next_tile = tile+128;

    if (next_tile<N) {

      int next_col = next_tile + laneId*4; // jump by 4 copied floats of cp.async (16B)
      const float *x_ij = x_i + next_col;
      
      float *smem_ij_next_tile = smem_i + xor_swap+laneId*4;

      if(B_idx<M)
        gmem_to_smem_safe(x_ij, *smem_ij_next_tile, (N-next_col)*4);
      asm volatile("cp.async.commit_group;\n" ::);
      asm volatile("cp.async.wait_group %0;" ::"n"(1));
    } else
      asm volatile("cp.async.wait_all;");

    __syncthreads();



    #pragma unroll
    for(int i=0; i<2; ++i)
    {
      if(B_idx<M && col+i*2<N)
      {
        float val0 = min(max(smem_ij[i*2], -quantile_clamp), quantile_clamp) * scale;
        float val1;
        if(B_idx<M && col+i*2+1<N)
          val1 = min(max(smem_ij[i*2+1], -quantile_clamp), quantile_clamp) * scale;
        else
          val1 = 0.0f;


        // Quantize to [-8, 7]
        int8_t q0 = max(-8, min(7, __float2int_rn(val0)));
        int8_t q1 = max(-8, min(7, __float2int_rn(val1)));

        // Pack into 1 byte (2x int4)

        int8_t packed = (q1 << 4) | (q0 & 0xF);
        x8_i[col2+i] = packed;
      }
    }
    __syncthreads();
  }


}
