

#include <algorithm>


#include "../structs/smem_cpasync_loader.h"
#include "../smem/gmem_to_smem.h"






__global__ void quantize_f32_i8_kernel(int8_t *x8, const float *x, float *scale_tensor, const float fraction, const int lower, const int upper, const int M, const int N, const int max_N, const int smem_N, const int num_warps, const int dims_prod)
{
  int tid = threadIdx.x;
  int warpId = threadIdx.x / warpSize;
  int laneId = threadIdx.x % warpSize;

  int cpasync_jump = num_warps / 4; //4 is from 4 jumped floats

  int block_x = blockIdx.x;

  int B_idx = block_x*num_warps + warpId;

  int xor_swap = 0, xor_swap_val = max_N/2;

  

  extern __shared__ float all_smem[];

  
  

  float *smem_i = all_smem + warpId*smem_N;
  

  const float *x_i = x + B_idx*N;
  int8_t *x8_i = x8 + B_idx*N;




  // // --- Load First Smem Tile and Get Quantization Statistics --- //
  int col = laneId*4;
  gmem_to_smem_safe(x_i, *(smem_i+col), (N-col)*4);
  asm volatile("cp.async.commit_group;");



  float maxval = -INFINITY;
  int tile=0;
  for(; tile<std::min(N, max_N); tile+=128)
  {
    col = tile + laneId*4;
    float *smem_ij = smem_i+col;
    int next_tile = tile+128;


    if (next_tile<max_N) {

      col = next_tile + laneId*4; // jump by 4 copied floats of cp.async (16B)
      const float *x_ij = x_i + col;
      
      
      float *smem_ij_next_tile = smem_i + col;
      
      gmem_to_smem_safe(x_ij, *smem_ij_next_tile, (N-col)*4);
      asm volatile("cp.async.commit_group;\n" ::);
      asm volatile("cp.async.wait_group %0;" ::"n"(1));
    } else
      asm volatile("cp.async.wait_all;");

    __syncthreads();
    


    for(int i=0; i<4; ++i)
    {
      float _maxval = smem_ij[i];
      if(_maxval<0)
        _maxval *= -1;


      float mask__maxval;
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



  float scale = 127/maxval;
  if(laneId==0)
  {
    if (B_idx<M)
      scale_tensor[B_idx] = scale;
  }  
  scale = __shfl_sync(0xFFFFFFFF, scale, 0);





  __syncthreads();

  for(tile=0; tile<std::min(N, max_N); tile+=128)
  {
    int col = tile+laneId*4;
    float *smem_ij = smem_i+col;

    for(int i=0; i<4; ++i)
    {
      if(col+i<N)
        x8_i[col+i] = (int8_t) min(max(smem_ij[i]*scale, -127.0f), 127.0f);
    }
    if (block_x==0&&threadIdx.x==0)
      printf("Tile %d\n", tile);
  }


  if (block_x==0&&threadIdx.x==0)
    printf("Tile post %d/%d\n ----", tile, N);



  if(N<max_N)
    return;

  gmem_to_smem_safe(x_i, *(smem_i+col), (N-col)*4);
  asm volatile("cp.async.commit_group;");

  for(; tile<(N-max_N); tile+=128)
  {

  }

}


