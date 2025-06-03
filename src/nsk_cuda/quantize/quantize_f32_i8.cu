
#include "../structs/smem_cpasync_loader.h"
#include "../smem/gmem_to_smem.h"







__global__ void quantize_f32_i8_kernel(int8_t *x8, const float *x, const float fraction, const int lower, const int upper, const int M, const int N, const int max_M, const int num_warps, const int dims_prod)
{
  int tid = threadIdx.x;
  int warpId = threadIdx.x / warpSize;
  int laneId = threadIdx.x % warpSize;

  int cpasync_jump = num_warps / 4; // = 2; 4 is from 4 jumped floats

  int block_x = blockIdx.x;
  int block_y = blockIdx.y;

  

  extern __shared__ float all_smem[];



  float *smem = all_smem;





  // if(threadIdx.x==0 && block_x==0)
  //   printf("block_y is %d - %d\n", block_y, block_y*max_M);
  // printf("%d - %d\n", (tid/cpasync_jump)*N, (block_x*num_warps+(tid%cpasync_jump)*4)); 
  

  for(int tile=0; tile<max_M; tile+=32)
  {
    const float *_x = x + (block_y*max_M+tile)*N + block_x*num_warps;
    int8_t *_x8 = x8 + (block_y*max_M+tile)*N + block_x*num_warps;
     
    if((block_y*max_M + tile + tid/cpasync_jump)<M && (block_x*num_warps+(tid%cpasync_jump)*4) < N)
    {
      // gmem_to_smem_xor(_x + (tid/cpasync_jump)*N + (tid%cpasync_jump)*4, *(smem + tid*4), ( std::min(std::max(N - (block_x*num_warps+(tid%cpasync_jump)*4), 16), 0)   )*4);
      for(int i=0; i<4; ++i)
      {
        if ((block_x*num_warps+(tid%cpasync_jump)*4)+i < N)
          smem[tid*4+i] = _x[(tid/cpasync_jump)*N + (tid%cpasync_jump)*4 + i];
      }
    }
    
    asm volatile("cp.async.commit_group;\n" ::);
    asm volatile("cp.async.wait_group %0;" ::"n"(1));

    __syncthreads();
    // printf("laneId*N %d warpId %d\n", laneId*N, warpId);

    if((block_y*max_M+tile+laneId)<M && block_x*num_warps+warpId<N)
    {
      int8_t q = (int8_t)fminf(fmaxf(smem[laneId*num_warps + warpId]*100, -128.0f), 127.0f);
      _x8[laneId*N + warpId] = q;
    }

  }
}
