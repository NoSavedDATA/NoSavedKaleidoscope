#pragma once


#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <mma.h>



template<int tile_size, int block_rows>
__global__ void transpose_kernel(const float *__restrict__ X, float *__restrict__ Y)
{
  __shared__ float tile[tile_size * tile_size];

  int x = blockIdx.x * tile_size + threadIdx.x;
  int y = blockIdx.y * tile_size + threadIdx.y;
  int width = gridDim.x * tile_size;

  for (int j = 0; j < tile_size; j += block_rows)
     tile[(threadIdx.y+j)*tile_size + threadIdx.x] = X[(y+j)*width + x];

  __syncthreads();

  for (int j = 0; j < tile_size; j += block_rows)
     Y[(y+j)*width + x] = tile[(threadIdx.y+j)*tile_size + threadIdx.x];          
}