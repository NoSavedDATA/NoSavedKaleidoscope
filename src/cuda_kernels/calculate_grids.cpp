#pragma once

#include <vector>

#include "../common/cu_commons.h"


cudaDeviceProp deviceProp;


std::vector<int> CalculateGridAndBlockSizes(int dims_prod, int pre_block_size=-1)
{

  int grid_size, block_size, shared_mem_size;

  if (pre_block_size==-1)
  {
    if (dims_prod<64)
      block_size = 32;
    else if (dims_prod<128)
      block_size = 64;
    else if (dims_prod<256)
      block_size = 128;
    else if (dims_prod<512)
      block_size = 256;
    else
      block_size = deviceProp.maxThreadsPerMultiProcessor == 1536 ? 768 : 1024;
  } else
    block_size = pre_block_size;

  grid_size = ceil_div(dims_prod, block_size);
  shared_mem_size = 2 * block_size / 32 * sizeof(float);
  //shared_mem_size = std::min(2 * block_size * sizeof(float), deviceProp.sharedMemPerBlock);

  std::vector<int> ret = {grid_size, block_size, shared_mem_size};
  return ret;
}


std::vector<int> CalculateSimpleWarpGridAndBlockSizes(int B)
{
  // Usually warp kernels deal with the C dim already, so
  // you should inform B as the first dim only in these cases.

  int grid_size, block_size;

  block_size = deviceProp.maxThreadsPerMultiProcessor == 1536 ? 768 : 1024;
  
  while (B < block_size/32 && block_size>32)
     block_size = block_size / 2;

  if (block_size<32)
    block_size = 32;

  grid_size = ceil_div(B, block_size/32);

  std::vector<int> ret = {grid_size, block_size};
  return ret;
}