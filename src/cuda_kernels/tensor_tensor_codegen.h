#pragma once

#include <vector>

#include "../tensor/tensor_struct.h"
#include "calculate_grids.h"
#include "tensor_tensor_kernels.h"



void hadamard_backward2(float *x, float *w, float *dx, float *dw, float *dy, float dims_prod);




inline void cpp_tensor_tensor_add(float *x, float *y, float dims_prod, int thread_id=0) {
  int grid_size, block_size, shared_mem_size;
  std::vector<int> grid_block_mem_sizes = CalculateGridAndBlockSizes(dims_prod);
  grid_size = grid_block_mem_sizes[0];
  block_size = grid_block_mem_sizes[1];

  cudaStream_t stream = ThreadsStream[thread_id];
  
  add_inplace<<<grid_size, block_size, 0, stream>>>(x, y, dims_prod);
}