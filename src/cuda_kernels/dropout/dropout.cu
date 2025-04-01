#include <vector>
#include <cuda_runtime.h>

#include "../../codegen/random.h"
#include "../../tensor/include.h"
#include "../../cuda_threads/include.h"
#include "../include.h"


extern "C" void *dropout(int thread_id, Tensor *tensor, float rate)
{
  if (nn_mode==training_mode&&thread_id==0)
  {
    float dims_prod = tensor->dims_prod;

    int grid_size, block_size;
    std::vector<int> grid_block_mem_sizes = CalculateGridAndBlockSizes(dims_prod);
    grid_size = grid_block_mem_sizes[0];
    block_size = grid_block_mem_sizes[1];

    float *dropout_ptr = get_from_pool(thread_id, dims_prod, "dropout forward");
    float *device_y = get_from_pool(thread_id, dims_prod, "dropout forward output");

    float scale = 1 / (1-rate);
    
    unsigned long long seed = get_int_seed();

    dropout_mask_kernel<<<grid_size, block_size, 0, main_stream->stream>>>(device_y, dropout_ptr, tensor->tensor_ptr, rate, scale, dims_prod, seed);
    
    Tensor *dropout_tensor = createTensor(dropout_ptr, tensor->dims, dims_prod, true, "");
    dropout_tensor->scopeless_name="";

    Tensor *new_tensor = createTensor(device_y, tensor->dims, dims_prod, false, "");
    new_tensor->AttrNodes(tensor, dropout_tensor, dropout_op);
    return new_tensor;
  }
  return tensor;
}


void dropout_backward(float *dx, float *mask, float *dy, float dims_prod)
{
  int grid_size, block_size;
  std::vector<int> grid_block_mem_sizes = CalculateGridAndBlockSizes(dims_prod);
  grid_size = grid_block_mem_sizes[0];
  block_size = grid_block_mem_sizes[1];

  dropout_backward_kernel<<<grid_size, block_size, 0, main_stream->stream>>>(dx, mask, dy, dims_prod);
}