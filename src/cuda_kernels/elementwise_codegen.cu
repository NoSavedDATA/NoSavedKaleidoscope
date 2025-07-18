#include <cuda_runtime.h>
#include <iostream>
#include <vector>

#include "../cuda_threads/include.h"
#include "../nsk_cuda/pool/include.h"
#include "../tensor/include.h"
#include "include.h"



extern "C" DT_tensor *logE(int thread_id, DT_tensor tensor) {
  //std::cout << "logE of: " << tensor.name << "\n";

  float * device_x = tensor.tensor_ptr;
  std::vector<int> dims = tensor.dims;
  int kDataLen = tensor.dims_prod;


  float* device_y = get_from_pool(thread_id, kDataLen, "scalar sub");


  int grid_size, block_size;
  CalculateGridAndBlockSizes(kDataLen, grid_size, block_size);

  tensor.Sync();
  cudaStream_t stream = ThreadsStream[thread_id];
  vec_log<<<grid_size, block_size, 0, stream>>>(device_x, device_y, kDataLen);

  DT_tensor *new_tensor = createTensor(device_y, dims, kDataLen, false, "");
  return new_tensor;
}

extern "C" DT_tensor *logE2(int thread_id, DT_tensor tensor) {
  std::cout << "logE2 of: " << tensor.name << "\n";

  float * device_x = tensor.tensor_ptr;
  std::vector<int> dims = tensor.dims;
  int kDataLen = tensor.dims_prod;


  float* device_y = get_from_pool(thread_id, kDataLen, "scalar sub");


  int grid_size, block_size;
  CalculateGridAndBlockSizes(kDataLen, grid_size, block_size);

  tensor.Sync();
  cudaStream_t stream = ThreadsStream[thread_id];
  vec_log2<<<grid_size, block_size, 0, stream>>>(device_x, device_y, kDataLen);

  DT_tensor *new_tensor = createTensor(device_y, dims, kDataLen, false, "");
  return new_tensor;
}


extern "C" DT_tensor *clip(int thread_id, DT_tensor tensor, float _min, float _max)
{
  float *tensor_ptr = tensor.tensor_ptr;
  std::vector<int> dims = tensor.dims;
  
  int B = DimsProd(dims);

  float* device_y = get_from_pool(thread_id, B,"clip");


  int grid_size = B;
  int block_size = 32;
  size_t shared_mem_size = 2 * block_size / 32 * sizeof(float);
  cudaStream_t stream = ThreadsStream[thread_id];
  tensor_clip<<<grid_size, block_size, shared_mem_size, stream>>>(tensor_ptr, device_y, _min, _max, B);

  
  DT_tensor *new_tensor = createTensor(device_y, dims, tensor.dims_prod, false, "");
  new_tensor->op=clip_op; //TODO: what is the grad of clip?
  return new_tensor;
}


