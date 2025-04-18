#include <vector>
#include <cuda_runtime.h>

#include "../tensor/include.h"
#include "../cuda_threads/include.h"
#include "../mangler/scope_struct.h"
#include "include.h"

extern "C" void *tensor_float_mult(Tensor *tensor, float R, Scope_Struct *scope_struct) {
  //std::cout << "CudaScalarMult by " << R << "\n";
  int thread_id = scope_struct->thread_id;
  
  int kDataLen = tensor->dims_prod;

  
  float* device_y = get_from_pool(thread_id, kDataLen, "scalar mult");
  

  int grid_size, block_size;
  std::vector<int> grid_block_mem_sizes = CalculateGridAndBlockSizes(kDataLen);
  grid_size = grid_block_mem_sizes[0];
  block_size = grid_block_mem_sizes[1];
  
  tensor->Sync();
  cudaStream_t stream = ThreadsStream[thread_id];
  vec_mult<<<grid_size, block_size, 0, stream>>>(R, tensor->tensor_ptr, device_y, kDataLen);

  Tensor *new_tensor = createTensor(device_y, tensor->dims, kDataLen, false, "");
  new_tensor->AttrLNode(tensor, scalar_mult_op);
  new_tensor->scalar = R;
  return new_tensor;
}


extern "C" void *tensor_float_div(Tensor tensor, float R, Scope_Struct *scope_struct) {
  int thread_id = scope_struct->thread_id;

  int kDataLen = tensor.dims_prod;


  
  float* device_y = get_from_pool(thread_id, kDataLen, "scalar div");


  int grid_size, block_size;
  std::vector<int> grid_block_mem_sizes = CalculateGridAndBlockSizes(kDataLen);
  grid_size = grid_block_mem_sizes[0];
  block_size = grid_block_mem_sizes[1];

  tensor.Sync();
  cudaStream_t stream = ThreadsStream[thread_id];
  vec_div<<<grid_size, block_size, 0, stream>>>(R, tensor.tensor_ptr, device_y, kDataLen);

  
  Tensor *new_tensor = createTensor(device_y, tensor.dims, kDataLen, false, "");
  return new_tensor;
}

// extern "C" void *CudaReverseScalarDiv(Tensor tensor, float R, Scope_Struct *scope_struct) {
//   int thread_id = scope_struct->thread_id;

//   int kDataLen = tensor.dims_prod;


  
//   float* device_y = get_from_pool(thread_id, kDataLen, "reverse scalar div");


//   int grid_size, block_size;
//   std::vector<int> grid_block_mem_sizes = CalculateGridAndBlockSizes(kDataLen);
//   grid_size = grid_block_mem_sizes[0];
//   block_size = grid_block_mem_sizes[1];
  
//   tensor.Sync();
//   cudaStream_t stream = ThreadsStream[thread_id];
//   vec_reverse_div<<<grid_size, block_size, 0, stream>>>(R, tensor.tensor_ptr, device_y, kDataLen);

//   Tensor *new_tensor = createTensor(device_y, tensor.dims, kDataLen, false, "");
//   return new_tensor;
// }

extern "C" void *tensor_float_add(Tensor *tensor, float R, Scope_Struct *scope_struct) {
  int thread_id = scope_struct->thread_id;
  
  int dims_prod = tensor->dims_prod;


  float* device_y = get_from_pool(thread_id, dims_prod, "scalar add");
  
  
  int grid_size, block_size;
  std::vector<int> grid_block_mem_sizes = CalculateGridAndBlockSizes(dims_prod);
  grid_size = grid_block_mem_sizes[0];
  block_size = grid_block_mem_sizes[1];
  
  tensor->Sync();
  cudaStream_t stream = ThreadsStream[thread_id];
  vec_add<<<grid_size, block_size, 0, stream>>>(R, tensor->tensor_ptr, device_y, dims_prod);
  
  Tensor *new_tensor = createTensor(device_y, tensor->dims, dims_prod, false, "");
  new_tensor->AttrLNode(tensor, scalar_add_op);
  return new_tensor;
}

extern "C" void *tensor_float_sub(Tensor *tensor, float R, Scope_Struct *scope_struct) {
  int thread_id = scope_struct->thread_id;

  int kDataLen = tensor->dims_prod;


  float* device_y = get_from_pool(thread_id, kDataLen, "scalar sub");


  int grid_size, block_size;
  std::vector<int> grid_block_mem_sizes = CalculateGridAndBlockSizes(kDataLen);
  grid_size = grid_block_mem_sizes[0];
  block_size = grid_block_mem_sizes[1];
  
  tensor->Sync();
  cudaStream_t stream = ThreadsStream[thread_id];
  vec_sub<<<grid_size, block_size, 0, stream>>>(R, tensor->tensor_ptr, device_y, kDataLen);

  Tensor *new_tensor = createTensor(device_y, tensor->dims, kDataLen, false, "");
  new_tensor->AttrLNode(tensor, scalar_sub_op);
  return new_tensor;
}

extern "C" void *tensor_float_equal(Tensor tensor, float R, Scope_Struct *scope_struct) {
  int thread_id = scope_struct->thread_id;

  int kDataLen = tensor.dims_prod;


  float* device_y = get_from_pool(thread_id, kDataLen, "scalar sub");


  int grid_size, block_size;
  std::vector<int> grid_block_mem_sizes = CalculateGridAndBlockSizes(kDataLen);
  grid_size = grid_block_mem_sizes[0];
  block_size = grid_block_mem_sizes[1];
  
  tensor.Sync();
  cudaStream_t stream = ThreadsStream[thread_id];
  vec_equal<<<grid_size, block_size, 0, stream>>>(R, tensor.tensor_ptr, device_y, kDataLen);

  Tensor *new_tensor = createTensor(device_y, tensor.dims, kDataLen, false, "");
  return new_tensor;
}

extern "C" void *tensor_float_diff(Tensor tensor, float R, Scope_Struct *scope_struct) {
  int thread_id = scope_struct->thread_id;

  int kDataLen = tensor.dims_prod;


  float* device_y = get_from_pool(thread_id, kDataLen, "scalar sub");


  int grid_size, block_size;
  std::vector<int> grid_block_mem_sizes = CalculateGridAndBlockSizes(kDataLen);
  grid_size = grid_block_mem_sizes[0];
  block_size = grid_block_mem_sizes[1];

  tensor.Sync();
  cudaStream_t stream = ThreadsStream[thread_id];
  vec_diff<<<grid_size, block_size, 0, stream>>>(R, tensor.tensor_ptr, device_y, kDataLen);

  Tensor *new_tensor = createTensor(device_y, tensor.dims, kDataLen, false, "");
  return new_tensor;
}
extern "C" void *tensor_float_minor(Tensor tensor, float R, Scope_Struct *scope_struct) {
  int thread_id = scope_struct->thread_id;

  int kDataLen = tensor.dims_prod;


  float* device_y = get_from_pool(thread_id, kDataLen, "scalar sub");


  int grid_size, block_size;
  std::vector<int> grid_block_mem_sizes = CalculateGridAndBlockSizes(kDataLen);
  grid_size = grid_block_mem_sizes[0];
  block_size = grid_block_mem_sizes[1];

  tensor.Sync();
  cudaStream_t stream = ThreadsStream[thread_id];
  vec_minor<<<grid_size, block_size, 0, stream>>>(R, tensor.tensor_ptr, device_y, kDataLen);

  Tensor *new_tensor = createTensor(device_y, tensor.dims, kDataLen, false, "");
  return new_tensor;
}
extern "C" void *tensor_float_minor_eq(Tensor tensor, float R, Scope_Struct *scope_struct) {
  int thread_id = scope_struct->thread_id;

  int kDataLen = tensor.dims_prod;


  float* device_y = get_from_pool(thread_id, kDataLen, "scalar sub");


  int grid_size, block_size;
  std::vector<int> grid_block_mem_sizes = CalculateGridAndBlockSizes(kDataLen);
  grid_size = grid_block_mem_sizes[0];
  block_size = grid_block_mem_sizes[1];

  tensor.Sync();
  cudaStream_t stream = ThreadsStream[thread_id];
  vec_minor_eq<<<grid_size, block_size, 0, stream>>>(R, tensor.tensor_ptr, device_y, kDataLen);

  Tensor *new_tensor = createTensor(device_y, tensor.dims, kDataLen, false, "");
  return new_tensor;
}
extern "C" void *tensor_float_higher(Tensor tensor, float R, Scope_Struct *scope_struct) {
  int thread_id = scope_struct->thread_id;

  int kDataLen = tensor.dims_prod;


  float* device_y = get_from_pool(thread_id, kDataLen, "scalar sub");


  int grid_size, block_size;
  std::vector<int> grid_block_mem_sizes = CalculateGridAndBlockSizes(kDataLen);
  grid_size = grid_block_mem_sizes[0];
  block_size = grid_block_mem_sizes[1];

  tensor.Sync();
  cudaStream_t stream = ThreadsStream[thread_id];
  vec_higher<<<grid_size, block_size, 0, stream>>>(R, tensor.tensor_ptr, device_y, kDataLen);

  Tensor *new_tensor = createTensor(device_y, tensor.dims, kDataLen, false, "");
  return new_tensor;
}
extern "C" void *tensor_float_higher_eq(Tensor tensor, float R, Scope_Struct *scope_struct) {
  int thread_id = scope_struct->thread_id;

  int kDataLen = tensor.dims_prod;


  float* device_y = get_from_pool(thread_id, kDataLen, "scalar sub");


  int grid_size, block_size;
  std::vector<int> grid_block_mem_sizes = CalculateGridAndBlockSizes(kDataLen);
  grid_size = grid_block_mem_sizes[0];
  block_size = grid_block_mem_sizes[1];

  tensor.Sync();
  cudaStream_t stream = ThreadsStream[thread_id];
  vec_higher_eq<<<grid_size, block_size, 0, stream>>>(R, tensor.tensor_ptr, device_y, kDataLen);

  Tensor *new_tensor = createTensor(device_y, tensor.dims, kDataLen, false, "");
  return new_tensor;
}


void scalarmult_backward(float *dx, float *dy, float scalar, float dims_prod)
{
  //std::cout << "scalar mult backward with scalar " << scalar <<  "\n";
  int grid_size, block_size;
  std::vector<int> grid_block_mem_sizes = CalculateGridAndBlockSizes(dims_prod);
  grid_size = grid_block_mem_sizes[0];
  block_size = grid_block_mem_sizes[1];

  scalarmult_backward_kernel<<<grid_size, block_size, 0, main_stream->stream>>>(dx, dy, scalar, dims_prod);
}