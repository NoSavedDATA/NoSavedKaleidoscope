#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <curand_kernel.h>
#include <vector>
#include <iostream>

#include "../../mangler/scope_struct.h"
#include "../../nsk_cuda/pool/include.h"
#include "../../tensor/include.h"
#include "../elementwise_kernels_inline.cu"
#include "../calculate_grids.h"
#include "activation_kernels.h"




extern "C" DT_tensor *relu(Scope_Struct *scope_struct, DT_tensor *tensor)
{
  //std::cout << "RELU THREAD IS: " << thread_id << "\n";
  int thread_id = scope_struct->thread_id;
  float *tensor_ptr = tensor->tensor_ptr;
  std::vector<int> dims = tensor->dims;
  float dims_prod = tensor->dims_prod;

  int grid_size, block_size; 
  CalculateGridAndBlockSizes(dims_prod, grid_size, block_size);
  

  float *y = get_from_pool(thread_id, dims_prod, "relu");

  tensor->Sync();
  cudaStream_t stream = ThreadsStream[thread_id];
  relu_forward<<<grid_size, block_size, 0, stream>>>(tensor_ptr, y, dims_prod);


  return customOpTensor(y, dims, DimsProd(dims), "relu_backward", nullptr, tensor);
}


void relu_backward(float *inp, int dims_prod, float *out,
                     float *dinp, float *dout,
                     void *network_module, DT_tensor *node) {
  int grid_size, block_size;
  CalculateGridAndBlockSizes(dims_prod, grid_size, block_size);
  relu_backward1<<<grid_size, block_size, 0, main_stream>>>(inp, dinp, dout, dims_prod);
}


// void gelu_backward(const float* inp, float dims_prod, float* dinp, const float* dout) {
void gelu_backward(float *inp, int dims_prod, float *out,
                     float *dinp, float *dout,
                     void *network_module, DT_tensor *node) {  
  int grid_size, block_size; 
  CalculateGridAndBlockSizes(dims_prod, grid_size, block_size);
  gelu_backward1<<<grid_size, block_size, 0, main_stream>>>(dinp, inp, dout, dims_prod);  
}

extern "C" DT_tensor *gelu(Scope_Struct *scope_struct, DT_tensor *tensor)
{
  int thread_id = scope_struct->thread_id;
  float *tensor_ptr = tensor->tensor_ptr;
  std::vector<int> dims = tensor->dims;

  // std::cout << "GELU AT THREAD " << thread_id << "\n";
  
  float dims_prod = DimsProd(dims);


  int grid_size, block_size; 
  CalculateGridAndBlockSizes(dims_prod, grid_size, block_size);

  
  float *y = get_from_pool(thread_id, dims_prod,"gelu");

  tensor->Sync();
  cudaStream_t stream = ThreadsStream[thread_id];
  gelu_forward_kernel1<<<grid_size, block_size, 0, stream>>>(tensor_ptr, y, dims_prod);
  
 
  return customOpTensor(y, dims, DimsProd(dims), "gelu_backward", nullptr, tensor);
}




void sigmoid_backward(float *inp, int dims_prod, float *out,
                     float *dinp, float *dout,
                     void *network_module, DT_tensor *node) {  
  
  int grid_size, block_size; 
  CalculateGridAndBlockSizes(dims_prod, grid_size, block_size);

  sigmoid_backward_kernel<<<grid_size, block_size, 0, main_stream>>>(dinp, out, dout, dims_prod);
  
}

extern "C" DT_tensor *sigmoid(Scope_Struct *scope_struct, DT_tensor *tensor)
{
  int thread_id = scope_struct->thread_id;
  float *tensor_ptr = tensor->tensor_ptr;
  std::vector<int> dims = tensor->dims;
  

  float dims_prod = DimsProd(dims);

  int grid_size, block_size; 
  CalculateGridAndBlockSizes(dims_prod, grid_size, block_size);

  
  float *y = get_from_pool(thread_id, dims_prod, "sigmoid");  
  
  tensor->Sync();
  cudaStream_t stream = ThreadsStream[thread_id];
  sigmoid_forward_kernel<<<grid_size, block_size, 0, stream>>>(tensor_ptr, y, dims_prod);
  

  
  int is_forward_func=1;


  return customOpTensor(y, dims, DimsProd(dims), "sigmoid_backward", nullptr, tensor);
}


void tanh_backward(float *inp, int dims_prod, float *out,
                     float *dinp, float *dout,
                     void *network_module, DT_tensor *node) {  
  
  int grid_size, block_size; 
  CalculateGridAndBlockSizes(dims_prod, grid_size, block_size);
  

  tanh_backward_kernel<<<grid_size, block_size, 0, main_stream>>>(dinp, out, dout, dims_prod);
  
}



extern "C" DT_tensor *_tanh(Scope_Struct *scope_struct, DT_tensor *tensor)
{
  int thread_id = scope_struct->thread_id;
  float *tensor_ptr = tensor->tensor_ptr;
  std::vector<int> dims = tensor->dims;
  

  float dims_prod = DimsProd(dims);

  int grid_size, block_size; 
  CalculateGridAndBlockSizes(dims_prod, grid_size, block_size);

  
  float *y = get_from_pool(thread_id, dims_prod, "tanh");

  tensor->Sync();
  cudaStream_t stream = ThreadsStream[thread_id];
  tanh_forward_kernel<<<grid_size, block_size, 0, stream>>>(tensor_ptr, y, dims_prod);
    
  int is_forward_func=1;

  //std::cout << "tanh tensor attribution from " << tensor->name<<"/"<<tensor->scopeless_name << "\n";

  return customOpTensor(y, dims, DimsProd(dims), "tanh_backward", nullptr, tensor);
}




extern "C" DT_tensor *softmax(Scope_Struct *scope_struct, DT_tensor *tensor)
{
  int thread_id = scope_struct->thread_id;
  float *tensor_ptr = tensor->tensor_ptr;
  std::vector<int> dims = tensor->dims;
  
  dims =  format_LinearLayer_Dims(dims);

  int B = dims[0];
  int C = dims[1];


  int grid_size, block_size;
  CalculateGridAndBlockSizes(B*C, grid_size, block_size);


  tensor->Sync();
  float *probs = get_from_pool(thread_id, B*C, "softmax");
  cudaStream_t stream = ThreadsStream[thread_id];
  set_to_zero_kernel<<<grid_size, block_size, 0, stream>>>(probs, B*C);


  
  /*
  grid_block_mem_sizes = CalculateGridAndBlockSizes(B*C);
  grid_size  = B;
  block_size = grid_block_mem_sizes[1];
  shared_mem_size = 2 * block_size / 32 * sizeof(float);
  softmax_forward_kernel4<<<grid_size, block_size, shared_mem_size, stream>>>(tensor_ptr, probs, B, C);
  */
 
 
  CalculateSimpleWarpGridAndBlockSizes(B, grid_size, block_size);

  online_softmax<<<grid_size, block_size, 0, stream>>>(tensor_ptr, probs, B, C);



  DT_tensor *new_tensor = createTensor(probs, tensor->dims, tensor->dims_prod, false, "");
  new_tensor->op=softmax_op;
  return new_tensor;
}