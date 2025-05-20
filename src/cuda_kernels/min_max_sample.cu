#pragma once

#include <cstdarg>
#include <iostream>

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <mma.h>

#include "../common/include.h"
#include "../compiler_frontend/include.h"
#include "../tensor/include.h"
#include "calculate_grids.h"
#include "elementwise_kernels_inline.cu"
#include "tensor_scalar_kernels_inline.cu"
#include "min_max_sample_kernels.h"



extern "C" DT_tensor *tensor_onehot(Scope_Struct *scope_struct, DT_tensor *tensor, float num_classes)
{
  int thread_id = scope_struct->thread_id;
  // std::cout << "ONEHOT OF " << tensor->name << "\n";

  float *tensor_ptr = tensor->tensor_ptr;
  std::vector<int> dims, new_dims;
  dims = tensor->dims;
  new_dims = tensor->dims;
  new_dims.push_back(num_classes);
  
  int B = DimsProd(dims);
  int C = (int)num_classes;

  
  int grid_size, block_size, shared_mem_size; 
  std::vector<int> grid_block_mem_sizes = CalculateGridAndBlockSizes(B*C);
  grid_size = grid_block_mem_sizes[0];
  block_size = grid_block_mem_sizes[1];
  shared_mem_size = grid_block_mem_sizes[2];


  tensor->Sync();

  float *probs = get_from_pool(thread_id, B*C, "onehot probs");

  cudaStream_t stream = ThreadsStream[thread_id];
  set_to_zero_kernel<<<grid_size, block_size, 0, stream>>>(probs, B*C);

  onehot_kernel<<<grid_size, block_size, shared_mem_size>>>(tensor_ptr, probs, B, C);
  //grid_size = ceil_div(B*C, block_size);
  //onehot_kernel<<<grid_size, block_size>>>(tensor, probs, B, C);


  DT_tensor *new_tensor = createTensor(probs, new_dims, DimsProd(new_dims), false, "");
  new_tensor->AttrLNode(tensor, onehot_op);
  return new_tensor;
}


extern "C" float priority_sample(int thread_id, DT_tensor *tensor, float max_idx, float seed)
{
  
  float *probs, *sampled, *probs_cpu;
  float ret;
  probs = get_from_pool(thread_id, max_idx, "priority sample");
  sampled = get_from_pool(thread_id, 1, "priority sample");
  probs_cpu = new float[1];


  int grid_size, block_size;
  std::vector<int> grid_block_mem_sizes = CalculateGridAndBlockSizes(max_idx, 32);
  grid_size = grid_block_mem_sizes[0];
  block_size = grid_block_mem_sizes[1];

  //TODO: optimize to use shared memory and suprass 32 threads.
  cudaStream_t stream = ThreadsStream[thread_id];
  warped_to_probs_single_dim<<<grid_size, block_size, 0, stream>>>(probs, tensor->tensor_ptr, max_idx);


  //unsigned long long seed = get_int_seed();


  sample_from_probs<<<1, 1, 0, stream>>>(probs, sampled, max_idx, seed);


  cudaStreamSynchronize(stream);
  cudaMemcpy(probs_cpu, sampled, 1*sizeof(float), cudaMemcpyDeviceToHost);
  ret = probs_cpu[0];
  delete[] probs_cpu;



  move_to_pool(thread_id, max_idx, probs, "priority sample");
  move_to_pool(thread_id, 1, sampled, "priority sample");


  return ret;
}

extern "C" float priority_sample_val(int thread_id, DT_tensor *tensor, float max_idx, float seed)
{  
  float *probs, *sampled, *probs_cpu;
  float ret;
  probs = get_from_pool(thread_id, max_idx, "priority sample");
  sampled = get_from_pool(thread_id, 1, "priority sample");
  probs_cpu = new float[1];


  int grid_size, block_size;
  std::vector<int> grid_block_mem_sizes = CalculateGridAndBlockSizes(max_idx, 32);
  grid_size = grid_block_mem_sizes[0];
  block_size = grid_block_mem_sizes[1];

  //TODO: optimize to use shared memory and suprass 32 threads.
  cudaStream_t stream = ThreadsStream[thread_id];
  warped_to_probs_single_dim<<<grid_size, block_size, 0, stream>>>(probs, tensor->tensor_ptr, max_idx);


  //unsigned long long seed = get_int_seed();

  grid_block_mem_sizes = CalculateGridAndBlockSizes(max_idx);
  grid_size = grid_block_mem_sizes[0];
  block_size = grid_block_mem_sizes[1];
  sample_val_from_probs<<<1, 1, 0, stream>>>(probs, sampled, max_idx, seed);


  cudaStreamSynchronize(stream);
  cudaMemcpy(probs_cpu, sampled, 1*sizeof(float), cudaMemcpyDeviceToHost);
  ret = probs_cpu[0];
  delete[] probs_cpu;

  move_to_pool(thread_id, max_idx, probs, "priority sample");
  move_to_pool(thread_id, 1, sampled, "priority sample");


  return ret;
}


extern "C" float importance_sample_idx(int thread_id, DT_tensor *tensor, float max_idx, float alpha, float beta, float seed)
{  
  
  float *probs, *sampled, *probs_cpu;
  float ret;
  probs = get_from_pool(thread_id, tensor->dims_prod, "priority sample probs");
  sampled = get_from_pool(thread_id, 1, "priority sample sampled");
  probs_cpu = new float[1];


  int grid_size, block_size;
  std::vector<int> grid_block_mem_sizes = CalculateGridAndBlockSizes(max_idx, 32);
  grid_size = grid_block_mem_sizes[0];
  block_size = grid_block_mem_sizes[1];

  //TODO: optimize to use shared memory and suprass 32 threads.
  cudaStream_t stream = ThreadsStream[thread_id];
  
  warped_to_probs_single_dim_pow<<<grid_size, block_size, 0, stream>>>(probs, tensor->tensor_ptr, alpha, max_idx);


  sample_from_probs<<<1, 1, 0, stream>>>(probs, sampled, max_idx, seed);
  

  cudaStreamSynchronize(stream);
  cudaMemcpy(probs_cpu, sampled, 1*sizeof(float), cudaMemcpyDeviceToHost);
  ret = probs_cpu[0];
  
  delete[] probs_cpu;


  move_to_pool(thread_id, tensor->dims_prod, probs, "priority sample");
  move_to_pool(thread_id, 1, sampled, "priority sample");


  return ret;
}



extern "C" float importance_sample_weight(int thread_id, DT_tensor *tensor, float max_idx, float alpha, float beta, float seed)
{  
  float *probs, *sampled, *is_w_cpu, *is_w;
  float ret;
  probs = get_from_pool(thread_id, tensor->dims_prod, "importance_sample_weight probs");
  sampled = get_from_pool(thread_id, 1, "importance_sample_weight sampled");
  is_w = get_from_pool(thread_id, 1, "importance_sample_weight is_w");
  is_w_cpu = new float[1];


  int grid_size, block_size;
  std::vector<int> grid_block_mem_sizes = CalculateGridAndBlockSizes(max_idx, 32);
  grid_size = grid_block_mem_sizes[0];
  block_size = grid_block_mem_sizes[1];

  //TODO: optimize to use shared memory and suprass 32 threads.
  cudaStream_t stream = ThreadsStream[thread_id];
  warped_to_probs_single_dim_pow<<<grid_size, block_size, 0, stream>>>(probs, tensor->tensor_ptr, alpha, max_idx);


  sample_from_probs<<<1, 1, 0, stream>>>(probs, sampled, max_idx, seed);

  
  is_w_kernel<<<grid_size, block_size, 0, stream>>>(is_w, probs, sampled, beta, max_idx);

  cudaStreamSynchronize(stream);
  cudaMemcpy(is_w_cpu, is_w, 1*sizeof(float), cudaMemcpyDeviceToHost);
  ret = is_w_cpu[0];
  delete[] is_w_cpu;

  

  move_to_pool(thread_id, tensor->dims_prod, probs, "importance_sample_weight");
  move_to_pool(thread_id, 1, sampled, "importance_sample_weight");
  move_to_pool(thread_id, 1, is_w, "importance_sample_weight");


  return ret;
}


extern "C" DT_tensor *tmax(int thread_id, DT_tensor *tensor, float first_dim, ...) 
{ //TODO: automatic type detection for max and min (float vs tensor)
  
  //std::cout << "MAX OF " << tensor.name << "\n";
  

  float *tensor_ptr = tensor->tensor_ptr;
  std::vector<int> dims = tensor->dims;
  float *summed;

  cudaStream_t stream = ThreadsStream[thread_id];
  tensor->Sync();

  va_list args;
  va_start(args, first_dim);

  if (first_dim==TERMINATE_VARARG)
  {
    // va_end(args);
    // int dims_prod = DimsProd(dims);

    // summed = get_from_pool(thread_id, dims_prod, "tmax all dims");
    // cudaMemcpyAsync(summed, tensor_ptr, dims_prod*sizeof(float), cudaMemcpyDeviceToHost, stream);
    
    // float tensor_sum=0;
    // for(int i=0; i<dims_prod; i++)
    //   tensor_sum += summed[i];
    // tensor_sum = tensor_sum;

    // std::cout << "Sum: " << tensor_sum << "\n";

    // return summed;
    LogErrorS("MUST REIMPLEMENT MAX WITH NO DIMS AS INPUT");
    return nullptr;
  }


  std::vector<int> sum_dims, new_dims;
  if (first_dim<0)
    first_dim = dims.size()+first_dim;
  sum_dims.push_back(first_dim);

  for (int i=0; i<10; i++)
  {
    if (i==9)
    {
      LogErrorS("A tensor with 10 dimensions??? (max)");
      return nullptr;
    }

    float dim = va_arg(args, float);
    
    if (dim==TERMINATE_VARARG)
      break;
    if (in_int(dim, sum_dims))  
    {
      std::string _error = "Dim "+std::to_string(dim) + " duplicated at tensor.sum() operation.";
      LogErrorS(_error);
      return nullptr;
    }
    if (dim<0)
      dim = dims.size()+dim;
    sum_dims.push_back(dim);
  }
  va_end(args);
  
  
  float summed_dim;
  for (int i=0; i<dims.size(); i++)
    if (!in_int(i, sum_dims))
      new_dims.push_back(dims[i]);
    else
      summed_dim=dims[i];


  int dims_prod = DimsProd(dims);
  int new_dims_prod = DimsProd(new_dims);

  
  summed = get_from_pool(thread_id, new_dims_prod, "tmax");
  cudaMemset(summed, 0, new_dims_prod * sizeof(float));


  //std::cout << "\n\nDims prod: " << dims_prod << "\nNew dims prod: " << new_dims_prod << "\nSummed dim size: " << summed_dim << "\n\n";

  
  int grid_size = dims_prod;
  int block_size = 32;
  size_t shared_mem_size = 2 * block_size / 32 * sizeof(float);

  // AtomicMax does not handle negative numbers, so gambiarra. :D (1 hour for this)
  vec_add<<<grid_size, block_size, shared_mem_size, stream>>>(50000, tensor_ptr, tensor_ptr, dims_prod);
  if (sum_dims[0]==(dims.size()-1))
    max_over_last_dim_kernel<<<grid_size, block_size, shared_mem_size, stream>>>(tensor_ptr, summed, dims_prod, summed_dim);
  if (sum_dims[0]==(dims.size()-2))
    max_over_semilast_dim_kernel<<<grid_size, block_size, shared_mem_size, stream>>>(tensor_ptr, summed, dims_prod, dims[dims.size()-1], dims[dims.size()-2]);
  vec_sub<<<grid_size, block_size, shared_mem_size, stream>>>(50000, summed, summed, new_dims_prod);
  vec_sub<<<grid_size, block_size, shared_mem_size, stream>>>(50000, tensor_ptr, tensor_ptr, dims_prod);


  DT_tensor *new_tensor = createTensor(summed, new_dims, DimsProd(new_dims), false, "");
  new_tensor->AttrLNode(tensor, max_op);
  return new_tensor;
}



extern "C" DT_tensor *tensor_argmax(Scope_Struct *scope_struct, DT_tensor *tensor, float first_dim, ...) 
{
  int thread_id = scope_struct->thread_id;
  //std::cout << "ARGMAX OF " << tensor->name << " at thread: " << thread_id << "\n";
  cudaCheck(cudaGetLastError());

  float *tensor_ptr = tensor->tensor_ptr;
  std::vector<int> dims = tensor->dims;
  float *maxed, *argmaxed;


  va_list args;
  va_start(args, first_dim);

  if (first_dim==TERMINATE_VARARG)
  {
    va_end(args);
    LogErrorS("Argmax is only supported at dim -1.");
    return nullptr;
  }


  std::vector<int> sum_dims, new_dims;
  if (first_dim<0)
    first_dim = dims.size()+first_dim;
  sum_dims.push_back(first_dim);

  for (int i=0; i<10; i++)
  {
    if (i==9)
    {
      LogErrorS("A tensor with 10 dimensions??? (argmax)");
      return nullptr;
    }

    float dim = va_arg(args, float);
    
    if (dim==TERMINATE_VARARG)
      break;
    if (in_int(dim, sum_dims))  
    {
      std::string _error = "Dim "+std::to_string(dim) + " duplicated at tensor.argmax() operation.";
      LogErrorS(_error);
      return nullptr;
    }
    if (dim<0)
      dim = dims.size()+dim;
    sum_dims.push_back(dim);
  }
  va_end(args);
  
  
  float maxed_dim;
  for (int i=0; i<dims.size(); i++)
    if (!in_int(i, sum_dims))
      new_dims.push_back(dims[i]);
    else
      maxed_dim=dims[i];


  int dims_prod = DimsProd(dims);
  int new_dims_prod = DimsProd(new_dims);

    
  int grid_size, block_size, shared_mem_size; 
  std::vector<int> grid_block_mem_sizes = CalculateGridAndBlockSizes(new_dims_prod);
  grid_size = grid_block_mem_sizes[0];
  block_size = grid_block_mem_sizes[1];
  

  tensor->Sync();
  maxed = get_from_pool(thread_id, new_dims_prod, "argmax maxed");
  argmaxed = get_from_pool(thread_id, new_dims_prod, "argmax");

  cudaStream_t stream = ThreadsStream[thread_id];
  set_to_zero_kernel<<<grid_size, block_size, 0, stream>>>(maxed, new_dims_prod);
  set_to_zero_kernel<<<grid_size, block_size, 0, stream>>>(argmaxed, new_dims_prod);


  //std::cout << "\n\nDims prod: " << dims_prod << "\nNew dims prod: " << new_dims_prod << "\nMaxed dim size: " << summed_dim << "\n\n";

  
  grid_block_mem_sizes = CalculateGridAndBlockSizes(dims_prod);
  grid_size = grid_block_mem_sizes[0];
  block_size = grid_block_mem_sizes[1];
  shared_mem_size = grid_block_mem_sizes[2];

  
  vec_add<<<grid_size, block_size, shared_mem_size, stream>>>(50000, tensor_ptr, tensor_ptr, dims_prod);
  if (sum_dims[0]==(dims.size()-1))
    argmax_over_last_dim_kernel<<<grid_size, block_size, shared_mem_size, stream>>>(tensor_ptr, maxed, argmaxed, dims_prod, maxed_dim);
  //if (sum_dims[0]==(dims.size()-2))
  //  max_over_semilast_dim_kernel<<<grid_size, block_size, shared_mem_size>>>(tensor, maxed, dims_prod, dims[dims.size()-1], dims[dims.size()-2]);
  vec_sub<<<grid_size, block_size, shared_mem_size, stream>>>(50000, tensor_ptr, tensor_ptr, dims_prod);

  if(thread_id==0)
  {
    // std::cout << "MOVING MAXED TO POOL. maxed is " << new_dims_prod << "\n";
    move_to_pool(thread_id, new_dims_prod, maxed, "argmax maxed");
  }
  else
    cudaFree(maxed);
  

  DT_tensor *new_tensor = createTensor(argmaxed, new_dims, DimsProd(new_dims), false, "");
  new_tensor->AttrLNode(tensor, argmax_op);
  cudaCheck(cudaGetLastError());
  return new_tensor;
}


extern "C" DT_tensor *topk(int thread_id, DT_tensor tensor, float k) 
{
  std::cout << "TOPK OF " << tensor.name << "\n";

  float *tensor_ptr = tensor.tensor_ptr;
  std::vector<int> dims = tensor.dims;
  float *maxed, *argmaxed, *topk, *tensor_copy;


  std::vector<int> new_dims = RemoveLastDim(dims);
  std::vector<int> topk_dims = RemoveLastDim(dims);
  float new_dims_prod = DimsProd(new_dims);
  int dims_prod = DimsProd(dims);
  topk_dims.push_back(k);
  float topk_dims_prod = DimsProd(topk_dims);

  float maxed_dim = dims[dims.size()-1];

  cudaStream_t stream = ThreadsStream[thread_id];
  
  cudaMalloc(&maxed, new_dims_prod*sizeof(float));
  cudaMalloc(&argmaxed, new_dims_prod*sizeof(float));
  cudaMalloc(&topk, topk_dims_prod * sizeof(float));
  cudaMalloc(&tensor_copy, dims_prod*sizeof(float));
  cudaMemset(maxed, 0, new_dims_prod*sizeof(float));
  cudaMemset(argmaxed, 0, new_dims_prod*sizeof(float));
  cudaMemset(topk, 0, topk_dims_prod * sizeof(float));
  cudaMemcpyAsync(tensor_copy, tensor_ptr, dims_prod*sizeof(float), cudaMemcpyDeviceToDevice, stream);


  //std::cout << "\n\nDims prod: " << dims_prod << "\nNew dims prod: " << new_dims_prod << "\nMaxed dim size: " << summed_dim << "\n\n";

  
  int grid_size = dims_prod;
  int block_size = 32;
  size_t shared_mem_size = 2 * block_size / 32 * sizeof(float);

  
  vec_add<<<grid_size, block_size, shared_mem_size, stream>>>(50000, tensor_copy, tensor_copy, dims_prod);
  
  for (int i=0; i<k; i++)
  {
    //std::cout << "Top k at iter:" << i << "\n";
    topk_kernel<<<grid_size, block_size, shared_mem_size, stream>>>(tensor_copy, topk, maxed, argmaxed, dims_prod, maxed_dim, i, k);
    //PrintTensorF(maxed, 3, 1);
    //PrintTensorF(argmaxed, 3, 1);
    //std::cout << "Topk" << "\n";
    //PrintTensorF(topk, 3, k);
    topk_erase_argmax_aux_kernel<<<grid_size, block_size, shared_mem_size, stream>>>(tensor_copy, argmaxed, dims_prod, maxed_dim);
    cudaMemset(maxed, 0, new_dims_prod*sizeof(float));
    //std::cout << "\n\n\n\n";
  }
  cudaCheck(cudaFree(tensor_copy));
  cudaCheck(cudaFree(maxed));
  cudaCheck(cudaFree(argmaxed));

  DT_tensor *new_tensor = createTensor(topk, topk_dims, topk_dims_prod, false, "");
  new_tensor->op=topk_op;
  return new_tensor;
}