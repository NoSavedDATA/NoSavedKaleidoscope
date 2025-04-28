#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <cstdarg>
#include <iostream>
#include <vector>


#include "../backprop/include.h"
#include "../common/include.h"
#include "../compiler_frontend/logging.h"
#include "../tensor/include.h"
#include "calculate_grids.h"
#include "dim_kernels.h"
#include "handles.h"
#include "template_dim_kernels.h"





extern "C" void *repeat_interleave(int thread_id, Tensor tensor, float repeats, float dim)
{
  //std::cout << "REPEAT_interleave OF " << tensor.name << " with " << repeats << " repeats.\n";

  float *tensor_ptr = tensor.tensor_ptr;
  std::vector<float> dims, new_dims;
  dims = tensor.dims;
  if (dim<0)
    dim = dims.size()+dim;
  new_dims = tensor.dims;
  new_dims[dim] = new_dims[dim]*repeats;
  
  int B = DimsProd(dims);
  int C = (int)repeats;

  float *probs;

  probs = get_from_pool(thread_id, B*C, "repeat_interleave");
  cudaMemset(probs, 0, B*C*sizeof(float));
  


  int grid_size = B;
  int block_size = 32;
  size_t shared_mem_size = 2 * block_size / 32 * sizeof(float);

  cudaStream_t stream = ThreadsStream[thread_id];
  if (dim==(dims.size()-1))
    repeat_interleave_kernel_last_dim<<<grid_size, block_size, shared_mem_size, stream>>>(tensor_ptr, probs, B, C);
  //grid_size = ceil_div(B*C, block_size);
  //onehot_kernel<<<grid_size, block_size>>>(tensor, probs, B, C);


  Tensor *new_tensor = createTensor(probs, new_dims, DimsProd(new_dims), false, "");
  return new_tensor;
}





//TODO: mean over axis
extern "C" void *tensor_mean(Scope_Struct *scope_struct, Tensor *tensor, float first_dim, ...)
{
  int thread_id = scope_struct->thread_id;
  //std::cout << "MEAN OF " << tensor->name << "\n";


  float *tensor_ptr = tensor->tensor_ptr;
  std::vector<float> dims = tensor->dims;
  float *summed;

  cudaStream_t stream = ThreadsStream[thread_id];

  va_list args;
  va_start(args, first_dim);

  if (first_dim==TERMINATE_VARARG)
  { 
    va_end(args);
    float *ret;
    int dims_prod = DimsProd(dims);

    summed = new float[dims_prod];
    cudaCheck(cudaMemcpyAsync(summed, tensor_ptr, dims_prod*sizeof(float), cudaMemcpyDeviceToHost, stream));

    cudaCheck(cudaMalloc(&ret, 1*sizeof(float)));
  
    float tensor_sum=0;
    for(int i=0; i<dims_prod; i++)
      tensor_sum += summed[i];
    tensor_sum = tensor_sum/tensor->dims_prod;
    
    delete[] summed;
  
    float *aux = new float[1];
    aux[0] = tensor_sum;
    cudaCheck(cudaMemcpyAsync(ret, aux, 1*sizeof(float), cudaMemcpyHostToDevice, stream));
    delete[] aux;
  
    std::vector<float> new_dims;
    new_dims.push_back(1.0f);
  
    Tensor *new_tensor = createTensor(ret, new_dims, 1.0f, false, "");
    new_tensor->op=mean_op;
    new_tensor->AttrLNode(tensor, mean_op);
    return new_tensor;
  }


  std::vector<float> sum_dims, new_dims;
  if (first_dim<0)
    first_dim = dims.size()+first_dim;
  sum_dims.push_back(first_dim);

  for (int i=0; i<10; i++)
  {
    if (i==9)
    {
      LogErrorS("A tensor with 10 dimensions??? (mean)");
      std::cout << "Input tensor dims:" << "\n";
      PrintDims(tensor->dims);
      std::cout << "Mean dims:" << "\n";
      PrintDims(sum_dims);
      return nullptr;
    }

    float dim = va_arg(args, float);
    
    if (dim==TERMINATE_VARARG)
      break;
    if (in_float_vec(dim, sum_dims))  
    {
      std::string _error = "Dim "+std::to_string(dim) + " duplicated at tensor.mean() operation.";
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
    if (!in_float_vec(i, sum_dims))
      new_dims.push_back(dims[i]);
    else
      summed_dim=dims[i];


  int dims_prod = DimsProd(dims);
  int new_dims_prod = DimsProd(new_dims);

  
  summed = get_from_pool(thread_id, new_dims_prod, "mean");
  cudaMemset(summed, 0, new_dims_prod * sizeof(float));

  //std::cout << "\n\nDims prod: " << dims_prod << "\nNew dims prod: " << new_dims_prod << "\nSummed dim size: " << summed_dim << "\n\n";

  
  
  

  if (sum_dims[0]==(dims.size()-2))
  {
    std::vector<float> _dims = RemoveLastDim(RemoveLastDim(dims));
    dims_prod = DimsProd(_dims);

    int warps_per_block = THREADS_PER_BLOCK/WARP_SIZE;
    //warps_per_block = fminf(warps_per_block, dims[dims.size()-2]);
    

    mean_over_semilast_dim_kernel<<<dims_prod, warps_per_block*WARP_SIZE, 0, stream>>>(tensor_ptr, summed, dims_prod, dims[dims.size()-2], dims[dims.size()-1], warps_per_block);

    Tensor *new_tensor = createTensor(summed, new_dims, new_dims_prod, false, "");
    new_tensor->AttrLNode(tensor, mean_over_semilast_dim_op);
    new_tensor->scalar = dims[dims.size()-2];
    return new_tensor;
  }

  /*
  if (dims.size()==1)
  {
    sum_single_dim_kernel<<<grid_size, block_size, shared_mem_size>>>(tensor_ptr, summed, dims_prod);
    new_dims = {1.0f};
  }
  else if (sum_dims[0]==(dims.size()-1))
    sum_over_last_dim_kernel<<<grid_size, block_size, shared_mem_size>>>(tensor_ptr, summed, dims_prod, summed_dim);
  if (sum_dims[0]==(dims.size()-2))
    sum_over_semilast_dim_kernel<<<grid_size, block_size, shared_mem_size>>>(tensor_ptr, summed, dims_prod, dims[dims.size()-1], dims[dims.size()-2]);


  Tensor *new_tensor = createTensor(summed, new_dims, DimsProd(new_dims), false, "");
  return new_tensor;
  */
  LogErrorS("Mean of specific dim is not implemented yet.");
  return nullptr;
}


void mean_over_semilast_dim_backward(float *dx, float *dy, Tensor *node)
{
  std::vector<float> dims = node->L_Node->dims;
  float x_dims_prod = node->L_Node->dims_prod;
  float y_dims_prod = node->dims_prod;


  mean_over_semilast_dim_backward_kernel<<<std::ceil(x_dims_prod/(float)THREADS_PER_BLOCK), THREADS_PER_BLOCK, 0, main_stream->stream>>>(dx, dy,  x_dims_prod, dims[dims.size()-2], dims[dims.size()-1]);
}

extern "C" void *sum(int thread_id, Tensor tensor, float first_dim, ...)
{
  //std::cout << "SUM OF " << tensor.name << "\n";


  float *tensor_ptr = tensor.tensor_ptr;
  std::vector<float> dims = tensor.dims;
  float *summed;

  cudaStream_t stream = ThreadsStream[thread_id];

  va_list args;
  va_start(args, first_dim);

  if (first_dim==TERMINATE_VARARG)
  {
    va_end(args);
    float *ret;
    int dims_prod = DimsProd(dims);

    summed = new float[dims_prod];
    cudaCheck(cudaMemcpyAsync(summed, tensor_ptr, dims_prod*sizeof(float), cudaMemcpyDeviceToHost, stream));

    cudaCheck(cudaMalloc(&ret, 1*sizeof(float)));
  
    float tensor_sum=0;
    for(int i=0; i<dims_prod; i++)
      tensor_sum += summed[i];
    
    delete[] summed;
  
    float *aux = new float[1];
    aux[0] = tensor_sum;
    cudaCheck(cudaMemcpy(ret, aux, 1*sizeof(float), cudaMemcpyHostToDevice));  
    delete[] aux;
  
    std::vector<float> new_dims;
    new_dims.push_back(1.0f);
  
    Tensor *new_tensor = createTensor(ret, new_dims, 1.0f, false, "");
    new_tensor->op=sum_op;
    return new_tensor;
  }


  std::vector<float> sum_dims, new_dims;
  if (first_dim<0)
    first_dim = dims.size()+first_dim;
  sum_dims.push_back(first_dim);

  for (int i=0; i<10; i++)
  {
    if (i==9)
    {
      LogErrorS("A tensor with 10 dimensions??? (sum)");
      return nullptr;
    }

    float dim = va_arg(args, float);
    
    if (dim==TERMINATE_VARARG)
      break;
    if (in_float_vec(dim, sum_dims))  
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
    if (!in_float_vec(i, sum_dims))
      new_dims.push_back(dims[i]);
    else
      summed_dim=dims[i];


  int dims_prod = DimsProd(dims);
  int new_dims_prod = DimsProd(new_dims);

  
  summed = get_from_pool(thread_id, new_dims_prod, "summed");
  cudaMemset(summed, 0, new_dims_prod * sizeof(float));

  //std::cout << "\n\nDims prod: " << dims_prod << "\nNew dims prod: " << new_dims_prod << "\nSummed dim size: " << summed_dim << "\n\n";

  
  int grid_size = dims_prod;
  int block_size = 32;
  size_t shared_mem_size = 2 * block_size / 32 * sizeof(float);

  
  if (dims.size()==1)
  {
    sum_single_dim_kernel<<<grid_size, block_size, shared_mem_size, stream>>>(tensor_ptr, summed, dims_prod);
    new_dims = {1.0f};
  }
  else if (sum_dims[0]==(dims.size()-1))
    sum_over_last_dim_kernel<<<grid_size, block_size, shared_mem_size, stream>>>(tensor_ptr, summed, dims_prod, summed_dim);
  if (sum_dims[0]==(dims.size()-2))
    sum_over_semilast_dim_kernel<<<grid_size, block_size, shared_mem_size, stream>>>(tensor_ptr, summed, dims_prod, dims[dims.size()-1], dims[dims.size()-2]);


  Tensor *new_tensor = createTensor(summed, new_dims, DimsProd(new_dims), false, "");
  new_tensor->op=sum_op;
  return new_tensor;
}



extern "C" void *prod(int thread_id, Tensor tensor, float first_dim, ...)
{
  //std::cout << "PROD OF " << tensor.name << "\n";


  float *tensor_ptr = tensor.tensor_ptr;
  std::vector<float> dims = tensor.dims;
  float *summed;

  cudaStream_t stream = ThreadsStream[thread_id];

  va_list args;
  va_start(args, first_dim);

  if (first_dim==TERMINATE_VARARG)
  {
    va_end(args);
    int dims_prod = DimsProd(dims);

    summed = get_from_pool(thread_id, dims_prod, "prod all dims");
    cudaMemcpyAsync(summed, tensor_ptr, dims_prod*sizeof(float), cudaMemcpyDeviceToHost, stream);
    
    float tensor_sum=0;
    for(int i=0; i<dims_prod; i++)
      tensor_sum += summed[i];
    tensor_sum = tensor_sum;

    std::cout << "prod: " << tensor_sum << "\n";

    return summed;
  }


  std::vector<float> sum_dims, new_dims;
  if (first_dim<0)
    first_dim = dims.size()+first_dim;
  sum_dims.push_back(first_dim);

  for (int i=0; i<10; i++)
  {
    if (i==9)
    {
      LogErrorS("A tensor with 10 dimensions??? (prod)");
      return nullptr;
    }

    float dim = va_arg(args, float);
    
    if (dim==TERMINATE_VARARG)
      break;
    if (in_float_vec(dim, sum_dims))  
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
    if (!in_float_vec(i, sum_dims))
      new_dims.push_back(dims[i]);
    else
      summed_dim=dims[i];


  int dims_prod = DimsProd(dims);
  int new_dims_prod = DimsProd(new_dims);

  
  float *init_prod = new float[new_dims_prod];
  init_prod = make_ones_float(new_dims_prod);
  
  summed = get_from_pool(thread_id, new_dims_prod, "prod");
  cudaMemcpyAsync(summed, init_prod, new_dims_prod * sizeof(float), cudaMemcpyHostToDevice, stream);
  delete[] init_prod;

  //PrintTensorF(summed, new_dims_prod,1);

  //std::cout << "\n\nDims prod: " << dims_prod << "\nNew dims prod: " << new_dims_prod << "\nSummed dim size: " << summed_dim << "\n\n";

  
  int grid_size = dims_prod;
  int block_size = 32;
  size_t shared_mem_size = 2 * block_size / 32 * sizeof(float);

  
  if (dims.size()==1)
  {
    prod_single_dim_kernel<<<grid_size, block_size, shared_mem_size, stream>>>(tensor_ptr, summed, dims_prod);
    new_dims = {1.0f};
  }
  else if (sum_dims[0]==(dims.size()-1))
  {
    prod_over_last_dim_kernel<<<grid_size, block_size, shared_mem_size, stream>>>(tensor_ptr, summed, dims_prod, summed_dim);
    //std::cout << "prod_over_last_dim_kernel" << "\n";
  }
  if (sum_dims[0]==(dims.size()-2))
    prod_over_semilast_dim_kernel<<<grid_size, block_size, shared_mem_size, stream>>>(tensor_ptr, summed, dims_prod, dims[dims.size()-1], dims[dims.size()-2]);


  Tensor *new_tensor = createTensor(summed, new_dims, DimsProd(new_dims), false, "");
  return new_tensor;
}



extern "C" void *gather(int thread_id, Tensor *tensor, Tensor *idx_tensor, float dim)
{
  //std::cout << "Gather THREAD IS: " << thread_id << "\n";


  if(dim<0)
    dim = tensor->dims.size()+dim;

  if(dim == tensor->dims.size()-1)
  {
    //std::cout << "Gather over last dim"  << "\n";

    float *tensor_ptr = tensor->tensor_ptr;
    std::vector<float> dims, new_dims;
    dims = tensor->dims;
    new_dims = RemoveLastDim(dims);
    float leading_dim = dims[dim];

    //PrintDims(dims);
    //PrintDims(new_dims);

    
    float dims_prod = tensor->dims_prod;
    float new_dims_prod = DimsProd(new_dims);

    int grid_size, block_size, shared_mem_size; 
    std::vector<int> grid_block_mem_sizes = CalculateGridAndBlockSizes(new_dims_prod);
    grid_size = grid_block_mem_sizes[0];
    block_size = grid_block_mem_sizes[1];
    

    float *y = get_from_pool(thread_id, new_dims_prod, "gather");
    //float *y;
    

    tensor->Sync();
    cudaStream_t stream = ThreadsStream[thread_id];
    gather_last_dim_kernel<<<grid_size, block_size, 0, stream>>>(y, tensor->tensor_ptr, idx_tensor->tensor_ptr, leading_dim, new_dims_prod);



    

    Tensor *new_tensor = createTensor(y, new_dims, new_dims_prod, false, "");
    //idx_tensor->op = detach_op;
    new_tensor->AttrNodes(tensor, wrapTensorWithDetached(idx_tensor), gather_last_dim_op);
    //new_tensor->AttrLNode(idx_tensor, gather_last_dim_op);
    todo_backward_tensors.push_back(new_tensor);
    return new_tensor;
  }
}



void gather_last_dim_backward(float *dx, float *dy, Tensor *node)
{
  // consider dx was set to zero already
  

  float *idx = node->R_Node->tensor_ptr;

  std::vector<float> dims = node->L_Node->dims;
  int leading_dim = dims[dims.size()-1];

  float dims_prod = node->dims_prod;


  int grid_size, block_size, shared_mem_size; 
  std::vector<int> grid_block_mem_sizes = CalculateGridAndBlockSizes(dims_prod);
  grid_size = grid_block_mem_sizes[0];
  block_size = grid_block_mem_sizes[1];


  gather_last_dim_backward_kernel<<<grid_size, block_size, 0, main_stream->stream>>>(dx, dy, idx, leading_dim, dims_prod);

  //PrintTensorF(idx, 1, node->R_Node->dims_prod);
  //PrintTensorF(dx, dims[0], dims[1]);

}


inline void transpose(Tensor *tensor, int thread_id, cudaStream_t stream)
{

  float *transposed = get_from_pool(thread_id, tensor->dims_prod, "transpose");


  constexpr int tile_size{32}; // todo

  dim3 grid_size(std::ceil(tensor->dims[0]/(float)tile_size), std::ceil(tensor->dims[1]/(float)tile_size));
  dim3 block_size(tile_size, 8);

  transpose_kernel<tile_size, 8><<<grid_size, block_size, 0, stream>>>(tensor->tensor_ptr, transposed);

  // move_to_pool(thread_id, tensor->dims_prod, tensor->tensor_ptr, "transpose");
  tensor->tensor_ptr = transposed;
}