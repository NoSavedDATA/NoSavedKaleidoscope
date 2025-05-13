
#include <iostream>
#include <string>
#include <cstdarg>
#include <cuda_runtime.h>

#include "../backprop/include.h"
#include "../common/cu_commons.h"
#include "../compiler_frontend/include.h"
#include "../char_pool/include.h"
#include "../cuda_kernels/include.h"
#include "tensor_dim_functions.h"
#include "tensor_struct.h"








// Copies a pinned_tensor's reserved memory into a tensor.
extern "C" float AttrTensorNoFree(char *tensor_name, DT_tensor *tensor, int thread_id)
{
  //std::cout << "\nAttrTensorNoFree -- Attributing to tensor: " << tensor_name << "\n\n";
  
  std::vector<float> new_dims = tensor->dims;
  float dims_prod = tensor->dims_prod;

  

  DT_tensor *tgt_tensor = NamedTensorsT[tensor_name];
  move_to_pool(tgt_tensor->thread_id, tgt_tensor->dims_prod, tgt_tensor->tensor_ptr, "pinned");
  

  //float *new_tensor;
  //cudaMalloc(&new_tensor, dims_prod*sizeof(float));
  float *new_tensor = get_from_pool(thread_id, dims_prod, "pinned");

  int grid_size, block_size, shared_mem_size; 
  std::vector<int> grid_block_mem_sizes = CalculateGridAndBlockSizes(dims_prod);
  grid_size = grid_block_mem_sizes[0];
  block_size = grid_block_mem_sizes[1];

  tensor->Sync();
  copy_tensor_kernel<<<grid_size, block_size>>>(new_tensor, tensor->tensor_ptr, dims_prod);
  //cudaCheck(cudaMemcpy(new_tensor, tensor->tensor_ptr, dims_prod*sizeof(float), cudaMemcpyDeviceToDevice));


  tgt_tensor->AttrTensor(new_tensor, new_dims, dims_prod);
  
  

  delete tensor;

  return 0;
}


extern "C" float AttrTensorOnIdx(char *tensor_name, DT_tensor *tensor, float idx_at, int thread_id)
{ 
  //std::cout << "AttrTensorOnIdx of" << tensor_name << " at idx " << idx_at << "\n";

  std::vector<float> dims, Rdims;
  DT_tensor *tgt_tensor = NamedTensorsT[tensor_name];
  dims = tgt_tensor->dims;
  int dims_prod = tgt_tensor->dims_prod;

  Rdims = tensor->dims;

  if (idx_at>(dims_prod-1))
  {
    std::string _error = "\n\t- Idexating at pos: \033[32m"+std::to_string((int)idx_at);
    _error = _error + "\033[0m on tensor \033[95m"+std::string(tensor_name);
    _error = _error + "\033[0m;\n\t- Max idx allowed:  \033[32m"+std::to_string(dims_prod)+"\033[0m.";

    LogErrorS(_error);
    std::cout << "Dimensions:" << "\n";
    PrintDims(dims);
    std::cout << "\n";

    return -1;
  }

  int R_dims_prod = tensor->dims_prod;
  if ((idx_at+R_dims_prod)>(dims_prod))
  {
    std::string _error = "\n\t- Attributing at pos: \033[32m"+std::to_string((int)idx_at)+"\033[0m with a tensor of size \033[32m"+std::to_string(R_dims_prod)+"\033[0m";
    _error = _error + "\033[0m on tensor \033[95m"+std::string(tensor_name);
    _error = _error + "\033[0m;\n\t- Max idx allowed:  \033[32m"+std::to_string(dims_prod)+"\033[0m.";

    LogErrorS(_error);
    std::cout << "Dimensions:" << "\n";
    PrintDims(dims);
    std::cout << "\n";

    return -1;
  }

  float *base_address = tgt_tensor->tensor_ptr;
  float *device_x = base_address + static_cast<int>(idx_at);

  cudaStream_t stream = ThreadsStream[thread_id];
  //TODO*: turn into kernel
  cudaCheck(cudaMemcpyAsync(device_x, tensor->tensor_ptr, R_dims_prod*sizeof(float), cudaMemcpyDeviceToDevice, stream));
    
  return 0;
}


extern "C" float AttrTensorOnIdxTensor(char *tensor_name, char *idx_tensor_name, DT_tensor *R_tensor, int thread_id)
{ 
  //std::cout << "ATTR Idx tensor " << tensor_name << " at index tensor " << idx_tensor_name << " with tensor " << R_tensor->name << "\n";

  //std::cout << "\n\n\nIDX " << tensor_name << "\n\n\n\n";  
  
  
  DT_tensor *tensor = NamedTensorsT[tensor_name];
  DT_tensor *idx_tensor = NamedTensorsT[idx_tensor_name];


  float *tensor_ptr, *idx_tensor_ptr, *r_tensor_ptr;
  float dims_prod, new_dims_prod;
  std::vector<float> dims, idx_dims, new_dims;

  tensor_ptr = tensor->tensor_ptr;
  idx_tensor_ptr = idx_tensor->tensor_ptr;
  r_tensor_ptr = R_tensor->tensor_ptr;

  dims = tensor->dims;
  idx_dims = idx_tensor->dims;
  dims_prod = tensor->dims_prod;
  


  //TODO: gather with smaller dimensions
  /*
  if (dims.size()==1)
    new_dims = {1.0f};
  else
    for (int i = 0; i < dims.size()-1; i++)
      new_dims.push_back(dims[i+1]);
  */

  if (dims.size()<idx_dims.size())
  {
    LogErrorS("Index tensor must have less dimensions than the indexed tensor.");
    std::cout << "DT_tensor dims:" << "\n";
    PrintDims(dims);
    std::cout << "Idx tensor dims:" << "\n";
    PrintDims(idx_dims);
    return 0;
  }

  

  //std::cout << "dim size diff: " << dims.size()-idx_dims.size()  << "\n";

  cudaStream_t stream = ThreadsStream[thread_id];
  std::vector<int> grid_block_mem_sizes;
  int grid_size, block_size;

  
  //if((dims.size()-idx_dims.size())==0)
  if(dims.size()==1 && idx_dims.size()==1)
  {
    //std::cout << "INDEX ATTR OVER SIMPLE 1 DIM" << "\n";

    float idx_dims_prod = DimsProd(idx_dims);

    grid_block_mem_sizes = CalculateGridAndBlockSizes(idx_dims_prod);
    grid_size = grid_block_mem_sizes[0];
    block_size = grid_block_mem_sizes[1];

    //std::cout << "grid size: " << grid_size << " and block size: " << block_size << "\n";

    idx_attr_simple_single_dim_kernel<<<grid_size, block_size, 0, stream>>>(tensor_ptr, idx_tensor_ptr, r_tensor_ptr, idx_dims_prod);

  }
  if((dims.size()-idx_dims.size())==1)
  {
    //new_dims_prod = idx_tensor->dims_prod;
    //new_dims = idx_tensor->dims;

    std::cout << "INDEX ATTR OVER SEMI-LAST DIM" << "\n";

    grid_block_mem_sizes = CalculateGridAndBlockSizes(dims_prod);
    grid_size = grid_block_mem_sizes[0];
    block_size = grid_block_mem_sizes[1];  

    idx_attr_semi_last_dim_kernel<<<grid_size, block_size, 0, stream>>>(tensor_ptr, r_tensor_ptr, idx_tensor_ptr, dims_prod, dims_prod/idx_tensor->dims_prod);
  }


  if(thread_id==0)
  {
    idx_tensor->op = detach_op;
    R_tensor->op = detach_op;
    todo_backward_tensors.push_back(idx_tensor);
    todo_backward_tensors.push_back(R_tensor);
  }

  return 0;
}









extern "C" float AttrPinnedFromTensorOnIdx(char *tensor_name, DT_tensor *Rtensor, int thread_id, float first_idx, ...)
{
  
  //std::cout << "\n\n\nIDX " << tensor_name << "\n\n\n\n";  

  std::vector<float> idxs;

  va_list args;
  va_start(args, first_idx);

  idxs.push_back(first_idx);

  for (int i=0; i<10; i++)
  {
    float dim = va_arg(args, float);
    if (dim==TERMINATE_VARARG)
      break;
    idxs.push_back(dim);
  }
  va_end(args);  

  PrintDims(idxs);

  float offset = 0;
  std::vector<float> dims, aux_dims, Rdims;
  
  
  DT_tensor *tensor = NamedTensorsT[tensor_name];
  Rdims = Rtensor->dims;
  float R_dims_prod = Rtensor->dims_prod;

  float *new_tensor;

  dims = tensor->dims;
  std::vector<float> new_dims;

  if(idxs.size()>dims.size())
  {
    LogErrorS("The index used contain more dimensions than the indexed tensor.");
    return 0;
  }

  if (dims.size()==1)
    new_dims = {1.0f};
  else
  {
    aux_dims = dims;
    for (int i = 0; i < idxs.size(); i++)
    {
      aux_dims = RemoveFirstDim(aux_dims);
      offset += idxs[i]*DimsProd(aux_dims);
      std::cout << "ATTR INDEX DIMS_PROD IS " << DimsProd(aux_dims) << "\n";
    }
    new_dims = aux_dims;
  }


  int dims_prod = DimsProd(dims);
  if (offset>(dims_prod-1))
  {
    std::string _error = "\n\t- Idexating at pos: \033[32m"+std::to_string((int)offset);
    _error = _error + "\033[0m on tensor \033[95m"+std::string(tensor_name);
    _error = _error + "\033[0m;\n\t- Max idx allowed:  \033[32m"+std::to_string(dims_prod)+"\033[0m.";

    LogErrorS(_error);
    std::cout << "Dimensions:" << "\n";
    PrintDims(dims);
    std::cout << "\n";

    return 0;
  }
  //std::cout << "IDX AT: " << offset << "\n";


  float *base_address = tensor->cpu_tensor_ptr;
  float *device_x = base_address + static_cast<int>(offset);



  for (int i=0; i<R_dims_prod; i++)
    device_x[i] = Rtensor->cpu_tensor_ptr[i];

  /*
  new_tensor = get_from_pool(thread_id, R_dims_prod, "idx tensor");
  int grid_size, block_size;
  std::vector<int> grid_block_mem_sizes = CalculateGridAndBlockSizes(R_dims_prod);
  grid_size = grid_block_mem_sizes[0];
  block_size = grid_block_mem_sizes[1];

  tensor->Sync();
  cudaStream_t stream = ThreadsStream[thread_id];
  copy_tensor_kernel<<<grid_size, block_size, 0, stream>>>(device_x, Rtensor->cpu_tensor_ptr, R_dims_prod);
  


  DT_tensor *indexed = createTensor(new_tensor, new_dims, R_dims_prod, true, "");
  */
  return 0;
}



extern "C" void *IdxTensor(char *tensor_name, char *scope, int thread_id, float first_idx, ...)
{
  
  //std::cout << "\n\n\nIDX " << tensor_name << "\n\n\n\n";  

  std::vector<float> idxs;

  va_list args;
  va_start(args, first_idx);

  idxs.push_back(first_idx);

  for (int i=0; i<10; i++)
  {
    float dim = va_arg(args, float);
    if (dim==TERMINATE_VARARG)
      break;
    idxs.push_back(dim);
  }
  va_end(args);  

  PrintDims(idxs);

  float offset = 0;
  
  
  DT_tensor *tensor = NamedTensorsT[tensor_name];


  float *new_tensor;

  std::vector<float> dims, aux_dims;
  dims = tensor->dims;
  std::vector<float> new_dims;

  if(idxs.size()>dims.size())
  {
    LogErrorS("The index used contain more dimensions than the indexed tensor.");
    return nullptr;
  }

  if (dims.size()==1)
    new_dims = {1.0f};
  else
  {
    aux_dims = dims;
    for (int i = 0; i < idxs.size(); i++)
    {
      aux_dims = RemoveFirstDim(aux_dims);
      offset += idxs[i]*DimsProd(aux_dims);
      std::cout << "INDEX DIMS_PROD IS " << DimsProd(aux_dims) << "\n";
    }
    new_dims = aux_dims;
  }

  int new_dims_prod = DimsProd(new_dims);

  int dims_prod = DimsProd(dims);
  if (offset>(dims_prod-1))
  {
    std::string _error = "\n\t- Idexating at pos: \033[32m"+std::to_string((int)offset);
    _error = _error + "\033[0m on tensor \033[95m"+std::string(tensor_name);
    _error = _error + "\033[0m;\n\t- Max idx allowed:  \033[32m"+std::to_string(dims_prod)+"\033[0m.";

    LogErrorS(_error);
    std::cout << "Dimensions:" << "\n";
    PrintDims(dims);
    std::cout << "\n";

    return nullptr;
  }
  //std::cout << "IDX AT: " << offset << "\n";


  float *base_address = tensor->tensor_ptr;
  float *device_x = base_address + static_cast<int>(offset);



  new_tensor = get_from_pool(thread_id, new_dims_prod, "idx tensor");
  int grid_size, block_size;
  std::vector<int> grid_block_mem_sizes = CalculateGridAndBlockSizes(new_dims_prod);
  grid_size = grid_block_mem_sizes[0];
  block_size = grid_block_mem_sizes[1];

  tensor->Sync();
  cudaStream_t stream = ThreadsStream[thread_id];
  copy_tensor_kernel<<<grid_size, block_size, 0, stream>>>(new_tensor, device_x, new_dims_prod);
  

  /*
  PrintTensorF(new_tensor, 1, 1);
  PrintDims(new_dims);
  std::cout << "dims prod:" << new_dims_prod  << "\n";
  */

  if(nn_mode==eval_mode)
    ForwardCleanupToPool(tensor, scope);

  DT_tensor *indexed = createTensor(new_tensor, new_dims, new_dims_prod, true, "");
  return indexed;
}


extern "C" void *IdxTensorWithTensor(char *tensor_name, char *idx_tensor_name, int thread_id)
{
  //std::cout << "INDEXATE TENSOR " << tensor_name << " WITH TENSOR " << idx_tensor_name << "\n";

  //std::cout << "\n\n\nIDX " << tensor_name << "\n\n\n\n";  
  
  
  DT_tensor *tensor = NamedTensorsT[tensor_name];
  DT_tensor *idx_tensor = NamedTensorsT[idx_tensor_name];


  float *tensor_ptr, *idx_tensor_ptr, *new_tensor;
  float new_dims_prod;
  std::vector<float> dims, idx_dims, new_dims;

  tensor_ptr = tensor->tensor_ptr;
  idx_tensor_ptr = idx_tensor->tensor_ptr;

  dims = tensor->dims;
  idx_dims = idx_tensor->dims;


  //TODO: gather with smaller dimensions
  /*
  if (dims.size()==1)
    new_dims = {1.0f};
  else
    for (int i = 0; i < dims.size()-1; i++)
      new_dims.push_back(dims[i+1]);
  */
  

  std::cout << "dim size diff: " << dims.size()-idx_dims.size()  << "\n";
  if((dims.size()-idx_dims.size())==1)
  {
    new_dims_prod = idx_tensor->dims_prod;
    new_dims = idx_tensor->dims;

    //std::cout << "INDEX OVER LAST DIM" << "\n";

    //cudaMalloc(&new_tensor, new_dims_prod*sizeof(float));
    //cudaMemset(new_tensor, 0, new_dims_prod*sizeof(float));

    new_tensor = get_from_pool(thread_id, new_dims_prod, "idx tensor with tensor");
    
    //int grid_size = tensor->dims_prod;
    //int block_size = 32;

    int grid_size, block_size;
    std::vector<int> grid_block_mem_sizes = CalculateGridAndBlockSizes(tensor->dims_prod);
    grid_size = grid_block_mem_sizes[0];
    block_size = grid_block_mem_sizes[1];
    
    cudaStream_t stream = ThreadsStream[thread_id];
    idx_last_dim_kernel<<<grid_size, block_size, 0, stream>>>(new_tensor, tensor_ptr, idx_tensor_ptr, tensor->dims_prod, tensor->dims_prod/idx_tensor->dims_prod);
  }

  
  //cudaCheck(cudaMemcpy(new_tensor, device_x, new_dims_prod*sizeof(float), cudaMemcpyHostToHost));


  DT_tensor *indexed = createTensor(new_tensor, new_dims, new_dims_prod, false, "");
  indexed->AttrNodes(tensor, idx_tensor, idx_with_tensor_op);
  return indexed;
}