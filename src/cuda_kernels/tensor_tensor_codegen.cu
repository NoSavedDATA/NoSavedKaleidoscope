#include <vector>
#include <iostream>
#include <cuda_runtime.h>


#include "../common/cu_commons.h"
#include "../compiler_frontend/logging.h"
#include "../cuda_threads/include.h"
#include "../mangler/scope_struct.h"
#include "../mma/general.h"
#include "../tensor/include.h"
#include "include.h"

extern "C" Tensor *tensor_tensor_mma(
                          Tensor *tensor_x, Tensor *tensor_w, Scope_Struct *scope_struct) {

  int thread_id = scope_struct->thread_id;

  std::vector<float> Ldims, Rdims;
  Ldims = tensor_x->dims;
  Rdims = tensor_w->dims;
  float *device_x = tensor_x->tensor_ptr;
  float *device_w = tensor_w->tensor_ptr;


  std::vector<float> linear_layer_dims = format_LinearLayer_Dims(Ldims);
  int input_dims_prod = DimsProd(linear_layer_dims);
  
  int resultingDimsProd = resultingDimsProdOnMult(linear_layer_dims, Rdims);


  float* device_y = get_from_pool(thread_id, resultingDimsProd, "cuda mult");
    

  if (Ldims.size()<2)
    LogErrorS("Tensors multiplication requires at least 2 dimensions.");



  tensor_x->Sync();
  tensor_w->Sync();

  matmul_forward(device_y, device_x, device_w,
                  linear_layer_dims[0], linear_layer_dims[1],
                  Rdims[0], thread_id);

  
  
  std::vector<float> new_dims = NewDimsOnMult(Ldims, Rdims);

  Tensor *new_tensor = createTensor(device_y, new_dims, resultingDimsProd, false, "");
  new_tensor->AttrNodes(tensor_x, tensor_w, mult_op);
  return new_tensor;
}


extern "C" Tensor *tensor_tensor_add(
                          Tensor *tensor_x, Tensor *tensor_w, Scope_Struct *scope_struct) {

  //std::cout << "Cuda add of\n      L " << tensor_x.name << "  &  R " << tensor_w.name << "\n";
    
  int thread_id = scope_struct->thread_id;

  std::vector<float> Ldims, Rdims;
  Ldims = tensor_x->dims;
  Rdims = tensor_w->dims;
  float *device_x = tensor_x->tensor_ptr;
  float *device_w = tensor_w->tensor_ptr;




  std::vector<float> linear_layer_dims = format_LinearLayer_Dims(Ldims);
  float dims_prod = tensor_x->dims_prod;


  float* device_y = get_from_pool(thread_id, dims_prod, "add");


  tensor_x->Sync();
  tensor_w->Sync();
  cudaStream_t stream = ThreadsStream[thread_id];



  if (Ldims==Rdims)
  {

    int grid_size, block_size;
    std::vector<int> grid_block_mem_sizes = CalculateGridAndBlockSizes(dims_prod);
    grid_size = grid_block_mem_sizes[0];
    block_size = grid_block_mem_sizes[1];
    
    add_forward<<<grid_size, block_size, 0, stream>>>(device_y, device_x, device_w, dims_prod);
    

    Tensor *new_tensor = createTensor(device_y, Ldims, dims_prod, false, "");
    new_tensor->AttrNodes(tensor_x, tensor_w, add_op);
    return new_tensor;
  }

  

  
  if(RemoveLastDim(Ldims)==Rdims||(RemoveLastDim(Ldims)==RemoveLastDim(Rdims)&&Rdims[Rdims.size()-1]==1))
  {
    int grid_size, block_size;
    std::vector<int> grid_block_mem_sizes = CalculateGridAndBlockSizes(dims_prod);
    grid_size = grid_block_mem_sizes[0];
    block_size = grid_block_mem_sizes[1];
    
    broadcast_lastdim_add<<<grid_size, block_size, 0, stream>>>(device_y, device_x, device_w, dims_prod, tensor_x->dims[tensor_x->dims.size()-1]);
    

    Tensor *new_tensor = createTensor(device_y, Ldims, dims_prod, false, "");
    new_tensor->AttrNodes(tensor_x, tensor_w, broadcast_lastdim_add_op);
    return new_tensor;
  }


  if (Ldims!=Rdims)
  {
    LogErrorS("Tried to add tensors of different dimenstions.");
    std::cout << "   Left tensor dims " << "\n   ";
    PrintDims(Ldims);
    std::cout << "\n   Right tensor dims " << "\n   ";
    PrintDims(Rdims);
    std::cout << "\n\n";
    return nullptr;
  }
}


extern "C" Tensor *tensor_tensor_sub(
                          Tensor *tensor_x, Tensor *tensor_w, Scope_Struct *scope_struct) {

  int thread_id = scope_struct->thread_id;

  //std::cout << "Cuda add of\n      L " << tensor_x.name << "  &  R " << tensor_w.name << "\n";
    
  std::vector<float> Ldims, Rdims;
  Ldims = tensor_x->dims;
  Rdims = tensor_w->dims;
  float *device_x = tensor_x->tensor_ptr;
  float *device_w = tensor_w->tensor_ptr;


  std::vector<float> linear_layer_dims = format_LinearLayer_Dims(Ldims);
  float dims_prod = tensor_x->dims_prod;




  float* device_y = get_from_pool(thread_id, dims_prod,"sub");



  int grid_size = dims_prod;
  int block_size = 512;
  
  tensor_x->Sync();
  tensor_w->Sync();
  cudaStream_t stream = ThreadsStream[thread_id];
  sub_forward<<<grid_size, block_size, 0, stream>>>(device_y, device_x, device_w, dims_prod);
  
  
  Tensor *new_tensor = createTensor(device_y, Ldims, dims_prod, false, "");  
  new_tensor->AttrNodes(tensor_x, tensor_w, sub_op);
  return new_tensor;
}


extern "C" Tensor *tensor_tensor_equal(
                          Tensor *tensor_x, Tensor *tensor_w, Scope_Struct *scope_struct) {

  int thread_id = scope_struct->thread_id;
                            
  //std::cout << "Cuda add of\n      L " << tensor_x.name << "  &  R " << tensor_w.name << "\n";
    
  std::vector<float> Ldims, Rdims;
  Ldims = tensor_x->dims;
  Rdims = tensor_w->dims;
  float *device_x = tensor_x->tensor_ptr;
  float *device_w = tensor_w->tensor_ptr;


  std::vector<float> linear_layer_dims = format_LinearLayer_Dims(Ldims);
  float dims_prod = tensor_x->dims_prod;


  float* device_y = get_from_pool(thread_id, dims_prod, "eq");


  int grid_size, block_size; 
  std::vector<int> grid_block_mem_sizes = CalculateGridAndBlockSizes(dims_prod);
  grid_size = grid_block_mem_sizes[0];
  block_size = grid_block_mem_sizes[1];
  
  tensor_x->Sync();
  tensor_w->Sync();
  cudaStream_t stream = ThreadsStream[thread_id];
  equal_forward<<<grid_size, block_size, 0, stream>>>(device_y, device_x, device_w, dims_prod);
  
  
  Tensor *new_tensor = createTensor(device_y, Ldims, dims_prod, false, "");  
  new_tensor->AttrNodes(tensor_x, tensor_w, equal_op);
  return new_tensor;
}


extern "C" Tensor *tensor_tensor_mult(
                          Tensor *tensor_x, Tensor *tensor_w, Scope_Struct *scope_struct) {

  int thread_id = scope_struct->thread_id;

  //std::cout << "      L " << tensor_x.name << "  &  R " << tensor_w.name << "\n";
    
  std::vector<float> Ldims, Rdims;
  Ldims = tensor_x->dims;
  Rdims = tensor_w->dims;
  float *device_x = tensor_x->tensor_ptr;
  float *device_w = tensor_w->tensor_ptr;

  float dims_prod = tensor_x->dims_prod;


  cudaStream_t stream = ThreadsStream[thread_id];
  if (Ldims!=Rdims) //Then broadcast
  { //TODO: change kernel instead
    bool first_iter = true;
    while (Ldims.size()>Rdims.size())
    {
      float tgt_dim_size = Ldims[Rdims.size()];
      float aux_size = DimsProd(Rdims);
      float *aux_tensor, *aux_free;
      cudaMalloc(&aux_tensor, aux_size*tgt_dim_size*sizeof(float));
      cudaMemset(aux_tensor, 0, aux_size*tgt_dim_size*sizeof(float));
      
      int grid_size = dims_prod;
      int block_size = 32;
      size_t shared_mem_size = 2 * block_size / 32 * sizeof(float);
      repeat_interleave_kernel_last_dim<<<grid_size, block_size, shared_mem_size, stream>>>(device_w, aux_tensor, aux_size, tgt_dim_size);

      if (!first_iter)
      {
        aux_free = device_w;
        cudaCheck(cudaFree(aux_free));
      }
      device_w = aux_tensor;
      Rdims.push_back(tgt_dim_size);
      first_iter=false;
    }

    while (Ldims.size()<Rdims.size())
    {
      float tgt_dim_size = Rdims[Ldims.size()];
      float aux_size = DimsProd(Ldims);
      float *aux_tensor, *aux_free;
      cudaMalloc(&aux_tensor, aux_size*tgt_dim_size*sizeof(float));
      cudaMemset(aux_tensor, 0, aux_size*tgt_dim_size*sizeof(float));
      
      int grid_size = dims_prod;
      int block_size = 32;
      size_t shared_mem_size = 2 * block_size / 32 * sizeof(float);
      repeat_interleave_kernel_last_dim<<<grid_size, block_size, shared_mem_size, stream>>>(device_x, aux_tensor, aux_size, tgt_dim_size);

      if (!first_iter)
      {
        aux_free = device_x;
        cudaCheck(cudaFree(aux_free));
      }
      device_x = aux_tensor;
      
      Ldims.push_back(tgt_dim_size);
      
      dims_prod = DimsProd(Ldims);
      first_iter=false;
    }
  }


  float *device_y = get_from_pool(thread_id, dims_prod, "hadamard");


  int grid_size, block_size;
  std::vector<int> grid_block_mem_sizes = CalculateGridAndBlockSizes(dims_prod);
  grid_size = grid_block_mem_sizes[0];
  block_size = grid_block_mem_sizes[1];

  tensor_x->Sync();
  tensor_w->Sync();
  hadamard_kernel<<<grid_size, block_size, 0, stream>>>(device_y, device_x, device_w, dims_prod);
  //PrintTensorF(device_y, 2, 2);



  Tensor *new_tensor = createTensor(device_y, Ldims, dims_prod, false, "");
  new_tensor->AttrNodes(tensor_x, tensor_w, hadamard_op);
  return new_tensor;
}



extern "C" void *tensor_tensor_div(
                          Tensor *tensor_x, Tensor *tensor_w, Scope_Struct *scope_struct) {
                            
  int thread_id = scope_struct->thread_id;
  
  //std::cout << "TENSOR TENSOR DIV" << "\n";
  
  std::vector<float> Ldims, Rdims;
  Ldims = tensor_x->dims;
  Rdims = tensor_w->dims;
  float *device_x = tensor_x->tensor_ptr;
  float *device_w = tensor_w->tensor_ptr;
  float dims_prod, R_dims_prod;
  dims_prod = tensor_x->dims_prod;
  R_dims_prod = tensor_w->dims_prod;


  cudaStream_t stream = ThreadsStream[thread_id];
  if (Ldims!=Rdims) //Then broadcast
  { //TODO: change kernel instead
    bool first_iter = true;
    while (Ldims.size()>Rdims.size())
    {
      float tgt_dim_size = Ldims[Rdims.size()];
      float aux_size = DimsProd(Rdims);
      float *aux_tensor, *aux_free;
      cudaMalloc(&aux_tensor, aux_size*tgt_dim_size*sizeof(float));
      cudaMemset(aux_tensor, 0, aux_size*tgt_dim_size*sizeof(float));
      
      int grid_size = dims_prod;
      int block_size = 32;
      size_t shared_mem_size = 2 * block_size / 32 * sizeof(float);
      repeat_interleave_kernel_last_dim<<<grid_size, block_size, shared_mem_size, stream>>>(device_w, aux_tensor, aux_size, tgt_dim_size);

      if (!first_iter)
      {
        aux_free = device_w;
        cudaCheck(cudaFree(aux_free));
      }
      device_w = aux_tensor;
      Rdims.push_back(tgt_dim_size);
      first_iter=false;
    }

    while (Ldims.size()<Rdims.size())
    {
      float tgt_dim_size = Rdims[Ldims.size()];
      float aux_size = DimsProd(Ldims);
      float *aux_tensor, *aux_free;
      cudaMalloc(&aux_tensor, aux_size*tgt_dim_size*sizeof(float));
      cudaMemset(aux_tensor, 0, aux_size*tgt_dim_size*sizeof(float));
      
      int grid_size = dims_prod;
      int block_size = 32;
      size_t shared_mem_size = 2 * block_size / 32 * sizeof(float);
      repeat_interleave_kernel_last_dim<<<grid_size, block_size, shared_mem_size, stream>>>(device_x, aux_tensor, aux_size, tgt_dim_size);

      if (!first_iter)
      {
        aux_free = device_x;
        cudaCheck(cudaFree(aux_free));
      }
      device_x = aux_tensor;
      
      Ldims.push_back(tgt_dim_size);
      
      dims_prod = DimsProd(Ldims);
      first_iter=false;
    }
  }


  //if (dims_prod!=R_dims_prod)
  //  LogErrorS("Tensors division has tensors of different dimensions.");


  float* device_y = get_from_pool(thread_id, dims_prod,"div");
  


  int grid_size = dims_prod;
  int block_size = 32;
  size_t shared_mem_size = 2 * block_size / 32 * sizeof(float);
  tensor_div<<<grid_size, block_size, shared_mem_size, stream>>>(device_w, device_x, device_y, dims_prod);

  Tensor *new_tensor = createTensor(device_y, Ldims, dims_prod, false, "");  
  new_tensor->AttrNodes(tensor_x, tensor_w, div_op);
  return new_tensor;
}


void hadamard_backward(float *x, float *w, float *dx, float *dw, float *dy, float dims_prod)
{
  //std::cout << "hadamard_backward" <<  "\n";
  int grid_size, block_size;
  std::vector<int> grid_block_mem_sizes = CalculateGridAndBlockSizes(dims_prod);
  grid_size = grid_block_mem_sizes[0];
  block_size = grid_block_mem_sizes[1];

  hadamard_backward_kernel<<<grid_size, block_size, 0, main_stream->stream>>>(x, w, dx, dw, dy, dims_prod);
}