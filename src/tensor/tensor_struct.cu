#include <algorithm>
#include <cstdarg>
#include <cassert>
#include <cctype>
#include <cstring>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <memory>
#include <string>
#include <iostream>
#include <numeric>
#include <utility>
#include <vector>
#include <iomanip>
#include <math.h>
#include <fenv.h>
#include <tuple>
#include <glob.h>
#include <chrono>
#include <thread>
#include <random>
#include <float.h>
#include <fstream>
#include <sstream>
#include <filesystem>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#include "../common/extension_functions.cu"
#include "tensor_struct.h"




CudaStreams *AllocateStream(int line)
{
  int free_stream = FirstNonzero(open_streams, num_parallel_streams);
  if (free_stream<0)
  LogErrorCodegen("Failed to allocate a cuda stream. Probably loading too many different tensors.", line);
  open_streams[free_stream] = 0;
  //std::cout << "Allocating stream " << free_stream << "\n";
  return parallel_streams[free_stream];
}


void SynchronizeStream(CudaStreams *cuda_stream)
{
  //std::cout << "Synchronizing stream " << cuda_stream->idx << "\n";
  cudaStreamSynchronize(cuda_stream->stream);
  open_streams[cuda_stream->idx] = 1;
}








void Loader::Load(float *tensor_ptr, const float *tensor_cpu, int all_dims_prod) {

  float quotient = std::floor(all_dims_prod / ASYNC_LOADER_THREADS);
  float remainder = all_dims_prod % ASYNC_LOADER_THREADS;


  std::vector<float> dims_prods;

  for(int i=0; i<ASYNC_LOADER_THREADS-1; i++)
    dims_prods.push_back(quotient);
  dims_prods.push_back(quotient+remainder);


  float offset, size;
  offset = 0;
  for(int i=0; i<ASYNC_LOADER_THREADS; i++)
  {
    size = dims_prods[i];
    CudaStreams *cuda_stream = AllocateStream(0);

    //copyChunk(tensor_ptr, tensor_cpu, offset, size, cuda_stream->stream);
    //threads.push_back(std::thread(copyChunk, tensor_ptr, tensor_cpu, offset, size, cuda_stream->stream));

    cudaMemcpyAsync(tensor_ptr + (int)offset, tensor_cpu + (int)offset, size*sizeof(float), cudaMemcpyHostToDevice, cuda_stream->stream);

    streams.push_back(cuda_stream);
    offset += size;
  }
}
    
void Loader::Sync()
{
  for(int i=0; i<ASYNC_LOADER_THREADS; i++)
  {
    SynchronizeStream(streams[i]);
    //threads[i].join();
  }
  streams.clear();
  //threads.clear();
}



void Tensor::NewNullTensor()
{
  tensor_ptr = nullptr;
  dims = {0};
  dims_prod = 0;
  cpu_tensor_ptr = nullptr;
  L_Node=nullptr;
  R_Node=nullptr;
  dy=nullptr;
  visited=false;
  weight=false;
  from_grad_or_load=false;
  cuda_stream = nullptr;
  loader = nullptr;
  from_cudnn = "";
  is_pinned=false;
  thread_id = 0;
  scalar=1;
  Sparse_Idx_Tensor=nullptr;
}

void Tensor::NewTensor(float *new_tensor_ptr, std::vector<float> new_dims, float new_dims_prod,
              bool new_is_leaf, std::string new_name, CudaStreams *_cuda_stream, Loader *_loader){
  tensor_ptr = new_tensor_ptr;
  dims = new_dims;
  dims_prod = new_dims_prod;
  leaf = new_is_leaf;
  name = new_name;
  cpu_tensor_ptr = nullptr;
  op=leaf;
  L_Node=nullptr;
  R_Node=nullptr;
  dy=nullptr;
  visited=false;
  weight=false;
  op=tensor_leaf;
  from_grad_or_load=false;
  cuda_stream = _cuda_stream;
  loader = _loader;
  from_cudnn = "";
  is_pinned=false;
  thread_id = 0;
  scalar=1;
  Sparse_Idx_Tensor=nullptr;
}
void Tensor::NewTensor(half *new_tensor_ptr, std::vector<float> new_dims, float new_dims_prod,
              bool new_is_leaf, std::string new_name, CudaStreams *_cuda_stream, Loader *_loader){
  half_ptr = new_tensor_ptr;
  dims = new_dims;
  dims_prod = new_dims_prod;
  leaf = new_is_leaf;
  name = new_name;
  cpu_tensor_ptr = nullptr;
  op=leaf;
  L_Node=nullptr;
  R_Node=nullptr;
  dy=nullptr;
  visited=false;
  weight=false;
  op=tensor_leaf;
  from_grad_or_load=false;
  cuda_stream = _cuda_stream;
  loader = _loader;
  from_cudnn = "";
  is_pinned=false;
  thread_id = 0;
  scalar=1;
  Sparse_Idx_Tensor=nullptr;
}

void Tensor::NewPinned(float *new_tensor_ptr, float *new_cpu_tensor_ptr,
              std::vector<float> new_dims, float new_dims_prod,
              bool new_is_leaf, std::string new_name){
  tensor_ptr = new_tensor_ptr;
  cpu_tensor_ptr = new_cpu_tensor_ptr;
  dims = new_dims;
  dims_prod = new_dims_prod;
  leaf = new_is_leaf;
  name = new_name;
  weight=false;
  from_grad_or_load=true;
  is_pinned=true;
  thread_id = 0;
  Sparse_Idx_Tensor=nullptr;
}

void Tensor::AttrTensor(float *new_tensor_ptr, std::vector<float> new_dims, float new_dims_prod, CudaStreams *_cuda_stream, Loader *_loader){
  tensor_ptr = new_tensor_ptr;
  dims = new_dims;
  dims_prod = new_dims_prod;
  cuda_stream = _cuda_stream;
  loader = _loader;
  is_pinned=false;
}


void Tensor::AttrNodes(Tensor *new_L_Tensor, Tensor *new_R_Tensor, int op_type)
{
  L_Node = new_L_Tensor;
  R_Node = new_R_Tensor;
  op = op_type;
  leaf=false;
  visited=false;
  dy=nullptr;
  weight=false;
  from_grad_or_load = ((from_grad_or_load||new_L_Tensor->from_grad_or_load||new_R_Tensor->from_grad_or_load)&&!in_int(op, gradless_ops));
  is_pinned=false;
}

void Tensor::AttrLNode(Tensor *new_L_Tensor, int op_type)
{
  L_Node = new_L_Tensor;
  R_Node=nullptr;
  op = op_type;
  leaf=false;
  visited=false;
  dy=nullptr;
  weight=false;
  from_grad_or_load = ((from_grad_or_load||new_L_Tensor->from_grad_or_load)&&!in_int(op, gradless_ops));
  is_pinned=false;
}

void Tensor::AttributionBackwardNode(std::string _name, Tensor *new_R_Tensor)
{
  name = _name;
  R_Node = new_R_Tensor;
  op = attribution;
  leaf=false;
  visited=false;
  
  L_Node=nullptr;
  dy=nullptr;
  weight=false;
  is_pinned=false;
}
void Tensor::SetIsWeight()
{
  weight=true;
  from_grad_or_load=true;
  is_pinned=false;
}
void Tensor::SetBias(float *b, int b_size)
{
  this->b=b;
  this->b_size=b_size;
  leaf=true;
  is_pinned=false;
}
void Tensor::Sync()
{
  if(loader!=nullptr)
  {
    loader->Sync();
    delete loader;
    loader=nullptr;
  }
  if(cuda_stream!=nullptr)
  {
    SynchronizeStream(cuda_stream);
    cuda_stream=nullptr;
  }
}



Tensor *createTensor(float* tensor_ptr, const std::vector<float>& dims, float kDataLen,
                     bool is_leaf, std::string name, CudaStreams *_cuda_stream, Loader *_loader) {
    Tensor *new_tensor = new Tensor();
    new_tensor->NewTensor(tensor_ptr, dims, kDataLen, is_leaf, name, _cuda_stream, _loader);
    return new_tensor;
}
Tensor *createTensorHalf(half* tensor_ptr, const std::vector<float>& dims, float kDataLen,
                     bool is_leaf, std::string name, CudaStreams *_cuda_stream, Loader *_loader) {
    Tensor *new_tensor = new Tensor();
    new_tensor->NewTensor(tensor_ptr, dims, kDataLen, is_leaf, name, _cuda_stream, _loader);
    return new_tensor;
}

Tensor *createPinned(float* tensor_ptr, float *tensor_cpu, const std::vector<float>& dims, float kDataLen,
                     std::string name) {
    Tensor *new_tensor = new Tensor();
    new_tensor->NewPinned(tensor_ptr, tensor_cpu, dims, kDataLen, true, name);
    return new_tensor;
}
Tensor *createBackward(std::string name, Tensor *tensor) {
    Tensor *new_tensor = new Tensor();
    new_tensor->AttributionBackwardNode(name, tensor);
    return new_tensor;
}
Tensor *wrapTensorWithDetached(Tensor* tensor) {
    /*
    Tensor *new_tensor = new Tensor();

    new_tensor->NewNullTensor();
    new_tensor->AttrLNode(tensor, detach_op);
    new_tensor->tensor_ptr = tensor->tensor_ptr;
    new_tensor->dims_prod = tensor->dims_prod;
    new_tensor->dims = tensor->dims;
    
    return new_tensor;
    */
    
    tensor->op = detach_op;
    return tensor;
}


bool in_tensor_ptr_vec(Tensor *value, const std::vector<Tensor *>& list) {
    return std::find(list.begin(), list.end(), value) != list.end();
}
