#include<algorithm>
#include<iostream>
#include<map>
#include<random>
#include<string>
#include<thread>
#include<vector>

#include <cuda_fp16.h>

#include "../common/include.h"
#include "../nsk_cuda/minimal_tensor.h"
#include "tensor_dim_functions.h"
#include "tensor_struct.h"








void DT_tensor::NewNullTensor()
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
  cuda_stream = nullptr;
  loader = nullptr;
  from_cudnn = "";
  is_pinned=false;
  thread_id = 0;
  scalar=1;
  Sparse_Idx_Tensor=nullptr;
}

void DT_tensor::NewTensor(float *new_tensor_ptr, std::vector<int> new_dims, int new_dims_prod,
              bool new_is_leaf, std::string new_name, cudaStream_t cuda_stream, Loader *_loader){
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
  cuda_stream = cuda_stream;
  loader = _loader;
  from_cudnn = "";
  is_pinned=false;
  thread_id = 0;
  scalar=1;
  Sparse_Idx_Tensor=nullptr;
}
void DT_tensor::NewTensor(half *new_tensor_ptr, std::vector<int> new_dims, int new_dims_prod,
              bool new_is_leaf, std::string new_name, cudaStream_t cuda_stream, Loader *_loader){
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
  cuda_stream = cuda_stream;
  loader = _loader;
  from_cudnn = "";
  is_pinned=false;
  thread_id = 0;
  scalar=1;
  Sparse_Idx_Tensor=nullptr;
}

void DT_tensor::NewPinned(float *new_tensor_ptr, float *new_cpu_tensor_ptr,
              std::vector<int> new_dims, int new_dims_prod,
              bool new_is_leaf, std::string new_name){
  tensor_ptr = new_tensor_ptr;
  cpu_tensor_ptr = new_cpu_tensor_ptr;
  dims = new_dims;
  dims_prod = new_dims_prod;
  leaf = new_is_leaf;
  name = new_name;
  weight=false;
  is_pinned=true;
  thread_id = 0;
  Sparse_Idx_Tensor=nullptr;
}

void DT_tensor::AttrTensor(float *new_tensor_ptr, std::vector<int> new_dims, int new_dims_prod, cudaStream_t cuda_stream, Loader *_loader){
  tensor_ptr = new_tensor_ptr;
  dims = new_dims;
  dims_prod = new_dims_prod;
  cuda_stream = cuda_stream;
  loader = _loader;
  is_pinned=false;
}


void DT_tensor::AttrNodes(DT_tensor *new_L_Tensor, DT_tensor *new_R_Tensor, int op_type)
{
  L_Node = new_L_Tensor;
  R_Node = new_R_Tensor;
  op = op_type;
  leaf=false;
  visited=false;
  dy=nullptr;
  weight=false;
  is_pinned=false;
}

void DT_tensor::AttrLNode(DT_tensor *new_L_Tensor, int op_type)
{
  L_Node = new_L_Tensor;
  R_Node=nullptr;
  op = op_type;
  leaf=false;
  visited=false;
  dy=nullptr;
  weight=false;
  is_pinned=false;
}

void DT_tensor::AttributionBackwardNode(std::string _name, DT_tensor *new_R_Tensor)
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
void DT_tensor::SetIsWeight()
{
  weight=true;
  is_pinned=false;
}

void DT_tensor::SetBias(float *b, int b_size)
{
  this->b=b;
  this->b_size=b_size;
  leaf=true;
  is_pinned=false;
}
void DT_tensor::Sync()
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



DT_tensor *createTensor(float* tensor_ptr, const std::vector<int>& dims, int kDataLen,
                     bool is_leaf, std::string name, cudaStream_t cuda_stream, Loader *_loader) {
    DT_tensor *new_tensor = new DT_tensor();
    new_tensor->NewTensor(tensor_ptr, dims, kDataLen, is_leaf, name, cuda_stream, _loader);
    return new_tensor;
}



DT_tensor *createCudaTensor(int thread_id, std::string type, const std::vector<int>& dims,
                     bool is_leaf, std::string name, cudaStream_t cuda_stream, Loader *_loader) {
    DT_tensor *new_tensor = new DT_tensor();
    std::vector<int> cuda_dims;
    
    if (dims.size()>=2)
      cuda_dims = format_LinearLayer_Dims(dims);
    else
      cuda_dims = {1, dims[0]};

    CudaTensor *cuda_tensor = new CudaTensor(thread_id, cuda_dims[0], cuda_dims[1], type);
    
    std::vector<int> new_dims = RemoveLastDim(dims);
    new_dims.push_back(cuda_tensor->aN);
    int kDataLen = cuda_dims[0] * cuda_tensor->aN;
    // std::cout << "kDataLen: " << kDataLen << ".\n";
    new_tensor->NewTensor((float*)cuda_tensor->tensor, new_dims, kDataLen, is_leaf, name, cuda_stream, _loader);
    new_tensor->cuda_tensor = cuda_tensor;

    return new_tensor;
}


DT_tensor *createTensorHalf(half* tensor_ptr, const std::vector<int>& dims, int kDataLen,
                     bool is_leaf, std::string name, cudaStream_t cuda_stream, Loader *_loader) {
    DT_tensor *new_tensor = new DT_tensor();
    new_tensor->NewTensor(tensor_ptr, dims, kDataLen, is_leaf, name, cuda_stream, _loader);
    return new_tensor;
}
DT_tensor *customOpTensor(float* tensor_ptr, const std::vector<int>& dims, int kDataLen,
                     std::string operation, void *network_module, DT_tensor *LTensor, cudaStream_t cuda_stream, Loader *_loader) {
    DT_tensor *new_tensor = new DT_tensor();
    new_tensor->NewTensor(tensor_ptr, dims, kDataLen, false, "", cuda_stream, _loader);
    // new_tensor->scopeless_name = module_name;
    new_tensor->network_module = network_module;
    new_tensor->operation = operation;
    new_tensor->AttrLNode(LTensor, custom_op);
    
    return new_tensor;
}

DT_tensor *createPinned(float* tensor_ptr, float *tensor_cpu, const std::vector<int>& dims, int kDataLen,
                     std::string name) {
    DT_tensor *new_tensor = new DT_tensor();
    new_tensor->NewPinned(tensor_ptr, tensor_cpu, dims, kDataLen, true, name);
    return new_tensor;
}
DT_tensor *createBackward(std::string name, DT_tensor *tensor) {
    DT_tensor *new_tensor = new DT_tensor();
    new_tensor->AttributionBackwardNode(name, tensor);
    return new_tensor;
}
DT_tensor *wrapTensorWithDetached(DT_tensor* tensor) {
    /*
    DT_tensor *new_tensor = new DT_tensor();

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


bool in_tensor_ptr_vec(DT_tensor *value, const std::vector<DT_tensor *>& list) {
    return std::find(list.begin(), list.end(), value) != list.end();
}
