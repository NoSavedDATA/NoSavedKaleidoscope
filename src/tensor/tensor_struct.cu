#include<string>
#include<map>
#include<vector>
#include<iostream>
#include<algorithm>
#include<random>
#include<thread>

#include <cuda_fp16.h>

#include "../common/include.h"
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
  from_grad_or_load=false;
  cuda_stream = nullptr;
  loader = nullptr;
  from_cudnn = "";
  is_pinned=false;
  thread_id = 0;
  scalar=1;
  Sparse_Idx_Tensor=nullptr;
}

void DT_tensor::NewTensor(float *new_tensor_ptr, std::vector<float> new_dims, float new_dims_prod,
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
void DT_tensor::NewTensor(half *new_tensor_ptr, std::vector<float> new_dims, float new_dims_prod,
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

void DT_tensor::NewPinned(float *new_tensor_ptr, float *new_cpu_tensor_ptr,
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

void DT_tensor::AttrTensor(float *new_tensor_ptr, std::vector<float> new_dims, float new_dims_prod, CudaStreams *_cuda_stream, Loader *_loader){
  tensor_ptr = new_tensor_ptr;
  dims = new_dims;
  dims_prod = new_dims_prod;
  cuda_stream = _cuda_stream;
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
  from_grad_or_load = ((from_grad_or_load||new_L_Tensor->from_grad_or_load||new_R_Tensor->from_grad_or_load)&&!in_int(op, gradless_ops));
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
  from_grad_or_load = ((from_grad_or_load||new_L_Tensor->from_grad_or_load)&&!in_int(op, gradless_ops));
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
  from_grad_or_load=true;
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



DT_tensor *createTensor(float* tensor_ptr, const std::vector<float>& dims, float kDataLen,
                     bool is_leaf, std::string name, CudaStreams *_cuda_stream, Loader *_loader) {
    DT_tensor *new_tensor = new DT_tensor();
    new_tensor->NewTensor(tensor_ptr, dims, kDataLen, is_leaf, name, _cuda_stream, _loader);
    return new_tensor;
}
DT_tensor *createTensorHalf(half* tensor_ptr, const std::vector<float>& dims, float kDataLen,
                     bool is_leaf, std::string name, CudaStreams *_cuda_stream, Loader *_loader) {
    DT_tensor *new_tensor = new DT_tensor();
    new_tensor->NewTensor(tensor_ptr, dims, kDataLen, is_leaf, name, _cuda_stream, _loader);
    return new_tensor;
}
DT_tensor *customOpTensor(float* tensor_ptr, const std::vector<float>& dims, float kDataLen,
                     std::string operation, std::string module_name, DT_tensor *LTensor, CudaStreams *_cuda_stream, Loader *_loader) {
    DT_tensor *new_tensor = new DT_tensor();
    new_tensor->NewTensor(tensor_ptr, dims, kDataLen, false, "", _cuda_stream, _loader);
    new_tensor->scopeless_name = module_name;
    new_tensor->operation = operation;
    new_tensor->AttrLNode(LTensor, custom_op);
    
    return new_tensor;
}

DT_tensor *createPinned(float* tensor_ptr, float *tensor_cpu, const std::vector<float>& dims, float kDataLen,
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
