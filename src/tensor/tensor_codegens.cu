#pragma once

#ifndef TENSOR
#define TENSOR



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


enum NN_Mode {
  eval_mode = 0,
  training_mode = 1,
};



enum BackwardTypes {
  create_tensor_from_brackets_op=-4,
  create_tensor_op=-3,
  leaf=-2,
  tensor_leaf = -1,
  weight_leaf = 0,
  bias_leaf = 1,
  attribution = 2,
  mult_op = 3,
  conv2d = 4,
  maxpool2d = 5,
  batchnorm2d = 6,
  bn2drelu = 32,
  relu_op = 7,
  cudnn_relu_op = 33,
  gelu_op = 8,
  sigmoid_op = 9,
  tanh_op = 10,
  cross_entropy_op = 11,
  cross_entropy_idx_op = 54,
  mse_op = 44,
  add_op = 12,
  sub_op = 25,
  hadamard_op = 13,
  div_op=26,
  softmax_op = 14,
  onehot_op = 15,
  view_op = 16,
  sum_op = 17,
  mean_op = 18,
  max_op = 19,
  argmax_op = 20,
  topk_op = 21,
  clip_op = 22,
  gpu_op = 23,
  cpu_op = 24,
  equal_op = 27,
  crop_op = 28,
  random_horizontal_flip_op = 29,
  normalize_img_op = 30,
  jitter_op = 46,
  randu_like_op = 31,
  scalar_add_op = 34,
  scalar_sub_op = 35,
  scalar_mult_op = 36,
  scalar_div_op = 37,
  dropout_op = 38,
  sigmoid_add2weights_op = 39,
  lstm_op = 40,
  embedding_op = 41,
  detach_op = 42,
  gather_last_dim_op = 43,
  self_attn_op = 45,
  mse_is_w_op = 47,
  no_op = 48,
  lgrad_op = 49,
  broadcast_lastdim_add_op = 50,
  idx_with_tensor_op = 51,
  mhsa_op = 52,
  mean_over_semilast_dim_op = 53,
  linear_op = 55,
};



enum Notators {
  bias=0,
  fp32=1,
  fp16=2,
  causal=3,
};








struct CudaStreams {
  cudaStream_t stream;
  int idx;
};



extern int ASYNC_LOADER_THREADS;
extern CudaStreams *parallel_streams[];
extern const int num_parallel_streams;
extern cudaEvent_t parallel_events[];
extern std::vector<cudaEvent_t> Registered_Events;
extern int open_streams[];
extern CudaStreams *main_stream, *backward_stream;
extern std::map<int, cudaStream_t> ThreadsStream;
extern std::vector<int> leaf_ops, loss_ops, gradless_ops, activation_ops, preprocessing_ops, tensor_scalar_ops, custom_ops, weightless_ops;
extern int nn_mode;
extern std::map<std::string, int> NotatorsMap;



inline CudaStreams *AllocateStream(int line)
{
  int free_stream = FirstNonzero(open_streams, num_parallel_streams);
  if (free_stream<0)
  LogErrorCodegen("Failed to allocate a cuda stream. Probably loading too many different tensors.", line);
  open_streams[free_stream] = 0;
  //std::cout << "Allocating stream " << free_stream << "\n";
  return parallel_streams[free_stream];
}


inline void SynchronizeStream(CudaStreams *cuda_stream)
{
  //std::cout << "Synchronizing stream " << cuda_stream->idx << "\n";
  cudaStreamSynchronize(cuda_stream->stream);
  open_streams[cuda_stream->idx] = 1;
}








struct Loader {
    std::vector<std::thread> threads;
    std::vector<CudaStreams *> streams;

    inline void Load(float *tensor_ptr, const float *tensor_cpu, int all_dims_prod) {

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
    
    inline void Sync()
    {
      for(int i=0; i<ASYNC_LOADER_THREADS; i++)
      {
        SynchronizeStream(streams[i]);
        //threads[i].join();
      }
      streams.clear();
      //threads.clear();
    }
};



struct Tensor {
  float *tensor_ptr;
  half  *half_ptr;
  float *cpu_tensor_ptr;
  std::vector<float> dims;
  float dims_prod;
  float *b=nullptr;
  float *dy=nullptr;
  float scalar;
  int b_size=0;
  int thread_id;
  bool is_pinned;

  CudaStreams *cuda_stream = nullptr;
  Loader *loader = nullptr;

  bool leaf, weight, from_grad_or_load;
  std::string view_of = "";
  std::string name;
  std::string scopeless_name;
  std::string from_cudnn;
  int op;

  Tensor *R_Node, *L_Node, *Sparse_Idx_Tensor;
  bool visited;

  inline void NewNullTensor()
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

  inline void NewTensor(float *new_tensor_ptr, std::vector<float> new_dims, float new_dims_prod,
                 bool new_is_leaf, std::string new_name, CudaStreams *_cuda_stream=nullptr, Loader *_loader=nullptr){
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
  inline void NewTensor(half *new_tensor_ptr, std::vector<float> new_dims, float new_dims_prod,
                 bool new_is_leaf, std::string new_name, CudaStreams *_cuda_stream=nullptr, Loader *_loader=nullptr){
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

  inline void NewPinned(float *new_tensor_ptr, float *new_cpu_tensor_ptr,
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

  inline void AttrTensor(float *new_tensor_ptr, std::vector<float> new_dims, float new_dims_prod, CudaStreams *_cuda_stream=nullptr, Loader *_loader=nullptr){
    tensor_ptr = new_tensor_ptr;
    dims = new_dims;
    dims_prod = new_dims_prod;
    cuda_stream = _cuda_stream;
    loader = _loader;
    is_pinned=false;
  }

  
  inline void AttrNodes(Tensor *new_L_Tensor, Tensor *new_R_Tensor, int op_type)
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

  inline void AttrLNode(Tensor *new_L_Tensor, int op_type)
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

  inline void AttributionBackwardNode(std::string _name, Tensor *new_R_Tensor)
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
  inline void SetIsWeight()
  {
    weight=true;
    from_grad_or_load=true;
    is_pinned=false;
  }
  inline void SetBias(float *b, int b_size)
  {
    this->b=b;
    this->b_size=b_size;
    leaf=true;
    is_pinned=false;
  }
  inline void Sync()
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
};

inline Tensor *createTensor(float* tensor_ptr, const std::vector<float>& dims, float kDataLen,
                     bool is_leaf, std::string name, CudaStreams *_cuda_stream=nullptr, Loader *_loader=nullptr) {
    Tensor *new_tensor = new Tensor();
    new_tensor->NewTensor(tensor_ptr, dims, kDataLen, is_leaf, name, _cuda_stream, _loader);
    return new_tensor;
}
inline Tensor *createTensorHalf(half* tensor_ptr, const std::vector<float>& dims, float kDataLen,
                     bool is_leaf, std::string name, CudaStreams *_cuda_stream=nullptr, Loader *_loader=nullptr) {
    Tensor *new_tensor = new Tensor();
    new_tensor->NewTensor(tensor_ptr, dims, kDataLen, is_leaf, name, _cuda_stream, _loader);
    return new_tensor;
}

inline Tensor *createPinned(float* tensor_ptr, float *tensor_cpu, const std::vector<float>& dims, float kDataLen,
                     std::string name) {
    Tensor *new_tensor = new Tensor();
    new_tensor->NewPinned(tensor_ptr, tensor_cpu, dims, kDataLen, true, name);
    return new_tensor;
}
inline Tensor *createBackward(std::string name, Tensor *tensor) {
    Tensor *new_tensor = new Tensor();
    new_tensor->AttributionBackwardNode(name, tensor);
    return new_tensor;
}
inline Tensor *wrapTensorWithDetached(Tensor* tensor) {
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


inline bool in_tensor_ptr_vec(Tensor *value, const std::vector<Tensor *>& list) {
    return std::find(list.begin(), list.end(), value) != list.end();
}


extern std::map<std::string, Tensor *> NamedTensorsT;
extern std::map<std::string, float *> NamedPinnedTensors;
extern std::map<std::string, std::vector<float>> NamedDims;
extern std::vector<Tensor> TensorsToDelete;

#endif