#pragma once



#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <thread>
#include <string>
#include <vector>
#include <map>

#include "../cuda_threads/include.h"

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
  custom_op = 56,
};













extern std::vector<int> leaf_ops, loss_ops, gradless_ops, activation_ops, preprocessing_ops, tensor_scalar_ops, custom_ops, weightless_ops;
extern int nn_mode;







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
  std::string scopeless_name = "";
  std::string from_cudnn;
  int op;
  std::string operation;

  Tensor *R_Node, *L_Node, *Sparse_Idx_Tensor;
  bool visited;

  void NewNullTensor();

  void NewTensor(float *new_tensor_ptr, std::vector<float> new_dims, float new_dims_prod,
                 bool new_is_leaf, std::string new_name, CudaStreams *_cuda_stream=nullptr, Loader *_loader=nullptr);
                 
  void NewTensor(half *new_tensor_ptr, std::vector<float> new_dims, float new_dims_prod,
                 bool new_is_leaf, std::string new_name, CudaStreams *_cuda_stream=nullptr, Loader *_loader=nullptr);

  void NewPinned(float *new_tensor_ptr, float *new_cpu_tensor_ptr,
                 std::vector<float> new_dims, float new_dims_prod,
                 bool new_is_leaf, std::string new_name);

  void AttrTensor(float *new_tensor_ptr, std::vector<float> new_dims, float new_dims_prod, CudaStreams *_cuda_stream=nullptr, Loader *_loader=nullptr);

  
  void AttrNodes(Tensor *new_L_Tensor, Tensor *new_R_Tensor, int op_type);

  void AttrLNode(Tensor *new_L_Tensor, int op_type);

  void AttributionBackwardNode(std::string _name, Tensor *new_R_Tensor);

  void SetIsWeight();

  void SetBias(float *b, int b_size);

  void Sync();
};

Tensor *createTensor(float* tensor_ptr, const std::vector<float>& dims, float kDataLen,
                     bool is_leaf, std::string name, CudaStreams *_cuda_stream=nullptr, Loader *_loader=nullptr);
                     
Tensor *createTensorHalf(half* tensor_ptr, const std::vector<float>& dims, float kDataLen,
                     bool is_leaf, std::string name, CudaStreams *_cuda_stream=nullptr, Loader *_loader=nullptr);

Tensor *customOpTensor(float* tensor_ptr, const std::vector<float>& dims, float kDataLen,
                     std::string operation, std::string module_name, Tensor *LTensor, CudaStreams *_cuda_stream=nullptr, Loader *_loader=nullptr);

Tensor *createPinned(float* tensor_ptr, float *tensor_cpu, const std::vector<float>& dims, float kDataLen,
                     std::string name)
                     ;
Tensor *createBackward(std::string name, Tensor *tensor);

Tensor *wrapTensorWithDetached(Tensor* tensor);


bool in_tensor_ptr_vec(Tensor *value, const std::vector<Tensor *>& list);


extern std::map<std::string, Tensor *> NamedTensorsT;
extern std::map<std::string, float *> NamedPinnedTensors;
extern std::map<std::string, std::vector<float>> NamedDims;
extern std::vector<Tensor> TensorsToDelete;
