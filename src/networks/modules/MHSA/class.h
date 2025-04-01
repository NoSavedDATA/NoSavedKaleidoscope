#pragma once

#include <cuda_runtime.h>
#include <cublasLt.h>
#include <cublas_v2.h>
#include <curand_kernel.h>
#include <cuda_fp16.h>
#include <cudnn.h>
#include <mma.h>

#include "../../../notators/notators.h"
#include "../../../tensor/tensor_struct.h"


class MHSA
{
  public:
    
    int B, T, maxT, nh, C, d, B_back, T_back;
    int M, Br, Bc, Tr, Tc;
    int Br_back, Bc_back, Tr_back, Tc_back;
    std::string Init, Name;
    float *W, *W_proj, *l, *qkv, *out, *qkv_back, *out_back, *l_back, *dW, *dW_proj;
    bool first_backward, changed_descriptors, _fp32, _fp32_back, _causal;
    int_vec *Notators;

    MHSA(int nh, int C, int maxT, std::string Init, int_vec *Notators, std::string Name);
  
  float *Forward(Tensor *, int, int, int);
  void SetDescriptors(int, int, int);
  void Backward(float *, float *, float *);
  void SetBackwardDescriptors();
  void FirstBackward();
};