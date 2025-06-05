#pragma once

#include <cuda_runtime.h>
#include <cublasLt.h>
#include <cublas_v2.h>
#include <curand_kernel.h>
#include <cuda_fp16.h>
#include <cudnn.h>
#include <mma.h>
#include <string>
#include <vector>



#include "../../../notators/notators.h"
#include "../../../tensor/tensor_struct.h"

#include "../../../nsk_cuda/minimal_tensor.h"


class LinearCPP
{
  public:
    
    int B, C, OC;
    std::string Init, Name;
    float *W, *dW;
    int8_t *x8=nullptr, *w8;
    Minimal_Tensor *scale_M=nullptr, *scale_N=nullptr, *scale_K=nullptr;
    bool first_backward, changed_descriptors;

    std::vector<std::string> Notes;
    int precision=0;

    LinearCPP(int C, int OC, std::string Init, std::vector<std::string> Notes, std::string Name);
  
  float *Forward(DT_tensor *, int);
  void SetDescriptors(int, int);
  void Backward(float *, float *, float *);
  void SetBackwardDescriptors();
  void FirstBackward();
};