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



class Linear
{
  public:
    
    int B, C, OC;
    std::string Init, Name;
    float *W, *dW;
    bool first_backward, changed_descriptors;

    int_vec *Notators;
    bool _fp32;

    Linear(int C, int OC, std::string Init, int_vec *Notators, std::string Name);
  
  float *Forward(Tensor *, int);
  void SetDescriptors(int, int);
  void Backward(float *, float *, float *);
  void SetBackwardDescriptors();
  void FirstBackward();
};