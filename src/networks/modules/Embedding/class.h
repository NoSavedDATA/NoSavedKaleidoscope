#pragma once

#include <cuda_runtime.h>
#include <cublasLt.h>
#include <cublas_v2.h>
#include <curand_kernel.h>
#include <cuda_fp16.h>
#include <cudnn.h>
#include <mma.h>

#include "../../../tensor/tensor_struct.h"


class Embedding
{
  public:
    
    int C, OC, B;
    std::string Init, Name;
    float *W, *dW;
    bool changed_descriptors;

    Embedding(int C, int OC, std::string Init, std::string Name); 
  
  void SetDescriptors(int);
  void SetBackwardDescriptors();
  float *Forward(data_type_tensor *, int, int);
  void Backward(float *, float *);
};