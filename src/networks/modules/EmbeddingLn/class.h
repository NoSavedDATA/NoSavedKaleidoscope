#pragma once

#include <cuda_runtime.h>
#include <cublasLt.h>
#include <cublas_v2.h>
#include <curand_kernel.h>
#include <cuda_fp16.h>
#include <cudnn.h>
#include <mma.h>

#include "../../../tensor/tensor_struct.h"


class DT_EmbeddingLn
{
  public:
    
    DT_tensor *Book_Tensor, *Weight_Tensor;

    int V, C, OC, B;
    std::string Init, Name;
    float *Book, *dBook, *W, *dW;
    bool changed_descriptors;

    DT_EmbeddingLn(int V, int C, int OC, std::string Init, std::string Name); 
  
  void SetDescriptors(int);
  void SetBackwardDescriptors();
  float *Forward(DT_tensor *, int, int);
  void Backward(float *, float *);
};