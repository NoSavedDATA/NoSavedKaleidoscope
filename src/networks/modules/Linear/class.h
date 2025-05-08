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



class LinearCPP
{
  public:
    
    int B, C, OC;
    std::string Init, Name;
    float *W, *dW;
    bool first_backward, changed_descriptors;

    std::vector<std::string> Notes;
    bool _fp32;

    LinearCPP(int C, int OC, std::string Init, std::vector<std::string> Notes, std::string Name);
  
  float *Forward(DT_tensor *, int);
  void SetDescriptors(int, int);
  void Backward(float *, float *, float *);
  void SetBackwardDescriptors();
  void FirstBackward();
};