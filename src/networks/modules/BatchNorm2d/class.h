#pragma once

#include <cuda_runtime.h>
#include <cublasLt.h>
#include <cublas_v2.h>
#include <curand_kernel.h>
#include <cuda_fp16.h>
#include <cudnn.h>
#include <mma.h>

#include "../../../tensor/tensor_struct.h"


class BatchNorm2dCPP
{
  public:
    cudnnTensorDescriptor_t input_desc, output_desc, scale_bias_mean_var_desc;
    
    bool first_backward=true;

    float *dW, *dB;
    float* scale=nullptr;
    float* bias=nullptr;
    float* running_mean=nullptr;
    float* running_var=nullptr;
    float* saved_mean=nullptr;
    float* saved_var=nullptr;
    float* dscale, dbias;
    int B = 0;
    int C;
    int H = 0;
    int W = 0;
    std::string Name;

    BatchNorm2dCPP(int C, std::string Name);

  
  void SetDescriptors(int, int, int, data_type_tensor *);
  void InitMovingAverages();
  float *Forward(data_type_tensor *, int, int, int, int, int);
  void Backward(float *, float *, float *);
  void FirstBackward();

};