#pragma once

#include <cuda_runtime.h>
#include <cublasLt.h>
#include <cublas_v2.h>
#include <curand_kernel.h>
#include <cuda_fp16.h>
#include <cudnn.h>
#include <mma.h>

#include "../../../tensor/tensor_struct.h"


class Relu
{
  public:
    cudnnTensorDescriptor_t input_desc, output_desc;
    cudnnActivationDescriptor_t activation_desc;


    int B = 0;
    int C = 0;
    int H = 0;
    int W = 0;
    std::string Name;

    Relu(std::string Name);

  
  void SetDescriptors(int, int, int, int, Tensor *);
  float *Forward(Tensor *, int, int, int, int);
  void Backward(float *, float *, float *, float *);

};