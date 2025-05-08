#pragma once

#include <cuda_runtime.h>
#include <cublasLt.h>
#include <cublas_v2.h>
#include <curand_kernel.h>
#include <cuda_fp16.h>
#include <cudnn.h>
#include <mma.h>

#include "../../../tensor/tensor_struct.h"


class MaxPool2dCPP
{
  public:
    // Forward
    cudnnTensorDescriptor_t input_desc, output_desc;
    cudnnPoolingDescriptor_t pooling_desc;
    
    std::string Type;
    int ks, stride, padding, out_H, out_W;
    int B = 0;
    int C = 0;
    int H = 0;
    int W = 0;

    MaxPool2dCPP(int ks, int stride, int padding, std::string Type);

  


  void SetDescriptors(int, int, int, int, DT_tensor *);
  float *Forward(DT_tensor *, int, int, int, int, int);
  void Backward(float *, float *, float *, float *);

};