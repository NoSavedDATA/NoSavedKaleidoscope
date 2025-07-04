#pragma once

#include <cuda_runtime.h>
#include <cublasLt.h>
#include <cublas_v2.h>
#include <curand_kernel.h>
#include <cuda_fp16.h>
#include <cudnn.h>
#include <mma.h>

#include "../../../tensor/tensor_struct.h"


class DT_LSTM
{
  public:
  
    DT_tensor *W_Tensor, *U_Tensor, *Bias_Tensor;

    int C, OC, B, T;
    std::string Init, Name;
    float *W, *U, *x_out, *fused_out, *b, *all_ht, *all_ct, *dW, *dU, *dB, *d_ht, *d_ct, *d_ifoc;
    bool changed_descriptors, first_backward;

    DT_LSTM(int C, int OC, std::string Init, std::string Name);

  
  void SetDescriptors(int, int, int);
  void SetBackwardDescriptors();
  void FirstBackward();
  float *Forward(DT_tensor *, DT_tensor *, DT_tensor *, int, int, int);
  void Backward(float *, float *, float *);

};