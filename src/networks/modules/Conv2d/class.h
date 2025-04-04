#pragma once

#include <cuda_runtime.h>
#include <cublasLt.h>
#include <cublas_v2.h>
#include <curand_kernel.h>
#include <cuda_fp16.h>
#include <cudnn.h>
#include <mma.h>

#include "../../../tensor/tensor_struct.h"


class Conv2d
{
  public:
    // Forward
    cudnnTensorDescriptor_t input_desc, output_desc;
    cudnnFilterDescriptor_t filter_desc;
    cudnnConvolutionDescriptor_t conv_desc;
    cudnnConvolutionFwdAlgo_t fwd_algo;  

    // Weight backward grad
    cudnnConvolutionBwdFilterAlgo_t w_bwd_algo;
    // Input backward grad
    cudnnConvolutionBwdDataAlgo_t y_bwd_algo;

    std::size_t workspace_size, workspace_size_w_back, workspace_size_y_back;
    float *d_workspace, *d_workspace_w_back, *d_workspace_y_back;


    float* d_filter=nullptr;
    float* d_filter_g=nullptr;
    int C, OC, ks, stride, padding, out_H, out_W;
    int B = 0;
    int H = 0;
    int W = 0;
    std::string Init, Name;

    Conv2d(int C, int OC, int ks, int stride, int padding, std::string Init, std::string Name); 


  void SetDescriptors(int, int, int, Tensor *tensor);
  void InitFilters();
  float *Forward(Tensor *, int, int, int, int);
  void Backward(float *, float *, float *, float *);
};