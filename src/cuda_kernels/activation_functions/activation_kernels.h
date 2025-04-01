#pragma once


#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <mma.h>

#include "../../tensor/tensor_struct.h"

__global__ void relu_forward(float* Z, float* A,
    const float dims_prod); 

__global__ void relu_backward1(float* Z, float* dZ, float* dA,
                                       float N); 

__global__ void gelu_forward_kernel1(const float* inp, float* out, int N);

__global__ void gelu_backward1(float* dinp, const float* inp, const float* dout, int N); 


__global__ void sigmoid_forward_kernel(const float* inp, float* out, int N);

__global__ void sigmoid_backward_kernel(float* dinp, const float* out, const float* dout, int N); 


__global__ void tanh_forward_kernel(const float* inp, float* out, int N);

__global__ void tanh_backward_kernel(float* dinp, const float* out, const float* dout, int N); 


__global__ void softmax_forward_kernel4(const float* inp, float* out, int N, int C); 


__global__ void online_softmax(const float* inp, float* out, int N, int C); 