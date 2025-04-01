#pragma once


#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <mma.h>

#include "../../tensor/tensor_struct.h"


extern "C" void *relu(int thread_id, Tensor *tensor);


void relu_backward(float* inp, float dims_prod, float* dinp, float* dout); 


void gelu_backward(const float* inp, float dims_prod, float* dinp, const float* dout); 
  
extern "C" void *gelu(int thread_id, Tensor *tensor);
  


void sigmoid_backward(const float* out, float dims_prod, float* dinp, const float* dout);

extern "C" void *sigmoid(int thread_id, Tensor *tensor);


void tanh_backward(const float* out, float dims_prod, float* dinp, const float* dout); 

extern "C" void *_tanh(int thread_id, Tensor *tensor);


extern "C" void *softmax(int thread_id, Tensor *tensor);
