#pragma once


#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <mma.h>

#include "../../mangler/scope_struct.h"
#include "../../tensor/tensor_struct.h"


extern "C" void *relu(Scope_Struct *scope_struct, Tensor *tensor);



void relu_backward(float* inp, float dims_prod, float *y, float* dinp, float* dout, std::string module_name);


void gelu_backward(const float* inp, float dims_prod, float* dinp, const float* dout); 
  
extern "C" void *gelu(Scope_Struct *scope_struct, Tensor *tensor);
  


void sigmoid_backward(const float* out, float dims_prod, float* dinp, const float* dout);

extern "C" void *sigmoid(Scope_Struct *scope_struct, Tensor *tensor);


void tanh_backward(const float* out, float dims_prod, float* dinp, const float* dout); 

extern "C" void *_tanh(Scope_Struct *scope_struct, Tensor *tensor);


extern "C" void *softmax(Scope_Struct *scope_struct, Tensor *tensor);
