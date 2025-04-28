#pragma once


#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <mma.h>

#include "../../mangler/scope_struct.h"
#include "../../tensor/tensor_struct.h"


void relu_backward(float* inp, float dims_prod, float *y, float* dinp, float* dout, std::string module_name);

void gelu_backward(float* inp, float dims_prod, float *y, float* dinp, float* dout, std::string module_name);

void sigmoid_backward(float* inp, float dims_prod, float *y, float* dinp, float* dout, std::string module_name);

void tanh_backward(float* inp, float dims_prod, float *out, float* dinp, float* dout, std::string module_name);