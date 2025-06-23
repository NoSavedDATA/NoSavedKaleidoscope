#pragma once


#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <mma.h>

#include "../../mangler/scope_struct.h"
#include "../../tensor/tensor_struct.h"


void relu_backward(float *inp, int size, float *out,
                     float *dinp, float *dout,
                     void *, DT_tensor *node);

void gelu_backward(float *inp, int size, float *out,
                     float *dinp, float *dout,
                     void *, DT_tensor *node);

void sigmoid_backward(float *inp, int size, float *out,
                     float *dinp, float *dout,
                     void *, DT_tensor *node);

void tanh_backward(float *inp, int size, float *out,
                     float *dinp, float *dout,
                     void *, DT_tensor *node);