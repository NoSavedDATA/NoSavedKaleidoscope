#pragma once

#include <cuda_runtime.h>
#include <cublasLt.h>
#include <cublas_v2.h>
#include <curand_kernel.h>
#include <cuda_fp16.h>
#include <cudnn.h>
#include <mma.h>



 

extern "C" void *ReluForward(char *self, Tensor *tensor, char *conv_namec, int is_obj_attr_or_self);



void cudnn_relu_backward(float *inp, float *out,
                     float *dinp, 
                     float *dout, std::string conv_name);


extern "C" float CreateBN2dReluOnDemand(char *tensor_name, float C);


extern "C" float CreateReluOnDemand(char *tensor_name);

