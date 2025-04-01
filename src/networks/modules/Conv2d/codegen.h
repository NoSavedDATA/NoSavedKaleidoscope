#pragma once

#include <cuda_runtime.h>
#include <cublasLt.h>
#include <cublas_v2.h>
#include <curand_kernel.h>
#include <cuda_fp16.h>
#include <cudnn.h>
#include <mma.h>



void conv2d_backward(float *inp,  float *weight,
                     float *dinp, float *dw,
                     float *dout, std::string conv_name);


extern "C" void *ConvForward2d(char *self, Tensor *tensor, int thread_id, char *conv_namec, int is_obj_attr_or_self);


extern "C" float CreateConv2dOnDemand(char *tensor_name, char *init,
                                      float C, float OC, float ks, float stride, float padding);
