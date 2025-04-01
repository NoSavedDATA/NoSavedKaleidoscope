#pragma once

#include <cuda_runtime.h>
#include <cublasLt.h>
#include <cublas_v2.h>
#include <curand_kernel.h>
#include <cuda_fp16.h>
#include <cudnn.h>
#include <mma.h>



extern "C" void *MaxPoolForward2d(char *self, Tensor *tensor, int thread_id, char *conv_namec, int is_obj_attr_or_self);


void maxpool2d_backward(float *inp,  float *out,
                     float *dinp,
                     float *dout, std::string conv_name);

extern "C" float CreateMaxPool2dOnDemand(char *tensor_name, char *type, float ks, float stride, float padding);
