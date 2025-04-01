#pragma once

#include <cuda_runtime.h>
#include <cublasLt.h>
#include <cublas_v2.h>
#include <curand_kernel.h>
#include <cuda_fp16.h>
#include <cudnn.h>
#include <mma.h>



void lstm_backward(float *x, float *dx, float *dy, std::string name);


extern "C" void *LSTMForward(char *self, Tensor *tensor_x, Tensor *tensor_ht, Tensor *tensor_ct, int thread_id, char *conv_namec, int is_obj_attr_or_self);



extern "C" float CreateLSTMOnDemand(char *tensor_name, char *init,
                                      float C, float OC);
