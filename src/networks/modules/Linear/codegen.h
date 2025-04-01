#pragma once

#include <cuda_runtime.h>
#include <cublasLt.h>
#include <cublas_v2.h>
#include <curand_kernel.h>
#include <cuda_fp16.h>
#include <cudnn.h>
#include <mma.h>
#include <string>




void linear_backward(float *x, float *dx, float *dy, std::string name);




extern "C" void *LinearForward(char *self, Tensor *tensor, int thread_id, char *conv_namec, int is_obj_attr_or_self);




extern "C" float CreateLinearOnDemand(char *tensor_name, char *init,
                                      float C, float OC, int_vec *Notators);
