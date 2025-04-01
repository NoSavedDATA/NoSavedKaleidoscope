#pragma once

#include <cuda_runtime.h>
#include <cublasLt.h>
#include <cublas_v2.h>
#include <curand_kernel.h>
#include <cuda_fp16.h>
#include <cudnn.h>
#include <mma.h>



void mhsa_backward(float *x, float *dx, float *dy, std::string name);


extern "C" void *MHSAForward(char *self, Tensor *tensor, int thread_id, char *conv_namec, int is_obj_attr_or_self);


extern "C" float CreateMHSAOnDemand(char *tensor_name, char *init,
                                      float nh, float C, float T, int_vec *notators);
