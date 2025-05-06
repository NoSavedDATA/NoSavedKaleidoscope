#pragma once

#include <cuda_runtime.h>
#include <cublasLt.h>
#include <cublas_v2.h>
#include <curand_kernel.h>
#include <cuda_fp16.h>
#include <cudnn.h>
#include <mma.h>



extern "C" void *BN2dReluForward(char *self, data_type_tensor *tensor, int thread_id, char *conv_namec, int is_obj_attr_or_self);


void bn2drelu_backward(float *inp, float *intermediate, float *out,
                     float *dinp, float *dw, float *db, float *dintermediate,
                     float *dout, std::string conv_name);



extern "C" float CreateBN2dReluOnDemand(char *tensor_name, float C);
