#pragma once

#include <cuda_runtime.h>
#include <cublasLt.h>
#include <cublas_v2.h>
#include <curand_kernel.h>
#include <cuda_fp16.h>
#include <cudnn.h>
#include <string>
#include <mma.h>



extern "C" void *EmbeddingForward(char *self, Tensor *tensor_x, int thread_id, char *conv_namec, int is_obj_attr_or_self);



extern "C" float CreateEmbeddingOnDemand(char *tensor_name, char *init,
  float C, float OC);


void embedding_backward(float *x, float *dy, std::string name);
