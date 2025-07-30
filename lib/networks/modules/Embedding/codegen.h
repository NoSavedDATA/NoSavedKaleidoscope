#pragma once

#include <cuda_runtime.h>
#include <cublasLt.h>
#include <cublas_v2.h>
#include <curand_kernel.h>
#include <cuda_fp16.h>
#include <cudnn.h>
#include <string>
#include <mma.h>



void embedding_backward(float *inp, int size, float *out,
                     float *dinp, float *dout,
                     void *, DT_tensor *node);