#pragma once

#include <cuda_runtime.h>
#include <cublasLt.h>
#include <cublas_v2.h>
#include <curand_kernel.h>
#include <cuda_fp16.h>
#include <cudnn.h>
#include <mma.h>
#include <string>





void linear_backward(float *inp, int size, float *out,
                     float *dinp, float *dout,
                     void *, DT_tensor *node);


