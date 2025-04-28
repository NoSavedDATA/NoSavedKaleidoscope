#pragma once

#include <cuda_runtime.h>
#include <cublasLt.h>
#include <cublas_v2.h>
#include <curand_kernel.h>
#include <cuda_fp16.h>
#include <cudnn.h>
#include <mma.h>



void pool2d_backward(float *inp, float size, float *out,
                     float *dinp, float *dout,
                     std::string conv_name);