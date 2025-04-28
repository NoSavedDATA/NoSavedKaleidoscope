#pragma once

#include <cuda_runtime.h>
#include <cublasLt.h>
#include <cublas_v2.h>
#include <curand_kernel.h>
#include <cuda_fp16.h>
#include <cudnn.h>
#include <mma.h>



void batchnormd2d_backward(float *inp, 
    float *dinp, float *dw, float *db,
    float *dout, std::string conv_name);