#pragma once

#include <cuda_runtime.h>


#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <mma.h>



__global__ void to_half(__half *y, const float *x, int dims_prod);
