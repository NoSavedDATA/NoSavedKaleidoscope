#pragma once

#include <cuda_runtime.h>
#include <cublasLt.h>
#include <cublas_v2.h>
#include <curand_kernel.h>
#include <cuda_fp16.h>
#include <cudnn.h>
#include <mma.h>



__global__ void embeddingln_forward_kernel(const float *x, const float *w,
                      float *out, const int tile_size, const int B, const int batches_per_block, const int C, const int OC); 




__global__ void embeddingln_backward_kernel(const float *x,
                      float *dw, const float *dy, const int tile_size,
                      int B, int C, int OC); 