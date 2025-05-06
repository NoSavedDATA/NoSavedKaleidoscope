#pragma once

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <mma.h>

#include "../tensor/tensor_struct.h"
#include "utils.h"

using namespace nvcuda;

__global__ void btc_mult_kernel(float *out, const float *x, const float *w, const int B, const int Tx, const int Tw, const int C, const int tile_size, const int tile_offset);
__global__ void btc_mult_kernelT(float *out, const float *x, const float *w, const int B, const int Tx, const int Tw, const int C, const int tile_size, const int tile_offset);



extern "C" data_type_tensor *btc_mult(int thread_id, data_type_tensor *x, data_type_tensor*w);
extern "C" data_type_tensor *btc_multT(int thread_id, data_type_tensor *x, data_type_tensor*w);
