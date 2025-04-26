#pragma once

#include <cuda_runtime.h>
#include <cublasLt.h>
#include <cublas_v2.h>
#include <curand_kernel.h>
#include <cuda_fp16.h>
#include <cudnn.h>
#include <mma.h>

#include "../../../data_types/codegen_notes.h"
#include "../../../mangler/scope_struct.h"


void conv2d_backward(float *, float, float *,
                     float *, float *,
                     std::string conv_name);





