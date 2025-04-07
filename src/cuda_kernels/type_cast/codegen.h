#pragma once

#include "../../tensor/tensor_struct.h"



inline Tensor *float_to_half(Tensor *tensor, int thread_id, cudaStream_t stream);


inline half *float_to_half(float *tensor, int thread_id, int dims_prod, cudaStream_t stream);


inline Tensor *half_to_float(Tensor *tensor, int thread_id, cudaStream_t stream);


inline float *half_to_float(half *tensor, int thread_id, int dims_prod, cudaStream_t stream);


inline float *half_to_float_overwrite(half *tensor, float *float_tensor, int dims_prod, cudaStream_t stream);
