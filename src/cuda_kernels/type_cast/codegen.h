#pragma once

#include "../../tensor/tensor_struct.h"



inline data_type_tensor *float_to_half(data_type_tensor *tensor, int thread_id, cudaStream_t stream);


inline half *float_to_half(float *tensor, int thread_id, int dims_prod, cudaStream_t stream);


inline data_type_tensor *half_to_float(data_type_tensor *tensor, int thread_id, cudaStream_t stream);


inline float *half_to_float(half *tensor, int thread_id, int dims_prod, cudaStream_t stream);


inline float *half_to_float_overwrite(half *tensor, float *float_tensor, int dims_prod, cudaStream_t stream);
