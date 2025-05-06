#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <mma.h>


#include "../tensor/include.h"






void mean_over_semilast_dim_backward(float *dx, float *dy, data_type_tensor *node);





void gather_last_dim_backward(float *dx, float *dy, data_type_tensor *node);


inline void transpose(data_type_tensor *tensor, int thread_id, cudaStream_t stream); 
