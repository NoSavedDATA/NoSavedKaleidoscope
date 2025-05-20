#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <mma.h>


#include "../tensor/include.h"







void mean_over_semilast_dim_backward(float *inp, int size, float *out,
                     float *dinp, float *dout,
                     std::string module_name, DT_tensor *node);


void gather_last_dim_backward(float *inp, int dims_prod, float *out,
                     float *dinp, float *dout,
                     std::string module_name, DT_tensor *node);


inline void transpose(DT_tensor *tensor, int thread_id, cudaStream_t stream); 
