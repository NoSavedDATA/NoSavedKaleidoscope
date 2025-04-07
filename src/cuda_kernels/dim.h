#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <mma.h>


#include "../tensor/include.h"




extern "C" void *repeat_interleave(int thread_id, Tensor tensor, float repeats, float dim);

//TODO: mean over axis
extern "C" void *mean(int thread_id, Tensor *tensor, float first_dim, ...);


void mean_over_semilast_dim_backward(float *dx, float *dy, Tensor *node);



extern "C" void *sum(int thread_id, Tensor tensor, float first_dim, ...);


extern "C" void *prod(int thread_id, Tensor tensor, float first_dim, ...);



extern "C" void *gather(int thread_id, Tensor *tensor, Tensor *idx_tensor, float dim);


void gather_last_dim_backward(float *dx, float *dy, Tensor *node);


inline void transpose(Tensor *tensor, int thread_id, cudaStream_t stream); 
