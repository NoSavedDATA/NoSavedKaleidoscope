#pragma once


#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <mma.h>

#include "../tensor/tensor_struct.h"


extern "C" void *onehot(int thread_id, Tensor *tensor, float num_classes);


extern "C" float priority_sample(int thread_id, Tensor *tensor, float max_idx, float seed);


extern "C" float priority_sample_val(int thread_id, Tensor *tensor, float max_idx, float seed);



extern "C" float importance_sample_idx(int thread_id, Tensor *tensor, float max_idx, float alpha, float beta, float seed);


extern "C" float importance_sample_weight(int thread_id, Tensor *tensor, float max_idx, float alpha, float beta, float seed);

extern "C" void *tmax(int thread_id, Tensor *tensor, float first_dim, ...);



extern "C" void *argmax(int thread_id, Tensor *tensor, float first_dim, ...);


extern "C" void *topk(int thread_id, Tensor tensor, float k);
