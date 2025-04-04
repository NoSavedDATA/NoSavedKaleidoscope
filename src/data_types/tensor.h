#pragma once

#include <cuda_runtime.h>

#include "../tensor/include.h"
#include "../char_pool/include.h"


extern "C" float tensor_Create(char *tensor_name, char *scopeless_name, float init_val, AnyVector *notes_vector, int thread_id, char *scope);

extern "C" void *LoadTensor(char *tensor_name);


extern "C" void *gpu(int thread_id, Tensor *tensor, Tensor *pinned_tensor);
extern "C" float gpuw(int thread_id, Tensor *tensor, Tensor *pinned_tensor, float idx);

extern "C" float cpu(int thread_id, Tensor *tensor);


extern "C" float cpu_idx(Tensor *tensor, float idx);


extern "C" void *randu_like(int thread_id, Tensor tensor);


void copyChunk(float* d_data, const float* h_data, int offset, float size, cudaStream_t stream);

extern "C" float write_zerosw(Tensor *tensor, float worker_idx);


extern "C" void *view(int thread_id, Tensor *tensor, float first_dim, ...);


extern "C" void *NewVecToTensor(int thread_id, float first_dim, ...);

