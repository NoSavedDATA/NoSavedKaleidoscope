#pragma once

#include <cuda_runtime.h>

#include "../char_pool/include.h"
#include "../mangler/scope_struct.h"
#include "../tensor/include.h"


extern "C" void *tensor_Load(char *, Scope_Struct *);
extern "C" float tensor_Store(char *, Tensor *, Scope_Struct *);




extern "C" void *gpu(Scope_Struct *scope_struct, Tensor *tensor, Tensor *pinned_tensor);
extern "C" float tensor_gpuw(Scope_Struct *, Tensor *, Tensor *, float);

extern "C" float cpu(Scope_Struct *scope_struct, Tensor *tensor);


extern "C" float cpu_idx(Scope_Struct *scope_struct, Tensor *tensor, float idx);


extern "C" void *randu_like(Scope_Struct *scope_struct, Tensor tensor);


void copyChunk(float* d_data, const float* h_data, int offset, float size, cudaStream_t stream);

extern "C" float write_zerosw(Tensor *tensor, float worker_idx);


extern "C" void *view(Scope_Struct *scope_struct, Tensor *, float first_dim, ...);


extern "C" void *NewVecToTensor(Scope_Struct *scope_struct, float first_dim, ...);


extern "C" float tensor_CalculateIdx(char *tensor_name, float first_idx, ...);
