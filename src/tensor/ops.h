#pragma once

#include "tensor_struct.h"



// Copies a pinned_tensor's reserved memory into a tensor.
extern "C" float AttrTensorNoFree(char *tensor_name, Tensor *tensor, int thread_id);

extern "C" float AttrTensorOnIdx(char *tensor_name, Tensor *tensor, float idx_at, int thread_id);

extern "C" float AttrTensorOnIdxTensor(char *tensor_name, char *idx_tensor_name, Tensor *R_tensor, int thread_id);

extern "C" void AttrPinnedOnIdx(char *tensor_name, float val, float idx_at);

extern "C" float AttrPinnedFromTensorOnIdx(char *tensor_name, Tensor *Rtensor, int thread_id, float first_idx, ...);

extern "C" void *IdxTensor(char *tensor_name, char *scope, int thread_id, float first_idx, ...);


extern "C" void *IdxTensorWithTensor(char *tensor_name, char *idx_tensor_name, int thread_id);

