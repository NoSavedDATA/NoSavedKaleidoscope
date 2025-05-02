#pragma once

#include <cuda_runtime.h>

#include "../char_pool/include.h"
#include "../mangler/scope_struct.h"
#include "../tensor/include.h"








void copyChunk(float* d_data, const float* h_data, int offset, float size, cudaStream_t stream);


void tensor_Clean_Up(std::string, void *);