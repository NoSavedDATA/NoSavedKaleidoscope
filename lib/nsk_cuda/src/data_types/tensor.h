#pragma once

#include <cuda_runtime.h>


#include "../../../src/nsk_cpp.h"




void copyChunk(float* d_data, const float* h_data, int offset, float size, cudaStream_t stream);


void tensor_Clean_Up(void *);