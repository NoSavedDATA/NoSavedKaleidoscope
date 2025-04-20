#pragma once

#include "../../mangler/scope_struct.h"
#include "../../tensor/tensor_struct.h"


void CrossEntropyBackward(float *y_hat,
                          float *y,
                          int B, int C, 
                          float *dloss,
                          float scale);

extern "C" float cross_entropy(Scope_Struct *, Tensor *y_hat, Tensor *y, float scale);


void CrossEntropyIdxBackward(float *y_hat,
                          float *y,
                          int B, int C, 
                          float *dloss,
                          float scale);


extern "C" float cross_entropy_idx(Scope_Struct *, Tensor *y_hat, Tensor *y, float scale);
