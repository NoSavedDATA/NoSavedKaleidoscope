#pragma once

#include "../../mangler/scope_struct.h"
#include "../../tensor/tensor_struct.h"


void CrossEntropyBackward(Tensor *L_tensor, Tensor *R_tensor, 
                          float *dloss,
                          float scale);

extern "C" float cross_entropy(Scope_Struct *, Tensor *y_hat, Tensor *y, float scale);


void CrossEntropyIdxBackward(Tensor *L_tensor, Tensor *R_tensor, 
    float *dloss,
    float scale);


extern "C" float cross_entropy_idx(Scope_Struct *, Tensor *y_hat, Tensor *y, float scale);
