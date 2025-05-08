#pragma once

#include "../../mangler/scope_struct.h"
#include "../../tensor/tensor_struct.h"


void CrossEntropyBackward(DT_tensor *L_tensor, DT_tensor *R_tensor, 
                          float *dloss,
                          float scale);

extern "C" float cross_entropy(Scope_Struct *, DT_tensor *y_hat, DT_tensor *y, float scale);


void CrossEntropyIdxBackward(DT_tensor *L_tensor, DT_tensor *R_tensor, 
    float *dloss,
    float scale);


extern "C" float cross_entropy_idx(Scope_Struct *, DT_tensor *y_hat, DT_tensor *y, float scale);
