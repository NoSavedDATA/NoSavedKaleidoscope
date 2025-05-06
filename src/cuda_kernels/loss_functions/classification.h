#pragma once

#include "../../mangler/scope_struct.h"
#include "../../tensor/tensor_struct.h"


void CrossEntropyBackward(data_type_tensor *L_tensor, data_type_tensor *R_tensor, 
                          float *dloss,
                          float scale);

extern "C" float cross_entropy(Scope_Struct *, data_type_tensor *y_hat, data_type_tensor *y, float scale);


void CrossEntropyIdxBackward(data_type_tensor *L_tensor, data_type_tensor *R_tensor, 
    float *dloss,
    float scale);


extern "C" float cross_entropy_idx(Scope_Struct *, data_type_tensor *y_hat, data_type_tensor *y, float scale);
