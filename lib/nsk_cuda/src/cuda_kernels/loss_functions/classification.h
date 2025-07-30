#pragma once


#include "../../../../../src/nsk_cpp.h"
#include "../../tensor/tensor_struct.h"


void CrossEntropyBackward(DT_tensor *L_tensor, DT_tensor *R_tensor, 
                          float *dloss,
                          float scale);



void CrossEntropyIdxBackward(DT_tensor *L_tensor, DT_tensor *R_tensor, 
    float *dloss,
    float scale);


