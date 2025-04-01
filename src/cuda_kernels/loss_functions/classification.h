#pragma once

#include "../../tensor/tensor_struct.h"


void CrossEntropyBackward(float *y_hat,
                          float *y,
                          int B, int C, 
                          float *dloss,
                          float scale);

extern "C" float cross_entropy(Tensor *y_hat, Tensor *y, float scale);


void CrossEntropyIdxBackward(float *y_hat,
                          float *y,
                          int B, int C, 
                          float *dloss,
                          float scale);


extern "C" float cross_entropy_idx(Tensor *y_hat, Tensor *y, float scale);
