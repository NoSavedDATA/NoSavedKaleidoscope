#pragma once

#include <vector>

#include "../../tensor/tensor_struct.h"

extern "C" void *dropout(int thread_id, Tensor *tensor, float rate);

void dropout_backward(float *dx, float *mask, float *dy, float dims_prod);
