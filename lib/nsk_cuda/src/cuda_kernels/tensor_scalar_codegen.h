#pragma once

#include <vector>

#include "../../../src/nsk_cpp.h"
#include "../tensor/tensor_struct.h"
#include "calculate_grids.h"



void scalarmult_backward(float *inp, int dims_prod, float *out,
                     float *dinp, float *dout,
                     void *, DT_tensor *node);