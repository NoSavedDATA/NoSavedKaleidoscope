#pragma once

#include <vector>

#include "../mangler/scope_struct.h"
#include "../tensor/tensor_struct.h"
#include "calculate_grids.h"



void scalarmult_backward(float *inp, int dims_prod, float *out,
                     float *dinp, float *dout,
                     std::string module_name, DT_tensor *node);