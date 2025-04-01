#pragma once

#include <vector>

#include "../tensor/tensor_struct.h"
#include "calculate_grids.h"


extern "C" void *logE(int thread_id, Tensor tensor); 

extern "C" void *logE2(int thread_id, Tensor tensor); 

extern "C" void *clip(int thread_id, Tensor tensor, float _min, float _max);
