#pragma once

#include "../mangler/scope_struct.h"
#include "tensor_struct.h"


extern "C" float PrintTensor(Scope_Struct *, char* tensorName);

extern "C" float print_tensor(Tensor tensor);

extern "C" float PrintTensorF(const float *cuda_tensor, int d1, int d2);