#pragma once

#include "../mangler/scope_struct.h"
#include "tensor_struct.h"


extern "C" float PrintTensor(Scope_Struct *scope_struct, char* tensorName);


extern "C" float PrintTensorF(const float *cuda_tensor, int d1, int d2);

extern "C" float PrintTensorI8(const int8_t *cuda_tensor, int d1, int d2);