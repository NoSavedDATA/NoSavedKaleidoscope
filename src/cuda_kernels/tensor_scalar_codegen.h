#pragma once

#include <vector>

#include "../mangler/scope_struct.h"
#include "../tensor/tensor_struct.h"
#include "calculate_grids.h"

extern "C" void *tensor_scalar_mult(Tensor *tensor, float R, Scope_Struct *scope_struct);
  
  
extern "C" void *tensor_scalar_div(Tensor tensor, float R, Scope_Struct *scope_struct);
  
// extern "C" void *CudaReverseScalarDiv(Tensor tensor, float R, Scope_Struct *scope_struct);

extern "C" void *tensor_scalar_add(Tensor *tensor, float R, Scope_Struct *scope_struct); 

extern "C" void *tensor_scalar_sub(Tensor *tensor, float R, Scope_Struct *scope_struct); 

extern "C" void *tensor_scalar_equal(Tensor tensor, float R, Scope_Struct *scope_struct); 
extern "C" void *tensor_scalar_diff(Tensor tensor, float R, Scope_Struct *scope_struct);
extern "C" void *tensor_scalar_minor(Tensor tensor, float R, Scope_Struct *scope_struct); 
extern "C" void *tensor_scalar_minor_eq(Tensor tensor, float R, Scope_Struct *scope_struct);
extern "C" void *tensor_scalar_higher(Tensor tensor, float R, Scope_Struct *scope_struct);
extern "C" void *tensor_scalar_higher_eq(Tensor tensor, float R, Scope_Struct *scope_struct);


void scalarmult_backward(float *dx, float *dy, float scalar, float dims_prod);
