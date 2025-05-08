#pragma once

#include <vector>

#include "../mangler/scope_struct.h"
#include "../tensor/tensor_struct.h"
#include "calculate_grids.h"

extern "C" void *tensor_scalar_mult(DT_tensor *tensor, float R, Scope_Struct *scope_struct);
  
  
extern "C" void *tensor_scalar_div(DT_tensor tensor, float R, Scope_Struct *scope_struct);
  
// extern "C" void *CudaReverseScalarDiv(DT_tensor tensor, float R, Scope_Struct *scope_struct);

extern "C" void *tensor_scalar_add(DT_tensor *tensor, float R, Scope_Struct *scope_struct); 

extern "C" void *tensor_scalar_sub(DT_tensor *tensor, float R, Scope_Struct *scope_struct); 

extern "C" void *tensor_scalar_equal(DT_tensor tensor, float R, Scope_Struct *scope_struct); 
extern "C" void *tensor_scalar_diff(DT_tensor tensor, float R, Scope_Struct *scope_struct);
extern "C" void *tensor_scalar_minor(DT_tensor tensor, float R, Scope_Struct *scope_struct); 
extern "C" void *tensor_scalar_minor_eq(DT_tensor tensor, float R, Scope_Struct *scope_struct);
extern "C" void *tensor_scalar_higher(DT_tensor tensor, float R, Scope_Struct *scope_struct);
extern "C" void *tensor_scalar_higher_eq(DT_tensor tensor, float R, Scope_Struct *scope_struct);


void scalarmult_backward(float *dx, float *dy, float scalar, float dims_prod);
