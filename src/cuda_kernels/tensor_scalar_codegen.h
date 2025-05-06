#pragma once

#include <vector>

#include "../mangler/scope_struct.h"
#include "../tensor/tensor_struct.h"
#include "calculate_grids.h"

extern "C" void *tensor_scalar_mult(data_type_tensor *tensor, float R, Scope_Struct *scope_struct);
  
  
extern "C" void *tensor_scalar_div(data_type_tensor tensor, float R, Scope_Struct *scope_struct);
  
// extern "C" void *CudaReverseScalarDiv(data_type_tensor tensor, float R, Scope_Struct *scope_struct);

extern "C" void *tensor_scalar_add(data_type_tensor *tensor, float R, Scope_Struct *scope_struct); 

extern "C" void *tensor_scalar_sub(data_type_tensor *tensor, float R, Scope_Struct *scope_struct); 

extern "C" void *tensor_scalar_equal(data_type_tensor tensor, float R, Scope_Struct *scope_struct); 
extern "C" void *tensor_scalar_diff(data_type_tensor tensor, float R, Scope_Struct *scope_struct);
extern "C" void *tensor_scalar_minor(data_type_tensor tensor, float R, Scope_Struct *scope_struct); 
extern "C" void *tensor_scalar_minor_eq(data_type_tensor tensor, float R, Scope_Struct *scope_struct);
extern "C" void *tensor_scalar_higher(data_type_tensor tensor, float R, Scope_Struct *scope_struct);
extern "C" void *tensor_scalar_higher_eq(data_type_tensor tensor, float R, Scope_Struct *scope_struct);


void scalarmult_backward(float *dx, float *dy, float scalar, float dims_prod);
