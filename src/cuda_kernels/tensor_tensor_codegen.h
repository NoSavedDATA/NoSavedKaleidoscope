#pragma once

#include <vector>

#include "../tensor/tensor_struct.h"
#include "calculate_grids.h"



extern "C" Tensor *tensor_tensor_mma(Tensor *tensor_x, Tensor *tensor_w, Scope_Struct *scope_struct); 


extern "C" Tensor *tensor_tensor_add(
            Tensor *tensor_x, Tensor *tensor_w, Scope_Struct *scope_struct); 


extern "C" Tensor *tensor_tensor_sub(
                          Tensor *tensor_x, Tensor *tensor_w, Scope_Struct *scope_struct); 


extern "C" Tensor *tensor_tensor_equal(
                          Tensor *tensor_x, Tensor *tensor_w, Scope_Struct *scope_struct);
                        
extern "C" Tensor *tensor_tensor_mult(
                          Tensor *tensor_x, Tensor *tensor_w, Scope_Struct *scope_struct); 



extern "C" void *tensor_tensor_div(
                          Tensor *tensor_x, Tensor *tensor_w, Scope_Struct *scope_struct); 

void hadamard_backward(float *x, float *w, float *dx, float *dw, float *dy, float dims_prod);
