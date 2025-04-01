#pragma once

#include <vector>

#include "../tensor/tensor_struct.h"
#include "calculate_grids.h"



extern "C" Tensor *CudaMult(int is_forward_func,
                          Tensor *tensor_x, Tensor *tensor_w, int thread_id); 


extern "C" Tensor *CudaAdd(int is_forward_func,
            Tensor *tensor_x, Tensor *tensor_w, int thread_id); 


extern "C" Tensor *CudaSub(int is_forward_func,
                          Tensor *tensor_x, Tensor *tensor_w, int thread_id); 


extern "C" Tensor *CudaEqual(int is_forward_func,
                          Tensor *tensor_x, Tensor *tensor_w, int thread_id);
                        
extern "C" Tensor *CudaHadamard(int is_forward_func,
                          Tensor *tensor_x, Tensor *tensor_w, int thread_id); 



extern "C" void *CudaDiv(int is_forward_func,
                          Tensor *tensor_x, Tensor *tensor_w, int thread_id); 

void hadamard_backward(float *x, float *w, float *dx, float *dw, float *dy, float dims_prod);
