#pragma once

#include <vector>

#include "../tensor/tensor_struct.h"
#include "calculate_grids.h"

extern "C" void *CudaScalarMult(Tensor *tensor, float R, int thread_id);
  
  
extern "C" void *CudaScalarDiv(Tensor tensor, float R, int thread_id);
  
extern "C" void *CudaReverseScalarDiv(Tensor tensor, float R, int thread_id);

extern "C" void *CudaScalarAdd(Tensor *tensor, float R, int thread_id); 

extern "C" void *CudaScalarSub(Tensor *tensor, float R, int thread_id); 

extern "C" void *CudaScalarEqual(Tensor tensor, float R, int thread_id); 
extern "C" void *CudaScalarDiff(Tensor tensor, float R, int thread_id);
extern "C" void *CudaScalarMinor(Tensor tensor, float R, int thread_id); 
extern "C" void *CudaScalarMinorEq(Tensor tensor, float R, int thread_id);
extern "C" void *CudaScalarHigher(Tensor tensor, float R, int thread_id);
extern "C" void *CudaScalarHigherEq(Tensor tensor, float R, int thread_id);
