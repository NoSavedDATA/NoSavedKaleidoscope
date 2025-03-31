#pragma once


#include "calculate_grids.h"
#include "dim_kernels.h"
#include "elementwise_codegen.h"
#include "elementwise_kernels.h"
#include "elementwise_kernels_inline.cu"
#include "handles.h"
#include "tensor_scalar_codegen.h"
#include "tensor_scalar_kernels.cu"


extern cudaDeviceProp deviceProp;