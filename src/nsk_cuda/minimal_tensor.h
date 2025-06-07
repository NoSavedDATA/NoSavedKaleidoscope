#pragma once

#include <cuda_runtime.h>
#include <string>
#include <type_traits>


#include "pool/include.h"




struct Minimal_Tensor {
    void *tensor=nullptr;
    int M, N, dims_prod;
};



struct CudaTensor {
    int M, N, aN;

    void *tensor;

    CudaTensor(int, int, int, std::string type="float"); 
};
