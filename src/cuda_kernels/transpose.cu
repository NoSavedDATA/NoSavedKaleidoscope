#pragma once

#include "../mangler/scope_struct.h"
// #include "../tensor/tensor_struct.h"
#include "../nsk_cuda/include.h"
#include "../tensor/include.h"



extern "C" float tensor_transpose(Scope_Struct *scope_struct, DT_tensor *x)
{
    //todo: add to backprop

    
    
    float *x_T = get_from_pool(scope_struct->thread_id, x->dims_prod, "transpose");
    
    int dims_size = x->dims.size();
    int C = x->dims[dims_size-1];
    int B = x->dims[dims_size-2];
    
    PrintTensorF(x->tensor_ptr, B, C);
    
    cudaStream_t stream = ThreadsStream[scope_struct->thread_id];  

    transpose_tensor(x_T, x->tensor_ptr, B, C, stream);


    PrintTensorF(x_T, 130, B);



    return 0;
}


