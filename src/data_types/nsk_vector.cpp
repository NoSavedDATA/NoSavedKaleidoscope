#include <cstdarg>
#include <stdlib.h>
#include <vector>

#include <iostream>

#include "../compiler_frontend/global_vars.h"
#include "../mangler/scope_struct.h"
#include "nsk_vector.h"



Nsk_Vector::Nsk_Vector() {
    size = 0;
    dims = {};
}
Nsk_Vector::Nsk_Vector(int size) : size(size) {
    dims = {size};
}
Nsk_Vector::Nsk_Vector(int size, std::vector<int> dims) : size(size), dims(std::move(dims)) {}




bool is_first_idx_call = true;

extern "C" int __idx__(Nsk_Vector *vec, int first_idx, ...) {
    if(is_first_idx_call)
        std::cout << "CALLING DEFAULT __idx__ FUNCTION" << ".\n";

    std::vector<int> indices;

    int dim=0, idx_at=0, rank=vec->dims.size();

    va_list args;
    va_start(args, first_idx);
    int idx = first_idx;
    do {
        if (idx<0)
            idx = vec->dims[dim] + idx;

        if(dim+1<rank)
            idx_at += idx*vec->dims[dim+1];
        else
            idx_at += idx;
        dim++;

        idx = va_arg(args, int);
    } while(idx!=TERMINATE_VARARG);
    va_end(args);


    if(is_first_idx_call) {
        std::cout << "Index at: " << idx_at << ".\n";
        is_first_idx_call=false;
    }

    return idx_at;
}


extern "C" int __sliced_idx__(Nsk_Vector *vec, int first_idx, ...) {
    int idx = 0;
    
    // va_list args;
    // va_start(args, first_idx);
    // int idx = first_idx;

    // while(idx!=TERMINATE_VARARG) {
    //     std::cout << "Got index: " << idx << ".\n";
    //     idx = va_arg(args, int);
    // } 
    // va_end(args);

    return idx;
}





Vec_Slices::Vec_Slices() : Nsk_Vector(0) {}

void Vec_Slices::push_back(DT_int_vec vec) {
    size++;
    slices.push_back(vec);
}
