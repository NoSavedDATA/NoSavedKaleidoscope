#include <cstdarg>
#include <stdlib.h>
#include <vector>

#include <iostream>

#include "../compiler_frontend/global_vars.h"
#include "../mangler/scope_struct.h"
#include "nsk_vector.h"



Nsk_Vector::Nsk_Vector(int size) : size(size) {
    dims = {size};
}
Nsk_Vector::Nsk_Vector(int size, std::vector<int> dims) : size(size), dims(std::move(dims)) {}

DT_int_vec::DT_int_vec(int size) : Nsk_Vector(size) {
  vec = (int*)malloc(size*sizeof(int));
}
DT_int_vec::~DT_int_vec() {
    // free(vec);
}




extern "C" int nsk_vec_size(Scope_Struct *scope_struct, Nsk_Vector *vec) {
    return vec->size;
}


extern "C" int __idx__(Nsk_Vector *vec, int first_idx, ...) {
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


    std::cout << "Index at: " << idx_at << ".\n";

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