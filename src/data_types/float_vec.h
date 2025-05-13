#pragma once

#include <iostream>
#include <vector>


#include "../mangler/scope_struct.h"


struct DT_float_vec {
    int size;
    float *vec;

    DT_float_vec(int size);
};

void float_vec_Clean_Up(void *);