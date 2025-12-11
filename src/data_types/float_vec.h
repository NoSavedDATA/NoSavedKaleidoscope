#pragma once

#include <iostream>
#include <vector>


#include "../mangler/scope_struct.h"


struct DT_float_vec {
    public:
		int size, elem_size=4;
        float *vec;

    DT_float_vec();
    DT_float_vec(int size);
    ~DT_float_vec();

    void New(int);
};

void float_vec_Clean_Up(void *);
