#pragma once

#include <iostream>
#include <vector>


#include "../mangler/scope_struct.h"
#include "nsk_vector.h"

class DT_int_vec {
    public:
		int size, elem_size=4;
        int *vec;

    DT_int_vec(int size);
    DT_int_vec();
    void New(int size);
    ~DT_int_vec();
};



void int_vec_Clean_Up(void *);

