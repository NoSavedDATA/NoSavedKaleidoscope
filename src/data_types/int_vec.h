#pragma once

#include <iostream>
#include <vector>


#include "../mangler/scope_struct.h"


struct DT_int_vec {
    int size;
    int *vec;

    DT_int_vec(int size);
};

void int_vec_Clean_Up(void *);