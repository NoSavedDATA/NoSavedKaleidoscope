#pragma once

#include <vector>
#include "int_vec.h"


class Nsk_Vector {

    public:
        int size;
        std::vector<int> dims;
    
        Nsk_Vector();
        Nsk_Vector(int);
        Nsk_Vector(int, std::vector<int>);
};


class Vec_Slices : public Nsk_Vector {

    public:
        std::vector<DT_int_vec> slices;
        Vec_Slices();

        void push_back(DT_int_vec);
};
