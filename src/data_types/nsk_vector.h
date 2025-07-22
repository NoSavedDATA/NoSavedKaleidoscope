#pragma once

#include <vector>


class Nsk_Vector {

    public:
        int size;
        std::vector<int> dims;
    
        Nsk_Vector(int);
        Nsk_Vector(int, std::vector<int>);
};

class DT_int_vec : public Nsk_Vector {
    public:
        int *vec;

    DT_int_vec(int size);
};