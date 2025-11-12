#pragma once

#include <any>
#include <iostream>
#include <map>
#include <vector>

#include "nsk_vector.h"


class DT_list : public Nsk_Vector {
public: 
    std::vector<std::any>* data; 
    std::vector<std::string>* data_types;
    DT_list(); 


    ~DT_list(); 

    void append(std::any value, std::string); 
    

    template <typename T>
    T get(size_t index); 
    
    template <typename T>
    T unqueue(); 

    size_t Size() const; 

    void print();
};



