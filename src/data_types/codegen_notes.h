#pragma once

#include <any>
#include <iostream>
#include <map>
#include <vector>


class data_type_list {
public: 
    std::vector<std::any>* data;  // Pointer to vector stored in heap
    std::vector<std::string>* data_types;  // Pointer to vector stored in heap
    data_type_list(); 

    ~data_type_list(); 

    void append(std::any value, std::string); 

    template <typename T>
    T get(size_t index); 

    size_t size() const; 

    void print();
};


extern std::map<std::string, data_type_list *> NamedVectors;

