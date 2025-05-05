#pragma once

#include <any>
#include <iostream>
#include <map>
#include <string>
#include <vector>


class data_type_dict {
public: 
    std::map<std::string, std::any>* data;  // Pointer to vector stored in heap
    std::map<std::string, std::string>* data_types;  // Pointer to vector stored in heap
    data_type_dict(); 

    ~data_type_dict(); 

    void append(char *, std::any value, std::string); 

    template <typename T>
    T get(std::string index); 
    
    void delete_type(std::string);

    size_t size() const; 

    void print();
};


extern std::map<std::string, data_type_dict *> NamedDicts;
