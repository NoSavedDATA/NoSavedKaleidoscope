#pragma once

#include <any>
#include <iostream>
#include <map>
#include <string>
#include <vector>


class DT_dict {
public: 
    std::map<std::string, std::any>* data; 
    std::map<std::string, std::string>* data_types; 
    DT_dict(); 

    ~DT_dict(); 

    void append(char *, std::any value, std::string); 

    template <typename T>
    T get(std::string index); 
    
    void delete_type(std::string);

    size_t size() const; 

    void print();
};


extern std::map<std::string, DT_dict *> NamedDicts;
