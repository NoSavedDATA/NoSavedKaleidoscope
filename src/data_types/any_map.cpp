#include <any>
#include <cstring>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include "../char_pool/char_pool.h"
#include "../clean_up/clean_up.h"
#include "any_map.h"


std::map<std::string, DT_dict *> NamedDicts;



// template char *DT_dict::get<char *>(std::string);


template <typename T>
T DT_dict::get(std::string key) {
    auto it = data->find(key);
    if (it!=data->end())
    {
        return static_cast<T>(std::any_cast<void *>((*data)[key]));
    } else {
        std::cout << "Key " << key << " was not found on dictionary.";
        return nullptr;
    }
}

template <>
float DT_dict::get<float>(std::string key) {
    auto it = data->find(key);
    if (it!=data->end())
    {
        return std::any_cast<float>(it->second);
    } else {
        std::cout << "Key " << key << " was not found on dictionary.";
        return -1;
    }
}


template <>
int DT_dict::get<int>(std::string key) {
    auto it = data->find(key);
    if (it!=data->end())
    {
        return std::any_cast<int>(it->second);
    } else {
        std::cout << "Key " << key << " was not found on dictionary.";
        return -1;
    }
}




void DT_dict::delete_type(std::string key) {
    auto it = data->find(key);
    auto it_type = data_types->find(key);
   

    void *data_ptr=nullptr;
    if(it_type->second!="float")
        data_ptr = std::any_cast<void *>(it->second);    
    
    // if(data_ptr!=nullptr)
    //     std::cout << "" << it_type->second << " NOT NULL.\n";
    // else 
    //     std::cout << "GOT NULL OF " << it_type->second << ".\n";

    clean_up_functions[it_type->second](data_ptr);
}








DT_dict::DT_dict() {
    
    data = new std::map<std::string, std::any>(); // Allocate memory
    data_types = new std::map<std::string, std::string>();
}

DT_dict::~DT_dict() {
    // std::cout << "DELETING A MAP" << ".\n";
    delete data;  // Free memory
    delete data_types;
}

void DT_dict::append(char *key, std::any value, std::string data_type) {
    std::string _key = key;

    (*data)[_key] = value;
    (*data_types)[_key] = data_type;

    // move_to_char_pool(strlen(key)+1, key, "Append to DT_dict");
}


size_t DT_dict::size() const {
    return data->size();
}


void DT_dict::print() {
    std::cout << "dict print" << ".\n";
    std::cout << "\n";
    for (const auto &pair : *data)
    {
        const std::string &key = pair.first;
        const std::string &type = data_types->at(key);

        if (type == "int")
        {
            std::cout << "dict[\"" << key << "\"]: " << std::any_cast<int>(pair.second) << ".\n";
        }
        else if (type == "float")
            std::cout << "dict[\"" << key << "\"]: " << std::any_cast<float>(pair.second) << ".\n";

        else
        {
            void *t = std::any_cast<void *>(pair.second);
            std::cout << "dict[\"" << key << "\"] is a void pointer: " << t << " of type " << type << ".\n";
        }
    }
    std::cout << "\n";
}



