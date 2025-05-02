#include <any>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include "../char_pool/char_pool.h"
#include "../clean_up/clean_up.h"
#include "../tensor/tensor_dim_functions.h"
#include "../tensor/tensor_struct.h"
#include "any_map.h"


std::map<std::string, AnyMap *> NamedDicts;



// template char *AnyMap::get<char *>(std::string);
// template Tensor *AnyMap::get<Tensor *>(std::string);


template <typename T>
T AnyMap::get(std::string key) {
    auto it = data->find(key);
    if (it!=data->end())
    {
        return static_cast<T>(std::any_cast<void *>((*data)[key]));
    } else {
        std::cout << "Key " << key << " was not found on dictionary.";
        return nullptr;
    }
    // return static_cast<T>(std::any_cast<void *>((*data)[key]));
}

template <>
float AnyMap::get<float>(std::string key) {
    auto it = data->find(key);
    if (it!=data->end())
    {
        return std::any_cast<float>(it->second);
    } else {
        std::cout << "Key " << key << " was not found on dictionary.";
        return -1;
    }
    // if (data->count(key)<=0) {
    //     // throw std::out_of_range("key out of range");
    //     std::cout << "key out of range.";
    // }

    // return std::any_cast<float>((*data)[key]);
}




void AnyMap::delete_type(std::string key) {
    auto it = data->find(key);
    auto it_type = data_types->find(key);
   

    void *data_ptr=nullptr;
    if(it_type->second!="float")
        data_ptr = std::any_cast<void *>(it->second);    
    
    // if(data_ptr!=nullptr)
    //     std::cout << "" << it_type->second << " NOT NULL.\n";
    // else 
    //     std::cout << "GOT NULL OF " << it_type->second << ".\n";

    clean_up_functions[it_type->second](it->first, data_ptr);
}








AnyMap::AnyMap() {
    
    data = new std::map<std::string, std::any>(); // Allocate memory
    data_types = new std::map<std::string, std::string>();
}

AnyMap::~AnyMap() {
    // std::cout << "DELETING A MAP" << ".\n";
    delete data;  // Free memory
    delete data_types;
}

void AnyMap::append(char *key, std::any value, std::string data_type) {
    std::string _key = key;

    (*data)[_key] = value;
    (*data_types)[_key] = data_type;

    move_to_char_pool(strlen(key)+1, key, "Append to AnyMap");
}


size_t AnyMap::size() const {
    return data->size();
}


void AnyMap::print() {
    std::cout << "\n";
    // for(int i=0; i<data->size(); i++)
    // {
    //     if (data_types->at(i)=="str")
    //         std::cout << "Notes["<<i<<"]: " << get<char *>(i) << ".\n";
    //     if (data_types->at(i)=="float")
    //         std::cout << "Notes["<<i<<"]: " << get<float>(i) << ".\n";
    //     if (data_types->at(i)=="tensor")
    //     {
    //         Tensor *t = get<Tensor *>(i);
    //         std::cout << "Notes["<<i<<"] is a tensor named: " << t->name << ".\n";
    //         PrintDims(t->dims);
    //     }
    // }
    std::cout << "\n";
}



extern "C" AnyMap *dictionary_Create() {
    // std::cout << "Creating vector\n";
    AnyMap *notes_vector = new AnyMap();
    // std::cout << "Notes Vector created.\n";

    return notes_vector;
}

extern "C" float dictionary_Dispose(AnyMap *notes_vector) {

    // for (int i=0; i<notes_vector->size(); i++)
    // {
    //     if (notes_vector->data_types->at(i)=="str")
    //     {
    //         char *val = notes_vector->get<char *>(i);
    //         delete[] val;
    //     }

    // }
    
    delete notes_vector;


    return 0;
}


// extern "C" AnyMap *Add_Float_To_NotesVector(AnyMap *notes_vector, float value) {

//     notes_vector->append(value, "float");

//     return notes_vector;
// }



// extern "C" AnyMap *Add_String_To_NotesVector(AnyMap *notes_vector, char *value) {

//     // std::cout << "Add_String " << value << " to notes_vector" << ".\n";
//     notes_vector->append((void *)value, "str");

//     return notes_vector;
// }