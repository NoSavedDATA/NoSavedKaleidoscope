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


std::map<std::string, DT_dict *> NamedDicts;



// template char *DT_dict::get<char *>(std::string);
// template DT_tensor *DT_dict::get<DT_tensor *>(std::string);


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
    // return static_cast<T>(std::any_cast<void *>((*data)[key]));
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
    // if (data->count(key)<=0) {
    //     // throw std::out_of_range("key out of range");
    //     std::cout << "key out of range.";
    // }

    // return std::any_cast<float>((*data)[key]);
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

    move_to_char_pool(strlen(key)+1, key, "Append to DT_dict");
}


size_t DT_dict::size() const {
    return data->size();
}


void DT_dict::print() {
    std::cout << "\n";
    // for(int i=0; i<data->size(); i++)
    // {
    //     if (data_types->at(i)=="str")
    //         std::cout << "Notes["<<i<<"]: " << get<char *>(i) << ".\n";
    //     if (data_types->at(i)=="float")
    //         std::cout << "Notes["<<i<<"]: " << get<float>(i) << ".\n";
    //     if (data_types->at(i)=="tensor")
    //     {
    //         DT_tensor *t = get<DT_tensor *>(i);
    //         std::cout << "Notes["<<i<<"] is a tensor named: " << t->name << ".\n";
    //         PrintDims(t->dims);
    //     }
    // }
    std::cout << "\n";
}



extern "C" DT_dict *dictionary_Create() {
    // std::cout << "Creating vector\n";
    DT_dict *notes_vector = new DT_dict();
    // std::cout << "Notes Vector created.\n";

    return notes_vector;
}

extern "C" float dictionary_Dispose(DT_dict *notes_vector) {

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


// extern "C" DT_dict *Add_Float_To_NotesVector(DT_dict *notes_vector, float value) {

//     notes_vector->append(value, "float");

//     return notes_vector;
// }



// extern "C" DT_dict *Add_String_To_NotesVector(DT_dict *notes_vector, char *value) {

//     // std::cout << "Add_String " << value << " to notes_vector" << ".\n";
//     notes_vector->append((void *)value, "str");

//     return notes_vector;
// }