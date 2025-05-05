#include <any>
#include <iostream>
#include <map>
#include <stdexcept>
#include <vector>

#include "../tensor/tensor_dim_functions.h"
#include "../tensor/tensor_struct.h"
#include "codegen_notes.h"


std::map<std::string, data_type_list *> NamedVectors;




// template float data_type_list::get<float>(size_t);
template char *data_type_list::get<char *>(size_t);
template Tensor *data_type_list::get<Tensor *>(size_t);


template <typename T>
T data_type_list::get(size_t index) {
    if (index >= data->size()) {
        // throw std::out_of_range("Index out of range");
        std::cout << "Index out of range.";
    }
    // std::cout << "Get idx " << index << " from data_type_list" << ".\n";
    return static_cast<T>(std::any_cast<void *>((*data)[index]));
}

template <>
float data_type_list::get<float>(size_t index) {
    if (index >= data->size()) {
        // throw std::out_of_range("Index out of range");
        std::cout << "Index out of range.";
    }
    // std::cout << "Get float at idx " << index << " from data_type_list" << ".\n";
    return std::any_cast<float>((*data)[index]);
}

data_type_list::data_type_list() {
    data = new std::vector<std::any>(); // Allocate memory
    data_types = new std::vector<std::string>();
}

data_type_list::~data_type_list() {
    delete data;  // Free memory
    delete data_types;
}

void data_type_list::append(std::any value, std::string data_type) {
    data->push_back(value);
    data_types->push_back(data_type);
}


size_t data_type_list::size() const {
    return data->size();
}


void data_type_list::print() {
    std::cout << "\n";
    for(int i=0; i<data->size(); i++)
    {
        if (data_types->at(i)=="str")
            std::cout << "Notes["<<i<<"]: " << get<char *>(i) << ".\n";
        if (data_types->at(i)=="float")
            std::cout << "Notes["<<i<<"]: " << get<float>(i) << ".\n";
        if (data_types->at(i)=="tensor")
        {
            Tensor *t = get<Tensor *>(i);
            std::cout << "Notes["<<i<<"] is a tensor named: " << t->name << ".\n";
            PrintDims(t->dims);
        }
    }
    std::cout << "\n";
}



extern "C" data_type_list *CreateNotesVector() {
    // std::cout << "Creating vector\n";
    data_type_list *notes_vector = new data_type_list();
    // std::cout << "Notes Vector created.\n";

    return notes_vector;
}

extern "C" float Dispose_NotesVector(data_type_list *notes_vector, char *scopeless_name) {

    for (int i=0; i<notes_vector->size(); i++)
    {
        if (notes_vector->data_types->at(i)=="str")
        {
            char *val = notes_vector->get<char *>(i);
            delete[] val;
        }

    }
    
    delete notes_vector;
    delete[] scopeless_name;


    return 0;
}


extern "C" data_type_list *Add_Float_To_NotesVector(data_type_list *notes_vector, float value) {

    notes_vector->append(value, "float");

    return notes_vector;
}



extern "C" data_type_list *Add_String_To_NotesVector(data_type_list *notes_vector, char *value) {

    // std::cout << "Add_String " << value << " to notes_vector" << ".\n";
    notes_vector->append((void *)value, "str");

    return notes_vector;
}