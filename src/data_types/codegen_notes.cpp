#include <any>
#include <iostream>
#include <map>
#include <stdexcept>
#include <vector>

#include "../tensor/tensor_dim_functions.h"
#include "../tensor/tensor_struct.h"
#include "codegen_notes.h"


std::map<std::string, AnyVector *> NamedVectors;


extern "C" AnyVector *CreateNotesVector();
extern "C" float Dispose_NotesVector(AnyVector *);


// template float AnyVector::get<float>(size_t);
template char *AnyVector::get<char *>(size_t);
template Tensor *AnyVector::get<Tensor *>(size_t);


template <typename T>
T AnyVector::get(size_t index) {
    if (index >= data->size()) {
        // throw std::out_of_range("Index out of range");
        std::cout << "Index out of range.";
    }
    // std::cout << "Get idx " << index << " from AnyVector" << ".\n";
    return static_cast<T>(std::any_cast<void *>((*data)[index]));
}

template <>
float AnyVector::get<float>(size_t index) {
    if (index >= data->size()) {
        // throw std::out_of_range("Index out of range");
        std::cout << "Index out of range.";
    }
    // std::cout << "Get float at idx " << index << " from AnyVector" << ".\n";
    return std::any_cast<float>((*data)[index]);
}

AnyVector::AnyVector() {
    data = new std::vector<std::any>(); // Allocate memory
    data_types = new std::vector<std::string>();
}

AnyVector::~AnyVector() {
    delete data;  // Free memory
    delete data_types;
}

void AnyVector::append(std::any value, std::string data_type) {
    data->push_back(value);
    data_types->push_back(data_type);
}


size_t AnyVector::size() const {
    return data->size();
}


void AnyVector::print() {
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



extern "C" AnyVector *CreateNotesVector() {
    // std::cout << "Creating vector\n";
    AnyVector *notes_vector = new AnyVector();
    // std::cout << "Notes Vector created.\n";

    return notes_vector;
}

extern "C" float Dispose_NotesVector(AnyVector *notes_vector) {

    for (int i=0; i<notes_vector->size(); i++)
    {
        if (notes_vector->data_types->at(i)=="str")
        {
            char *val = notes_vector->get<char *>(i);
            delete[] val;
        }

    }
    
    delete notes_vector;


    return 0;
}


extern "C" AnyVector *Add_Float_To_NotesVector(AnyVector *notes_vector, float value) {

    notes_vector->append(value, "float");

    return notes_vector;
}



extern "C" AnyVector *Add_String_To_NotesVector(AnyVector *notes_vector, char *value) {

    // std::cout << "Add_String " << value << " to notes_vector" << ".\n";
    notes_vector->append((void *)value, "str");

    return notes_vector;
}