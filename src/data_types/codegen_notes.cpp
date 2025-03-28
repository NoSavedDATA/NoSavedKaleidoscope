#pragma once

#include <iostream>
#include <vector>
#include <any>
#include <stdexcept>

#include "codegen_notes.h"


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

template <typename T>
T AnyVector::get(size_t index) {
    if (index >= data->size()) {
        // throw std::out_of_range("Index out of range");
        std::cout << "Index out of range.";
    }
    return std::any_cast<T>((*data)[index]);
}

size_t AnyVector::size() const {
    return data->size();
}




extern "C" AnyVector *CreateNotesVector() {
    // std::cout << "Creating vector\n";
    AnyVector *notes_vector = new AnyVector();
    std::cout << "Notes Vector created.\n";

    return notes_vector;
}

extern "C" float Dispose_NotesVector(AnyVector *notes_vector) {

    for (int i=0; i<notes_vector->size(); i++)
    {
        if (notes_vector->data_types->at(i)=="string")
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

    notes_vector->append(value, "string");

    return notes_vector;
}