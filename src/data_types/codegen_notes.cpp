#include <any>
#include <iostream>
#include <map>
#include <stdexcept>
#include <vector>


#include "../compiler_frontend/logging_v.h"
#include "codegen_notes.h"
#include "nsk_vector.h"






template char *DT_list::get<char *>(size_t);
template void *DT_list::get<void *>(size_t);

template char *DT_list::unqueue<char *>();
template void *DT_list::unqueue<void *>();

template <typename T>
T DT_list::get(size_t index) {
    if (index >= data->size()) {
        // throw std::out_of_range("Index out of range");
        std::cout << "Index out of range.";
    }
    // std::cout << "Get idx " << index << " from DT_list" << ".\n";
    return static_cast<T>(std::any_cast<void *>((*data)[index]));
}

template </*float*/>
float DT_list::get<float>(size_t index) {
    if (index >= data->size()) {
        // throw std::out_of_range("Index out of range");
        std::cout << "Index out of range.";
    }
    // std::cout << "Get float at idx " << index << " from DT_list" << ".\n";
    return std::any_cast<float>((*data)[index]);
}

template </*int*/>
int DT_list::get<int>(size_t index) {
    if (index >= data->size()) {
        // throw std::out_of_range("Index out of range");
        std::cout << "Index out of range.";
    }
    // std::cout << "Get float at idx " << index << " from DT_list" << ".\n";
    return std::any_cast<int>((*data)[index]);
}

template <typename T>
T DT_list::unqueue() {
    if (data->empty()) {
        // throw std::out_of_range("DT_list::unqueue: list is empty");
        LogErrorC(-1, "list out of range.");
    }

    // extract the first element
    T value = std::any_cast<T>((*data)[0]);

    // erase it from the front
    data->erase(data->begin());
    data_types->erase(data_types->begin());

    return value;
}




DT_list::DT_list() : Nsk_Vector(0) {
    data = new std::vector<std::any>(); // Allocate memory
    data_types = new std::vector<std::string>();
}

DT_list::~DT_list() {
    delete data;  // Free memory
    delete data_types;
}

void DT_list::append(std::any value, std::string data_type) {
    size++;
    data->push_back(value);
    data_types->push_back(data_type);
}


size_t DT_list::Size() const {
    return size;
}


void DT_list::print() {
    std::cout << "\n";
    for(int i=0; i<data->size(); i++)
    {
        if (data_types->at(i)=="str")
            std::cout << "Notes["<<i<<"]: " << get<char *>(i) << ".\n";
        else if (data_types->at(i)=="float")
            std::cout << "Notes["<<i<<"]: " << get<float>(i) << ".\n";
        else if (data_types->at(i)=="int")
            std::cout << "Notes["<<i<<"]: " << get<int>(i) << ".\n";
        else
        {
            void *t = get<void *>(i);
            std::cout << "Notes["<<i<<"] is a pointer" << ".\n";
        }
    }
    std::cout << "\n";
}



extern "C" DT_list *CreateNotesVector() {
    // std::cout << "Creating vector\n";
    DT_list *notes_vector = new DT_list();
    // std::cout << "Notes Vector created.\n";

    return notes_vector;
}

extern "C" float Dispose_NotesVector(DT_list *notes_vector, char *scopeless_name) {

    for (int i=0; i<notes_vector->size; i++)
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


extern "C" DT_list *Add_To_NotesVector_float(DT_list *notes_vector, float value) {

    notes_vector->append(value, "float");

    return notes_vector;
}


extern "C" DT_list *Add_To_NotesVector_int(DT_list *notes_vector, int value) {

    notes_vector->append(value, "int");

    return notes_vector;
}



extern "C" DT_list *Add_To_NotesVector_str(DT_list *notes_vector, char *value) {

    // std::cout << "Add_String " << value << " to notes_vector" << ".\n";
    notes_vector->append((void *)value, "str");

    return notes_vector;
}