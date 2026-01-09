#include <any>
#include <iostream>
#include <map>
#include <stdexcept>
#include <vector>


#include "../compiler_frontend/logging_v.h"
#include "codegen_notes.h"
#include "nsk_vector.h"






template void *DT_list::get<void *>(size_t);

template char *DT_list::unqueue<char *>();
template bool  DT_list::unqueue<bool>();
template void *DT_list::unqueue<void *>();
template int   DT_list::unqueue<int>();
template float DT_list::unqueue<float>();



template <typename T>
T DT_list::get(size_t index) {
    if (index >= data->size()) {
        // throw std::out_of_range("Index out of range");
        std::cout << "Index out of range.";
    }
    // std::cout << "Get idx " << index << " from DT_list" << ".\n";
    return static_cast<T>(std::any_cast<void *>((*data)[index]));
}

template </*bool*/>
bool DT_list::get<bool>(size_t index) {
    if (index >= data->size()) {
        // throw std::out_of_range("Index out of range");
        std::cout << "Index out of range.";
    }
    // std::cout << "Get bool at idx " << index << " from DT_list" << ".\n";
    return std::any_cast<bool>((*data)[index]);
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


template </*char**/>
char *DT_list::get<char*>(size_t index) {
    if (index >= data->size()) {
        // throw std::out_of_range("Index out of range");
        std::cout << "Index out of range.";
    }
    // std::cout << "Get float at idx " << index << " from DT_list" << ".\n";
    return std::any_cast<char*>((*data)[index]);
}





template <typename T>
T DT_list::unqueue() {
    if (data->empty()) {
        LogErrorC(-1, "list out of range.");
    }
    T value = std::any_cast<T>((*data)[0]);

    data->erase(data->begin());
    data_types->erase(data_types->begin());
    size--;

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
    std::cout << "[";

    int i=0;
    if (data_types->at(i)=="str") {
        std::cout << "print type str char *"  << ".\n";
        std::cout << "size: " << size << ".\n";
        std::cout << get<char *>(i);
    }
    else if (data_types->at(i)=="float")
        std::cout << get<float>(i);
    else if (data_types->at(i)=="int")
        std::cout << get<int>(i);
    else
        std::cout << ", void*";

    for(i=1; i<data->size(); i++)
    {
        if (data_types->at(i)=="str")
            std::cout << ", " << get<char *>(i);
        else if (data_types->at(i)=="float")
            std::cout << ", " << get<float>(i);
        else if (data_types->at(i)=="int")
            std::cout << ", " << get<int>(i);
        else
        {
            void *t = get<void *>(i);
            std::cout << ", void*";
        }
    }
    std::cout << "]\n";
}



extern "C" DT_list *CreateNotesVector() {
    // std::cout << "Creating vector\n";
    DT_list *notes_vector = new DT_list();
    notes_vector->size=0;
    // std::cout << "Notes Vector created.\n";

    return notes_vector;
}

extern "C" float Dispose_NotesVector(DT_list *notes_vector, char *scopeless_name) {

    // for (int i=0; i<notes_vector->size; i++)
    // {
    //     if (notes_vector->data_types->at(i)=="str")
    //     {
    //         char *val = notes_vector->get<char *>(i);
    //         delete[] val;
    //     }

    // }
    
    delete notes_vector;
    // delete[] scopeless_name;


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
    notes_vector->append(value, "str");

    return notes_vector;
}
