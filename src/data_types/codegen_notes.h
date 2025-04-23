#pragma once

#include <iostream>
#include <vector>
#include <any>


class AnyVector {
public: 
    std::vector<std::any>* data;  // Pointer to vector stored in heap
    std::vector<std::string>* data_types;  // Pointer to vector stored in heap
    AnyVector(); 

    ~AnyVector(); 

    void append(std::any value, std::string); 

    template <typename T>
    T get(size_t index); 

    size_t size() const; 

    void print();
};


extern std::map<std::string, AnyVector *> NamedVectors;


extern "C" AnyVector *CreateNotesVector();
extern "C" float Dispose_NotesVector(AnyVector *);


extern "C" AnyVector *Add_Float_To_NotesVector(AnyVector *, float value);


extern "C" AnyVector *Add_String_To_NotesVector(AnyVector *, char *value);