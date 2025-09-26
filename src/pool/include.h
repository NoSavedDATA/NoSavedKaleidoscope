#pragma once


#include "../mangler/scope_struct.h"
#include "../mark_sweep/include.h"

#include "pool.h"



template<typename T>
T *allocate(Scope_Struct *scope_struct, int size, std::string type) {
    if (size==0)
        return nullptr;

    void *v_ptr = malloc(size * sizeof(T));
    
    if(scope_struct!=nullptr)
        scope_struct->gc.pointer_nodes.push_back(GC_Node(v_ptr, type));


    return static_cast<T*>(v_ptr);
}



template<typename T>
T *newT(Scope_Struct *scope_struct, std::string type) {

    
    T *ptr = new T();
    
    if(scope_struct!=nullptr)
        scope_struct->gc.pointer_nodes.push_back(GC_Node(static_cast<void *>(ptr), type));


    return ptr;
}
