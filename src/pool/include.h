#pragma once

#include <iostream>


#include "../mangler/scope_struct.h"
#include "../mark_sweep/include.h"

#include "pool.h"



template<typename T>
T *allocate(Scope_Struct *scope_struct, int size, std::string type) {
    if (size==0)
        return nullptr;

    int alloc_size = size*sizeof(T);
    void *v_ptr = malloc(alloc_size);
    
    if(scope_struct!=nullptr)
    {
        Scope_Struct *inner_most_scope = get_inner_most_scope(scope_struct);
        inner_most_scope->gc.size_occupied += alloc_size;
        inner_most_scope->gc.allocations++;

        scope_struct->gc.pointer_nodes.push_back(GC_Node(v_ptr, type));
    }
    
    return static_cast<T*>(v_ptr);
}



template<typename T>
T *newT(Scope_Struct *scope_struct, std::string type) { 
    
    T *ptr = new T();

    if(scope_struct!=nullptr)
    {
        Scope_Struct *inner_most_scope = get_inner_most_scope(scope_struct);
        int size = sizeof(T);
        inner_most_scope->gc.size_occupied += size;
        inner_most_scope->gc.allocations++;

        scope_struct->gc.pointer_nodes.push_back(GC_Node(static_cast<void *>(ptr), type));
    }

    return ptr;
}
