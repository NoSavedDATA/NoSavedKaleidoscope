#pragma once

#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "../mark_sweep/include.h" 


constexpr int ContextStackSize = 4096;

struct Scope_Struct { 
    int code_line=0;
    int thread_id=0;
    void *pointers_stack[ContextStackSize];
    int stack_top=0;
    void *object_ptr = nullptr;
    GC *gc=nullptr;
    std::vector<GC_Node> root_nodes;


    Scope_Struct *previous_scope=nullptr;

    char *first_arg = nullptr;
    char *scope = nullptr;
    char *function_name = nullptr;

    Scope_Struct *inner_most = nullptr;

    // GarbageCollector gc;
    
    int has_grad=1;

    int asyncs_count = 0;

    bool is_at_return = false;



    // std::map<std::string, std::vector<std::vector<std::string>>> debug_map;


    Scope_Struct();

    void Set_First_Arg(char *);
    void Set_Scope(char *);
    void Set_Function_Name(char *);
    void Set_Thread_Id(int);
    void Set_Has_Grad(int);

    void Copy(Scope_Struct *);

    void Print();
    void Print_Stack();

    void *Allocate(int, int);
    // inline void *Allocate(int size) {
    //     // std::cout << "Allcoate " << size << " on " << _gc << ".\n";
    //     return nullptr;
    // }
};


void alloc_gc_vspace(Scope_Struct *scope_struct, int size);

extern std::map<std::string, Scope_Struct *> NamedScopeStructs;
