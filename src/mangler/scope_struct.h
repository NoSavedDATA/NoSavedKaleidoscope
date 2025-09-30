#pragma once

#include <map>
#include <string>
#include <vector>

#include "../mark_sweep/include.h" 


struct Scope_Struct { 
    Scope_Struct *previous_scope=nullptr;
    void *object_ptr = nullptr;

    char *first_arg = nullptr;
    char *scope = nullptr;
    char *function_name = nullptr;

    // MarkSweep *mark_sweep_map = nullptr;
    MarkSweep *mark_sweep_map = nullptr;

    GarbageCollector gc;
    
    int thread_id=0;
    int has_grad=1;

    int code_line = 0;
    int asyncs_count = 0;

    bool is_at_return = false;



    // std::map<std::string, std::vector<std::vector<std::string>>> debug_map;


    Scope_Struct();

    void Set_First_Arg(char *);
    void Set_Scope(char *);
    void Set_Function_Name(char *);
    void Set_Thread_Id(int);
    void Set_Has_Grad(int);
    void Alloc_MarkSweepMap();

    void Copy(Scope_Struct *);

    void Print();

};

Scope_Struct *get_inner_most_scope(Scope_Struct *);

extern std::map<std::string, Scope_Struct *> NamedScopeStructs;
