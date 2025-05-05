#pragma once

#include <map>
#include <string>

#include "../data_types/any_map.h" 
#include "../data_types/codegen_notes.h" 


struct Scope_Struct { 
    char *first_arg = nullptr;
    char *scope = nullptr;
    char *previous_scope = nullptr;
    char *function_name = nullptr;

    data_type_dict *mark_sweep_map = nullptr;
    
    int thread_id=0;
    int has_grad=1;

    int code_line = 0;


    Scope_Struct();

    void Set_First_Arg(char *);
    void Set_Scope(char *);
    void Set_Previous_Scope(char *);
    void Set_Function_Name(char *);
    void Set_Thread_Id(int);
    void Set_Has_Grad(int);
    void Alloc_MarkSweepMap();

    void Copy(Scope_Struct *);

    void Print();

};


extern std::map<std::string, Scope_Struct *> NamedScopeStructs;
