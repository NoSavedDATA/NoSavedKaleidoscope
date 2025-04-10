#pragma once

#include <map>
#include <string>


struct Scope_Struct { 
    char *first_arg = nullptr;
    char *scope = nullptr;
    char *previous_scope = nullptr;
    
    int thread_id=0;
    int has_grad=1;

    Scope_Struct();

    void Set_First_Arg(char *);
    void Set_Scope(char *);
    void Set_Previous_Scope(char *);
    void Set_Thread_Id(int);
    void Set_Has_Grad(int);

    void Copy(Scope_Struct *);

    void Print();
};


extern std::map<std::string, Scope_Struct *> NamedScopeStructs;

extern "C" Scope_Struct *scope_struct_Create();



extern "C" void set_scope_first_arg(Scope_Struct *, char *);
extern "C" void set_scope_scope(Scope_Struct *, char *);
extern "C" void set_scope_previous_scope(Scope_Struct *, char *);
extern "C" void set_scope_thread_id(Scope_Struct *, int);
extern "C" void set_scope_has_grad(Scope_Struct *, int);


extern "C" char *get_scope_first_arg(Scope_Struct *);
extern "C" char *get_scope_scope(Scope_Struct *);
extern "C" char *get_scope_previous_scope(Scope_Struct *);

extern "C" int get_scope_thread_id(Scope_Struct *scope_struct);
extern "C" int get_scope_has_grad(Scope_Struct *scope_struct);


extern "C" void scope_struct_Save_for_Async(Scope_Struct *, char *);
extern "C" Scope_Struct *scope_struct_Load_for_Async(char *);

extern "C" void print_scope_Copy(Scope_Struct *);

extern "C" void print_scope_struct(Scope_Struct *);