#pragma once


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



extern "C" Scope_Struct *scope_struct_Create();



extern "C" void set_scope_first_arg(Scope_Struct *, char *);
extern "C" void set_scope_scope(Scope_Struct *, char *);
extern "C" void set_scope_previous_scope(Scope_Struct *, char *);
extern "C" void set_scope_thread_id(Scope_Struct *, int);
extern "C" void set_scope_has_grad(Scope_Struct *, int);

extern "C" void print_scope_Copy(Scope_Struct *);

extern "C" void print_scope_struct(Scope_Struct *);