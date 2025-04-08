#pragma once


struct Scope_Mangler { 
    char *first_arg = nullptr;
    char *scope = nullptr;
    char *previous_scope = nullptr;
    int thread_id=0;
    int has_grad=1;

    Scope_Mangler();
};



extern "C" Scope_Mangler *scope_mangler_Create();
