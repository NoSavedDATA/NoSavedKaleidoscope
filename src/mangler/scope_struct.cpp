#include"../char_pool/include.h"
#include"../codegen/string.h"

#include "scope_struct.h"


Scope_Struct::Scope_Struct() {
    first_arg = get_from_char_pool(1,"Scope mangler first_arg");
    scope = get_from_char_pool(1,"Scope mangler scope");
    previous_scope = get_from_char_pool(1,"Scope mangler previous scope");

    first_arg[0] = '\0';
    scope[0] = '\0';
    previous_scope[0] = '\0';
}
std::map<std::string, Scope_Struct *> NamedScopeStructs;

void Scope_Struct::Set_First_Arg(char *first_arg) {
    this->first_arg = CopyString(first_arg);
}
void Scope_Struct::Set_Scope(char *scope) {
    this->scope = CopyString(scope);
}
void Scope_Struct::Set_Previous_Scope(char *previous_scope) {
    this->previous_scope = CopyString(previous_scope);
}
void Scope_Struct::Set_Thread_Id(int thread_id) {
    this->thread_id = thread_id;
}
void Scope_Struct::Set_Has_Grad(int has_grad) {
    this->has_grad = has_grad;
}
void Scope_Struct::Copy(Scope_Struct *scope_to_copy)
{
    first_arg = CopyString(scope_to_copy->first_arg);
    scope = CopyString(scope_to_copy->scope);
    previous_scope = CopyString(scope_to_copy->previous_scope);

    thread_id = scope_to_copy->thread_id;
    has_grad = scope_to_copy->has_grad;
}


void Scope_Struct::Print() {
    std::cout << "Scope struct:\n\tFirst arg: " << first_arg << "\n\tScope: " << scope << "\n\tPrevious scope: " << previous_scope << "\n\tThread id: " << thread_id << "\n\tHas grad: " << has_grad << ".\n\n";
}


extern "C" Scope_Struct *scope_struct_Create() {
    Scope_Struct *scope_struct = new Scope_Struct();
    // std::cout << "Created scope struct" << ".\n";
    return scope_struct;
}

extern "C" Scope_Struct *scope_struct_Copy(Scope_Struct *scope_to_copy) {
    
    // std::cout << "Copying scope struct" << ".\n";

    Scope_Struct *scope_struct = new Scope_Struct();
    scope_struct->Copy(scope_to_copy);

    // std::cout << "Scope struct copied" << ".\n";
    return scope_struct;
}


extern "C" void set_scope_first_arg(Scope_Struct *scope_struct, char *first_arg) {
    // std::cout << "set_scope_first_arg: " << first_arg << ".\n";
    scope_struct->Set_First_Arg(first_arg);
}
extern "C" void set_scope_scope(Scope_Struct *scope_struct, char *scope) {
    // std::cout << "set_scope_scope: " << scope << ".\n";
    scope_struct->Set_Scope(scope);
}
extern "C" void set_scope_previous_scope(Scope_Struct *scope_struct, char *previous_scope) {
    // std::cout << "set_scope_previous_scope: " << previous_scope << ".\n";
    scope_struct->Set_Previous_Scope(previous_scope);
}
extern "C" void set_scope_thread_id(Scope_Struct *scope_struct, int thread_id) {
    // std::cout << "set_scope_thread_id: " << thread_id << ".\n";
    scope_struct->Set_Thread_Id(thread_id);
}
extern "C" void set_scope_has_grad(Scope_Struct *scope_struct, int has_grad) {
    // std::cout << "set_scope_has_grad: " << has_grad << ".\n";
    scope_struct->Set_Has_Grad(has_grad);
}

extern "C" char *get_scope_first_arg(Scope_Struct *scope_struct) {
    // std::cout << "get_scope_first_arg: " << scope_struct->first_arg << ".\n";
    return scope_struct->first_arg;
}
extern "C" char *get_scope_scope(Scope_Struct *scope_struct) {
    // std::cout << "get_scope_scope: " << scope_struct->scope << ".\n";
    return scope_struct->scope;
}
extern "C" char *get_scope_previous_scope(Scope_Struct *scope_struct) {
    // std::cout << "get_scope_previous_scope" << ".\n";
    return scope_struct->previous_scope;
}
extern "C" int get_scope_thread_id(Scope_Struct *scope_struct) {
    // std:cout << "get_scope_thread_id" << ".\n";
    return scope_struct->thread_id;
}
extern "C" int get_scope_has_grad(Scope_Struct *scope_struct) {
    // std::cout << "get_scope_has_grad" << ".\n";
    return scope_struct->has_grad;
}




extern "C" void scope_struct_Save_for_Async(Scope_Struct *scope_struct, char *fn_name) {
    Scope_Struct *scope_struct_copy = new Scope_Struct();
    scope_struct_copy->Copy(scope_struct);
    
    NamedScopeStructs[fn_name] = scope_struct_copy;
}
extern "C" Scope_Struct *scope_struct_Load_for_Async(char *fn_name)
{
    return NamedScopeStructs[fn_name];
}


extern "C" void scope_struct_Print(Scope_Struct *scope_struct) {
    std::cout << "Printing scope:" << ".\n";
    scope_struct->Print();
}