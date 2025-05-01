#include "../char_pool/include.h"
#include "../codegen/string.h"
#include "../data_types/include.h" 

#include "scope_struct.h"


Scope_Struct::Scope_Struct() {
    first_arg = get_from_char_pool(1,"Scope mangler first_arg");
    scope = get_from_char_pool(1,"Scope mangler scope");
    previous_scope = get_from_char_pool(1,"Scope mangler previous scope");
    function_name = get_from_char_pool(1,"Scope mangler previous scope");



    first_arg[0] = '\0';
    scope[0] = '\0';
    previous_scope[0] = '\0';
    function_name[0] = '\0';
}
std::map<std::string, Scope_Struct *> NamedScopeStructs;

void Scope_Struct::Set_First_Arg(char *first_arg) {
    delete[] this->first_arg;
    this->first_arg = first_arg;
}
void Scope_Struct::Set_Scope(char *scope) {
    delete[] this->scope;
    this->scope = scope;
}
void Scope_Struct::Set_Previous_Scope(char *previous_scope) {
    delete[] this->previous_scope;
    this->previous_scope = previous_scope;
}
void Scope_Struct::Set_Function_Name(char *function_name) {
    delete[] this->function_name;
    this->function_name = CopyString(function_name);
}
void Scope_Struct::Set_Thread_Id(int thread_id) {
    this->thread_id = thread_id;
}
void Scope_Struct::Set_Has_Grad(int has_grad) {
    this->has_grad = has_grad;
}
void Scope_Struct::Copy(Scope_Struct *scope_to_copy)
{
    delete[] first_arg;
    delete[] scope;
    delete[] previous_scope;
    delete[] function_name;

    first_arg = CopyString(scope_to_copy->first_arg);
    scope = CopyString(scope_to_copy->scope);
    previous_scope = CopyString(scope_to_copy->previous_scope);
    function_name = CopyString(scope_to_copy->function_name);


    thread_id = scope_to_copy->thread_id;
    has_grad = scope_to_copy->has_grad;
    code_line = scope_to_copy->code_line;

    
}

void Scope_Struct::Alloc_MarkSweepMap() {
    mark_sweep_map = new AnyMap();
}

    


void Scope_Struct::Print() {
    std::cout << "Scope struct:\n\tFirst arg: " << first_arg << "\n\tScope: " << scope << "\n\tPrevious scope: " << previous_scope << "\n\tThread id: " << thread_id << "\n\tHas grad: " << has_grad << ".\n\n";
}


extern "C" Scope_Struct *scope_struct_Create() {
    Scope_Struct *scope_struct = new Scope_Struct();
    return scope_struct;
}

extern "C" Scope_Struct *scope_struct_Copy(Scope_Struct *scope_to_copy) {
    
    // std::cout << "Copying scope struct" << ".\n";

    Scope_Struct *scope_struct = new Scope_Struct();
    scope_struct->Copy(scope_to_copy);

    // std::cout << "Scope struct copied" << ".\n";
    return scope_struct;
}


extern "C" Scope_Struct *scope_struct_Dive(Scope_Struct *scope_struct) {
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

extern "C" void set_scope_function_name(Scope_Struct *scope_struct, char *function_name) {
    // std::cout << "set_scope_has_grad: " << has_grad << ".\n";
    scope_struct->Set_Function_Name(function_name);
}


extern "C" char *get_scope_first_arg(Scope_Struct *scope_struct) {
    // std::cout << "get_scope_first_arg: " << scope_struct->first_arg << ".\n";
    return scope_struct->first_arg;
}
extern "C" char *get_scope_scope(Scope_Struct *scope_struct) {
    // std::cout << "get scope scope" << ".\n";
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


extern "C" void scope_struct_Get_Async_Scope(Scope_Struct *scope_struct, int thread_id, int has_grad) {
    // std::cout << "SET ASYNC SCOPE" << ".\n";
    scope_struct->scope = GetEmptyChar();
    scope_struct->thread_id = thread_id;
    scope_struct->has_grad = has_grad;
    // std::cout << "ASYNC SCOPE SET" << ".\n";
}


extern "C" void scope_struct_New_Anon_Expr(Scope_Struct *scope_struct) {
    scope_struct->first_arg = GetEmptyChar();
    scope_struct->scope = GetEmptyChar();
    scope_struct->previous_scope = GetEmptyChar();
}


extern "C" void scope_struct_Alloc_MarkSweepMap(Scope_Struct *scope_struct) {
    scope_struct->Alloc_MarkSweepMap();
}

extern "C" void scope_struct_Copy_MarkSweepMap(Scope_Struct *in_scope, Scope_Struct *out_scope) {
    // std::cout << "COPY MARKSWEEP" << ".\n";
    // The input struct receives the scope vars to clean. Cleaning scope vars from FunctionAST did not work.
    in_scope->mark_sweep_map = out_scope->mark_sweep_map;
}



inline void delete_scope(Scope_Struct *scope_struct) {

    delete[] scope_struct->first_arg;
    delete[] scope_struct->scope;
    delete[] scope_struct->previous_scope;
    delete[] scope_struct->function_name;

    if (scope_struct->mark_sweep_map!=nullptr)
        delete scope_struct->mark_sweep_map;


    delete scope_struct;
}


extern "C" void scope_struct_Clean_Scope(Scope_Struct *scope_struct) {
    if (strcmp(scope_struct->function_name,"")==0)
        return;

    // if (scope_struct->thread_id==0)

    for (auto &pair : *scope_struct->mark_sweep_map->data_types)
    {
        // std::cout << "Shall delete " << pair.first << "/" << pair.second << " on function " << scope_struct->first_arg << "_" << scope_struct->function_name << ".\n";

        if(pair.second=="float")
        {
            // std::cout << "Shall delete " << pair.first << "/" << pair.second << " on function " << scope_struct->first_arg << "_" << scope_struct->function_name << ".\n";
            NamedClassValues.erase(pair.first);
            // std::cout << "Cleaned " << pair.first << ".\n";
        }

        if(pair.second=="str")
        {
            // std::cout << "Shall delete " << pair.first << "/" << pair.second << " on function " << scope_struct->first_arg << "_" << scope_struct->function_name << ".\n";
            char *val = NamedStrs[pair.first];
            NamedClassValues.erase(pair.first);
            move_to_char_pool(strlen(val)+1, val, "Mark sweep of str");
            // std::cout << "Cleaned" << pair.first << ".\n";

        }

        if(pair.second=="tensor")
        {
            // std::cout << "Shall delete " << pair.first << "/" << pair.second << " on function " << scope_struct->first_arg << "_" << scope_struct->function_name << ".\n";
            Tensor *tensor = NamedTensorsT[pair.first];
            // if (nn_mode==eval_mode||scope_struct->thread_id!=0)
            // {
            //     std::cout << "Delete tensor on " << scope_struct->thread_id << ".\n";
            //     delete tensor;
            // }
            // if (nn_mode==eval_mode)
            //     delete tensor;
            // move_to_pool(scope_struct->thread_id, tensor->dims_prod, tensor->tensor_ptr, "tensor MarkSweep");
            NamedTensorsT.erase(pair.first);
        }

    }

    delete_scope(scope_struct);
}


extern "C" void scope_struct_Delete(Scope_Struct *scope_struct) {
    // std::cout << "Delete scope struct" << ".\n";    
    delete_scope(scope_struct);
}