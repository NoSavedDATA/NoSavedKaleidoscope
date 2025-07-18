#include <mutex>
#include <set>

#include "../char_pool/include.h"
#include "../codegen/string.h"
#include "../codegen/time.h"
#include "../data_types/include.h" 
#include "../mark_sweep/include.h" 
#include "../threads/include.h"

#include "scope_struct.h"


Scope_Struct::Scope_Struct() {
    first_arg = get_from_char_pool(1,"Scope mangler first_arg");
    scope = get_from_char_pool(1,"Scope mangler scope");
    function_name = get_from_char_pool(1,"Scope mangler function name");

    first_arg[0] = '\0';
    scope[0] = '\0';
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
    delete[] function_name;


    object_ptr = scope_to_copy->object_ptr;
    first_arg = CopyString(scope_to_copy->first_arg);
    scope = CopyString(scope_to_copy->scope);
    function_name = CopyString(scope_to_copy->function_name);


    thread_id = scope_to_copy->thread_id;
    has_grad = scope_to_copy->has_grad;
    code_line = scope_to_copy->code_line;

    asyncs_count = scope_to_copy->asyncs_count;
}

void Scope_Struct::Alloc_MarkSweepMap() {
    mark_sweep_map = new MarkSweep();
}


void Scope_Struct::Print() {
    std::cout << "Scope struct:\n\tFirst arg: " << first_arg << "\n\tScope: " << scope << "\n\tThread id: " << thread_id << "\n\tHas grad: " << has_grad << ".\n\n";
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
extern "C" Scope_Struct *scope_struct_Overwrite(Scope_Struct *scope_struct, Scope_Struct *scope_to_copy) {
    scope_struct->Copy(scope_to_copy);
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
extern "C" int get_scope_thread_id(Scope_Struct *scope_struct) {
    // std::cout << "get_scope_thread_id " << scope_struct->thread_id << ".\n";
    return scope_struct->thread_id;
}
extern "C" int get_scope_has_grad(Scope_Struct *scope_struct) {
    // std::cout << "get_scope_has_grad" << ".\n";
    return scope_struct->has_grad;
}


std::set<int> assigned_ids;
std::mutex id_mutex;


extern "C" float scope_struct_Reset_Threads(Scope_Struct *scope_struct) {

    assigned_ids.clear();    
    return 0;
}

extern "C" float scope_struct_Increment_Thread(Scope_Struct *scope_struct) {
    // std::cout << "get_scope_has_grad" << ".\n";
    // pthread_mutex_lock(&create_thread_mutex);
    int thread_id = 0;

    // main_mutex.lock();
    
    std::lock_guard<std::mutex> lock(id_mutex);
    int candidate;
    candidate = 1;
    while (assigned_ids.count(candidate)) {
        candidate++;
    }
    assigned_ids.insert(candidate);
    


    scope_struct->thread_id = candidate;
    // std::cout << "INCREMENT SCOPE STRUCT THREAD ID " << scope_struct->thread_id << ".\n";

    // main_mutex.unlock();
    // pthread_mutex_unlock(&create_thread_mutex);
    // return scope_struct->has_grad;
    // std::exit(0);
    return 0;
}


extern "C" void set_scope_object(Scope_Struct *scope_struct, void *object_ptr) {
    // std::cout << "Set scope object " << object_ptr << ".\n";
    scope_struct->object_ptr = object_ptr;
    // std::cout << "done" << ".\n";
}
extern "C" void *get_scope_object(Scope_Struct *scope_struct) {
    // std::cout << "get scope object " << scope_struct->object_ptr << ".\n";
    return scope_struct->object_ptr;
}



extern "C" void scope_struct_Save_for_Async(Scope_Struct *scope_struct, char *fn_name) {
    // std::cout << "save for async: " << fn_name << ".\n";
    Scope_Struct *scope_struct_copy = new Scope_Struct();
    scope_struct_copy->Copy(scope_struct);
    
    NamedScopeStructs[fn_name] = scope_struct_copy;
}

extern "C" Scope_Struct *scope_struct_Load_for_Async(char *fn_name)
{
    return NamedScopeStructs[fn_name];
}

extern "C" void scope_struct_Store_Asyncs_Count(Scope_Struct *scope_struct, int asyncs_count) {
    scope_struct->asyncs_count = asyncs_count;
}

extern "C" void scope_struct_Print(Scope_Struct *scope_struct) {
    std::cout << "Printing scope:" << ".\n";
    scope_struct->Print();
}


extern "C" void scope_struct_Get_Async_Scope(Scope_Struct *scope_struct, int thread_id, int has_grad) {
    // std::exit(0);
    scope_struct->scope = GetEmptyChar();
    scope_struct->thread_id = thread_id;
    scope_struct->has_grad = has_grad;
    // std::cout << "ASYNC SCOPE SET" << ".\n";
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
    delete[] scope_struct->function_name;

    if (scope_struct->mark_sweep_map!=nullptr)
    {
        // std::cout << "Delete mark sweep" << ".\n";
        delete scope_struct->mark_sweep_map;
    }


    delete scope_struct;
}




extern "C" void scope_struct_Sweep(Scope_Struct *scope_struct) {
    scope_struct->mark_sweep_map->clean_up(false);
}  



extern "C" void scope_struct_Clean_Scope(Scope_Struct *scope_struct) {
    if (strcmp(scope_struct->function_name,"")==0)
    {
        // std::cout << "\n\n\n\nCLEANING SCOPE OF " <<  scope_struct->function_name << "-----------------------------------------------------------*****************----------------.\n\n\n\n\n";
        return;
    }

    // std::cout << "\n\n\n\nCLEANING SCOPE OF " <<  scope_struct->function_name << "-----------------------------------------------------------*****************----------------.\n\n\n\n\n";
    scope_struct->mark_sweep_map->clean_up(true);
    delete_scope(scope_struct);
}


extern "C" void scope_struct_Delete(Scope_Struct *scope_struct) {
    // std::cout << "Delete scope struct" << ".\n";    
    delete_scope(scope_struct);
}