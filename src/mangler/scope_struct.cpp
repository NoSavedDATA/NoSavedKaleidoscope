#include <atomic>
#include <cstring>
#include <mutex>
#include <set>
#include <thread>
#include <unordered_set>

#include "../char_pool/include.h"
#include "../codegen/string.h"
#include "../codegen/time.h"
#include "../compiler_frontend/global_vars.h"
#include "../mark_sweep/include.h" 
#include "../pool/include.h" 
#include "../threads/include.h"

#include "scope_struct.h"


void check_exit() {
    if (Shall_Exit)
        std::exit(0);
}

Scope_Struct::Scope_Struct() {
    first_arg = (char*)malloc(1);
    scope = (char*)malloc(1);
    function_name = (char*)malloc(1);

    first_arg[0] = '\0';
    scope[0] = '\0';
    function_name[0] = '\0';
}

Scope_Struct *get_inner_most_scope(Scope_Struct *scope_struct) {    
    // Get the inner most scope that belongs to the same thread.
    if(scope_struct->inner_most!=nullptr)
        return scope_struct->inner_most;

    Scope_Struct *inner_most = scope_struct;
    while(inner_most->previous_scope!=nullptr) {
        if (inner_most->thread_id!=inner_most->previous_scope->thread_id)
            break;
        inner_most = inner_most->previous_scope;
    } 
    scope_struct->inner_most = inner_most;
    return inner_most;
}

std::map<std::string, Scope_Struct *> NamedScopeStructs;

void Scope_Struct::Set_First_Arg(char *first_arg) {
    free(this->first_arg);
    this->first_arg = first_arg;
}
void Scope_Struct::Set_Scope(char *scope) {
    free(this->scope);
    this->scope = scope;
}
void Scope_Struct::Set_Function_Name(char *function_name) {
    free(this->function_name);
    this->function_name = CopyString(this, function_name);
}
void Scope_Struct::Set_Thread_Id(int thread_id) {
    this->thread_id = thread_id;
}
void Scope_Struct::Set_Has_Grad(int has_grad) {
    this->has_grad = has_grad;
}
void Scope_Struct::Copy(Scope_Struct *scope_to_copy)
{
    object_ptr = scope_to_copy->object_ptr;
    first_arg = CopyString(scope_to_copy, scope_to_copy->first_arg);
    scope = CopyString(scope_to_copy, scope_to_copy->scope);
    function_name = CopyString(scope_to_copy, scope_to_copy->function_name);

    thread_id = scope_to_copy->thread_id;
    has_grad = scope_to_copy->has_grad;
    code_line = scope_to_copy->code_line;

    asyncs_count = scope_to_copy->asyncs_count;
    
    previous_scope = scope_to_copy;
    get_inner_most_scope(this);
    _gc = inner_most->_gc;
}


void *Scope_Struct::Allocate(int size) {
    return _gc->Allocate(size);
}

void Scope_Struct::Print() {
    std::cout << "Scope struct:\n\tFirst arg: " << first_arg << "\n\tScope: " << scope << "\n\tThread id: " << thread_id << "\n\tHas grad: " << has_grad << ".\n\n";
}



extern "C" float scope_struct_spec(Scope_Struct *scope_struct) {
    std::cout << "scope_struct: " << scope_struct << ".\n";
    return 0;
}

extern "C" void set_scope_line(Scope_Struct *scope_struct, int line) {
    scope_struct->code_line = line;
}

extern "C" Scope_Struct *scope_struct_Create() {
    // check_exit();
    Scope_Struct *scope_struct = new Scope_Struct();
    // std::cout << "Create scope " << scope_struct << ".\n";
    return scope_struct;
}

extern "C" Scope_Struct *scope_struct_Copy(Scope_Struct *scope_to_copy) {
    Scope_Struct *scope_struct = new Scope_Struct();
    scope_struct->Copy(scope_to_copy);
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


std::unordered_set<int> assigned_ids;
std::mutex id_mutex;
static std::atomic<int> next_thread_id(1);
std::atomic<int> next_id(1);

extern "C" float scope_struct_Reset_Threads(Scope_Struct *scope_struct) {
    // std::cout << "-----------------RESET THREADS--------------------------"  << ".\n";
    assigned_ids.clear();    
    next_id = 1;
    return 0;
}

extern "C" float scope_struct_Increment_Thread(Scope_Struct *scope_struct) {
    static thread_local bool has_id = false;
    static thread_local int thread_id = 0;
    if (!has_id) {
        std::lock_guard<std::mutex> lock(id_mutex);
        thread_id = next_id++;
        // Handle wrap-around
        if (thread_id < 1) {
            next_id = 1;
            thread_id = 1;
        }
        assigned_ids.insert(thread_id);
        has_id = true;
    }
    
    scope_struct->thread_id = thread_id;
    return 0;
}

extern "C" void scope_struct_Print(Scope_Struct *scope_struct) {
    std::cout << "Printing scope:" << ".\n";
    scope_struct->Print();
}

extern "C" void set_scope_object(Scope_Struct *scope_struct, void *object_ptr) {
    scope_struct->object_ptr = object_ptr;
}
extern "C" void *get_scope_object(Scope_Struct *scope_struct) {
    return scope_struct->object_ptr;
}



extern "C" void scope_struct_Save_for_Async(Scope_Struct *scope_struct, char *fn_name) {
    Scope_Struct *scope_struct_copy = new Scope_Struct();
    // scope_struct_copy->Copy(scope_struct);
    // NamedScopeStructs[fn_name] = scope_struct_copy;
    NamedScopeStructs[fn_name] = scope_struct;
}

extern "C" void *scope_struct_Load_for_Async(char *fn_name) {
    Scope_Struct *scope_struct = NamedScopeStructs[fn_name];

    Scope_Struct *scope_struct_copy = new Scope_Struct();
    scope_struct_copy->Copy(scope_struct);
    // std::cout << "Threaded scope is: " << scope_struct_copy << "/" << scope_struct << ".\n";

    scope_struct_copy->_gc = new GC();
    return scope_struct_copy;
}

extern "C" void scope_struct_Store_Asyncs_Count(Scope_Struct *scope_struct, int asyncs_count) {
    scope_struct->asyncs_count = asyncs_count;
}




extern "C" void scope_struct_Get_Async_Scope(Scope_Struct *scope_struct, int thread_id, int has_grad) {
    scope_struct->scope = GetEmptyChar(scope_struct);
    scope_struct->thread_id = thread_id;
    scope_struct->has_grad = has_grad;
}





extern "C" void scope_struct_Copy_MarkSweepMap(Scope_Struct *in_scope, Scope_Struct *out_scope) {
    // in_scope->mark_sweep_map = out_scope->mark_sweep_map;
}



extern "C" void scope_struct_Clear_GC_Root(Scope_Struct *scope_struct) {
    // std::cout << "clearing " << scope_struct->gc.root_nodes.size() << ".\n";
    scope_struct->gc.root_nodes.clear();
}

extern "C" void scope_struct_Add_GC_Root(Scope_Struct *scope_struct, void *root_pointer, char *type) {
    scope_struct->gc.root_nodes.push_back(GC_Node(root_pointer, type));
}




inline void delete_scope(Scope_Struct *scope_struct) {

    free(scope_struct->first_arg);
    free(scope_struct->scope);
    free(scope_struct->function_name);

    // if (scope_struct->mark_sweep_map!=nullptr)
    //     delete scope_struct->mark_sweep_map;


    delete scope_struct;
}


void alloc_gc_vspace(Scope_Struct *scope_struct, int size) {
    GarbageCollector &gc = get_inner_most_scope(scope_struct)->gc;
    gc.size_occupied += size;
}

extern "C" void scope_struct_Sweep(Scope_Struct *scope_struct) {
    GarbageCollector &gc = get_inner_most_scope(scope_struct)->gc;

    if (gc.allocations>1000||gc.size_occupied>1000000) {
        // std::cout << "sweep" << ".\n";
        scope_struct->gc.sweep(scope_struct);
    }
}

extern "C" void scope_struct_Clean_Scope(Scope_Struct *scope_struct) {
    Scope_Struct *inner_most_scope = get_inner_most_scope(scope_struct);
    GarbageCollector &gc = inner_most_scope->gc;
    
    if (gc.allocations>1000||gc.size_occupied>1000000) {
        // std::cout << "CLEAN UP" << ".\n";
        scope_struct->gc.sweep(scope_struct);
    }
    else {
        if (inner_most_scope!=scope_struct) {
            for (const auto &node : scope_struct->gc.pointer_nodes)
                inner_most_scope->gc.pointer_nodes.push_back(GC_Node(node.ptr, node.type));
        }
    }
    delete_scope(scope_struct);
}

extern "C" void scope_struct_Delete(Scope_Struct *scope_struct) {
    delete_scope(scope_struct);
}