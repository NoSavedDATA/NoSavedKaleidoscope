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


extern "C" void print_stack1(Scope_Struct *scope_struct) {
    std::cout << "stack[0]: " << scope_struct->pointers_stack[0] << ".\n";
    std::cout << "stack[1]: " << scope_struct->pointers_stack[1] << ".\n";
}

extern "C" void print_stack(Scope_Struct *scope_struct, void *stack) {
    std::cout << "stack[0]: " << scope_struct->pointers_stack[0] << ".\n";
    std::cout << "stack[1]: " << scope_struct->pointers_stack[1] << ".\n";
    std::cout << "stack top: " << stack << ".\n";
}




void check_exit() {
    if (Shall_Exit)
        std::exit(0);
}

Scope_Struct::Scope_Struct() {
}


std::map<std::string, Scope_Struct *> NamedScopeStructs;


void Scope_Struct::Set_Thread_Id(int thread_id) {
    this->thread_id = thread_id;
}
void Scope_Struct::Set_Has_Grad(int has_grad) {
    this->has_grad = has_grad;
}
void Scope_Struct::Copy(Scope_Struct *scope_to_copy)
{
    // std::cout << "copy scope" << ".\n";
    object_ptr = scope_to_copy->object_ptr;

    thread_id = scope_to_copy->thread_id;
    has_grad = scope_to_copy->has_grad;
    code_line = scope_to_copy->code_line;

    asyncs_count = scope_to_copy->asyncs_count;
    
    previous_scope = scope_to_copy;
    inner_most = scope_to_copy->inner_most;
    
    gc = inner_most->gc;
}


void *Scope_Struct::Allocate(int size, int type_id) {
    void *ret = gc->Allocate(size, type_id, thread_id); 
    // if(thread_id!=0) {
    //     std::cout << " --> Allocating: " << ret << ".\n";
    // }
    return ret;
}

void Scope_Struct::Print() {
    std::cout << "Scope struct:" << "\n\tThread id: " << thread_id << ".\n\n";
}

void Scope_Struct::Print_Stack() {
    std::cout << "has " << stack_top << " stack items.\n";
    for (int i=0; i<stack_top; ++i) {
        std::cout << i << ": " << pointers_stack[i] << "\n";        
    }
}



extern "C" float scope_struct_spec(Scope_Struct *scope_struct) {
    std::cout << "scope_struct: " << scope_struct << ".\n";
    return 0;
}

extern "C" void set_scope_line(Scope_Struct *scope_struct, int line) {
    scope_struct->code_line = line;
}


extern "C" Scope_Struct *scope_struct_CreateFirst() {
    Scope_Struct *scope_struct = new Scope_Struct();
    scope_struct->inner_most = scope_struct;
    return scope_struct;
}
extern "C" Scope_Struct *scope_struct_Create() {
    // check_exit();
    Scope_Struct *scope_struct = new Scope_Struct();
    // std::cout << "Create scope " << scope_struct << ".\n";
    return scope_struct;
}

extern "C" Scope_Struct *scope_struct_Overwrite(Scope_Struct *scope_struct, Scope_Struct *scope_to_copy) {
    scope_struct->Copy(scope_to_copy);
    return scope_struct;
}






extern "C" void set_scope_thread_id(Scope_Struct *scope_struct, int thread_id) {
    // std::cout << "set_scope_thread_id: " << thread_id << ".\n";
    scope_struct->Set_Thread_Id(thread_id);
}



extern "C" int get_scope_thread_id(Scope_Struct *scope_struct) {
    // std::cout << "get_scope_thread_id " << scope_struct->thread_id << ".\n";
    return scope_struct->thread_id;
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
    scope_struct->gc = new GC(thread_id);
    return 0;
}

extern "C" void scope_struct_Print(Scope_Struct *scope_struct) {
    std::cout << "Printing scope:" << ".\n";
    scope_struct->Print();
}




extern "C" void scope_struct_Save_for_Async(Scope_Struct *scope_struct, char *fn_name) {
    // Scope_Struct *scope_struct_copy = new Scope_Struct();
    // scope_struct_copy->Copy(scope_struct);
    // NamedScopeStructs[fn_name] = scope_struct_copy;
    NamedScopeStructs[fn_name] = scope_struct;
}

extern "C" void *scope_struct_Load_for_Async(char *fn_name) {
    Scope_Struct *scope_struct = NamedScopeStructs[fn_name];

    Scope_Struct *scope_struct_copy = new Scope_Struct();
    scope_struct_copy->Copy(scope_struct);
    scope_struct_copy->inner_most = scope_struct_copy;
    // std::cout << "Threaded scope is: " << scope_struct << "/" << scope_struct_copy << ".\n";
    
    return scope_struct_copy;
}

extern "C" void scope_struct_Store_Asyncs_Count(Scope_Struct *scope_struct, int asyncs_count) {
    scope_struct->asyncs_count = asyncs_count;
}


extern "C" void scope_struct_Get_Async_Scope(Scope_Struct *scope_struct, int thread_id, int has_grad) {
    scope_struct->thread_id = thread_id;
}





extern "C" void scope_struct_Clear_GC_Root(Scope_Struct *scope_struct) {
    // std::cout << "----->clearing " << scope_struct->root_nodes.size() << ".\n";
    scope_struct->root_nodes.clear();
}

extern "C" void scope_struct_Add_GC_Root(Scope_Struct *scope_struct, void *root_pointer, char *type) {
    scope_struct->root_nodes.push_back(GC_Node(root_pointer, type));
}


inline void delete_scope(Scope_Struct *scope_struct) {
    delete scope_struct;
}


void alloc_gc_vspace(Scope_Struct *scope_struct, int size) {
    // std::cout << "alloc vspace of " << size << ".\n";
    scope_struct->gc->size_occupied += size;
}


extern "C" void scope_struct_Sweep(Scope_Struct *scope_struct) {
    GC *gc = scope_struct->gc;
    // std::cout << "sweep: " << scope_struct << " - / - " << gc << ".\n";
    // scope_struct->Print_Stack();
    // std::cout << "sweep check: " << gc->allocations << "/" << gc->size_occupied << ".\n";
    gc->Sweep(scope_struct);
}

extern "C" void scope_struct_Delete(Scope_Struct *scope_struct) {
    GC *gc = scope_struct->gc;

    gc->Sweep(scope_struct);

    for (auto arena : gc->arenas) {        
        for (auto span_vec_pair : arena->Spans) {
            for (auto span : span_vec_pair.second) {
                // free(span->mark_bits);
                // free(span->type_metadata);
                // free(span);
            }
        } 
        // free(arena->arena); // todo: channels may receive data only after the arena was cleaned
        free(arena->metadata);
    }

    free(scope_struct);
    free(gc);
}

