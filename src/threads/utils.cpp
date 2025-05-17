#include <iostream>

#include "../mangler/scope_struct.h"


extern "C" float _tid(Scope_Struct *scope_struct) {
    std::cout << "Thread id is: " << scope_struct->thread_id-1 << ".\n";
    return 0;
}

int last_thread_id=1;

extern "C" int tid(Scope_Struct *scope_struct) {
    int thread_id = scope_struct->thread_id-1;
    return thread_id;
}