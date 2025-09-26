#include <iostream>

#include "../mangler/scope_struct.h"
#include "../mark_sweep/include.h"

extern "C" void *allocate_void(Scope_Struct *scope_struct, int size, char *type) {

    void *ptr = malloc(size);

    scope_struct->gc.pointer_nodes.push_back(GC_Node(ptr, type));

    return ptr;
}

