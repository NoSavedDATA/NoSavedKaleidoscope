#include <iostream>

#include "../mangler/scope_struct.h"
#include "../mark_sweep/include.h"

extern "C" void *allocate_void(Scope_Struct *scope_struct, int size, char *type) {
    void *ptr = malloc(size);

    scope_struct->gc.pointer_nodes.push_back(GC_Node(ptr, type));

    Scope_Struct *inner_most_scope = get_inner_most_scope(scope_struct);
    inner_most_scope->gc.size_occupied += size;

    return ptr;
}

