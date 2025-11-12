#include <iostream>
#include <stdlib.h>  // malloc, free
#include <string.h>  // memset, memcmp
#include <stdint.h>  // uint8_t
#include <stdbool.h>

#include "../compiler_frontend/global_vars.h"
#include "../mangler/scope_struct.h"
#include "../mark_sweep/include.h"


#define SENTINEL_BYTE 0xCD


bool check_initialized_field(void *ptr) {
    uint8_t sentinel[8];
    memset(sentinel, SENTINEL_BYTE, sizeof(sentinel));
    return memcmp(ptr, sentinel, sizeof(sentinel)) != 0;
}


extern "C" void *allocate_void(Scope_Struct *scope_struct, int size, char *type) {
    void *ptr = malloc(size);
    // void *ptr = scope_struct->Allocate(size);
    memset(ptr, SENTINEL_BYTE, size);

    scope_struct->gc.pointer_nodes.push_back(GC_Node(ptr, type));

    Scope_Struct *inner_most_scope = get_inner_most_scope(scope_struct);
    inner_most_scope->gc.size_occupied += size;
    inner_most_scope->gc.allocations++;
    
    return ptr;
}

