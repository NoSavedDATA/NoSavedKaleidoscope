#include <iostream>
#include <stdlib.h>  // malloc, free
#include <string.h>  // memset, memcmp
#include <stdint.h>  // uint8_t
#include <stdbool.h>

#include "../compiler_frontend/global_vars.h"
#include "../compiler_frontend/logging_v.h"
#include "../mangler/scope_struct.h"
#include "../mark_sweep/include.h"




bool check_initialized_field(void *ptr) {
    uint8_t sentinel[8];
    memset(sentinel, SENTINEL_BYTE, sizeof(sentinel));
    return memcmp(ptr, sentinel, sizeof(sentinel)) != 0;
}


extern "C" void *allocate_void(Scope_Struct *scope_struct, int size, const char *type) { 
    auto it = data_name_to_type.find(type);
    if (it==data_name_to_type.end())
        LogErrorC(-1, std::string("Type ") + type + " not implemented.");
    uint16_t type_id = it->second;

    // void *ptr = malloc(size);

    if(type_id==0) {
        LogErrorC(-1, "Undefined type");
        std::exit(0);
    }
    void *ptr = scope_struct->Allocate(size, type_id);
    memset(ptr, SENTINEL_BYTE, size);


    scope_struct->gc->size_occupied += size;
    scope_struct->gc->allocations++;
    
    return ptr;
}

