#pragma once

#include <cmath>
#include <iostream>
#include <limits>
#include <map>
#include <string>
#include <unordered_map>

#include "../../compiler_frontend/global_vars.h"
#include "../../compiler_frontend/logging_v.h"
#include "../../compiler_frontend/logging.h"
#include "../../clean_up/clean_up.h"
#include "../../data_types/list.h"
#include "../../mangler/scope_struct.h"
#include "../../pool/pool.h"
#include "../include.h"


std::string get_pool_obj_type(Scope_Struct *scope_struct, void *ptr) {
    GC *gc = scope_struct->gc;

    char *arena_addr = arena_base_addr[scope_struct->thread_id];

    long arena_offset = static_cast<char*>(ptr) - arena_addr;
    int arena =  arena_offset / GC_arena_size;
    if (arena<0) {
        std::cout << "at get_pool_obj_type()" << ".\n";
        std::cout << " Address " << ptr << " address does not reside in any memory pool.";
        std::exit(0);
    }
    int page  =  (arena_offset / GC_page_size) % pages_per_arena;
    // std::cout << "Belongs to arena: " << arena << ".\n";
    // std::cout << "Belongs to page: " << page << ".\n";

    GC_Span *span = scope_struct->gc->arenas[arena]->page_to_span[page];

    long obj_idx = (static_cast<char*>(ptr) - static_cast<char*>(span->span_address)) / span->traits->obj_size;

    return data_type_to_name[get_16_r12(span->type_metadata, obj_idx)];
}
