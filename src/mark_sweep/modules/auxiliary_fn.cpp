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

    int tid = scope_struct->thread_id;
    char *arena_addr;

    char *p = static_cast<char*>(ptr);
    // int arena =  arena_offset / GC_arena_size;

    bool in_bounds;
    int arena_id=-1;
    do {
        arena_id++;
        arena_addr = arena_base_addr[tid][arena_id]; 
        // arena =  arena_offset / GC_arena_size;
        in_bounds = (p>=arena_addr||p<arena_addr+GC_arena_size);
    } while(!in_bounds&&arena_id<arena_base_addr[tid].size()-1);
    if (!in_bounds) {
        std::cout << "at get_pool_obj_type()" << ".\n";
        std::cout << " Address " << ptr << " address does not reside in any memory pool.\n";
        std::exit(0);
    }
    long arena_offset = static_cast<char*>(ptr) - arena_addr;
    int page  =  (arena_offset / GC_page_size) % pages_per_arena;

    GC_Span *span = scope_struct->gc->arenas[arena_id]->page_to_span[page];

    long obj_idx = (static_cast<char*>(ptr) - static_cast<char*>(span->span_address)) / span->traits->obj_size;

    return data_type_to_name[get_16_r12(span->type_metadata, obj_idx)];
}
