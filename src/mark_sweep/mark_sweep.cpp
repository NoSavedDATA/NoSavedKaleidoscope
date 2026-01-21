#include <cmath>
#include <iostream>
#include <limits>
#include <map>
#include <string>
#include <unordered_map>

#include "../compiler_frontend/global_vars.h"
#include "../compiler_frontend/logging_v.h"
#include "../compiler_frontend/logging.h"
#include "../clean_up/clean_up.h"
#include "../data_types/list.h"
#include "../mangler/scope_struct.h"
#include "../pool/pool.h"
#include "include.h"




GC_span_traits::GC_span_traits(int obj_size) : obj_size(obj_size) {
    while (N<32&&pages<4) {
        pages++;
        N = (8192*pages) / obj_size;
        // std::cout << "N: " << N << ", pages: " << pages << ", obj size: " << obj_size << ".\n";
    }
    size = (8192*pages);

    if ((float)((8192*pages)/obj_size) != ((8192*pages)/(float)obj_size))
        N-=1;
    // std::cout << "Trait " << obj_size << "\n\tPages: " << pages << "\n\tN_elem: " << N << "\n\tTotal size: " << size << "\n\n";
}

GC_Span::GC_Span(GC_Arena *arena, GC_span_traits *traits) : arena(arena), traits(traits) {

    // Get Span address
    span_address = static_cast<char*>(arena->arena) + arena->size_allocated;
    arena->size_allocated += traits->size;
    // Set arena page idx
    for (int i=0; i<traits->pages; ++i) {
        arena->page_to_span[arena->pages_allocated] = this;
        arena->pages_allocated++;
    }

    // Get & initialize mark-bits
    words = (traits->N + 63) / 64;
    mark_bits = (uint64_t*)malloc(words*sizeof(uint64_t));
    for (int i=0; i<words; ++i)
       mark_bits[i] = 0ULL; 

    for (int i=traits->N; i<words*64; ++i)
        mark_bits_alloc(mark_bits, i);
    
    // Initialize type-metadata
    int types_per_word = 64 / 16;
    type_words = (traits->N + types_per_word-1) / types_per_word;
    type_metadata = (uint64_t*)malloc(type_words*sizeof(uint64_t));
    for (int i=0; i<type_words; ++i)
       type_metadata[i] = 0ULL; 

    for (int i=traits->N; i< ((traits->N + types_per_word-1) / types_per_word)*types_per_word; ++i)
        set_16_L2(type_metadata, i, 1u); // Set as protected
}

GC_Arena::GC_Arena(int tid) {
    arena = aligned_alloc(8192, arena_size);
    // arena = aligned_alloc(64u, arena_size);
    arena_base_addr[tid].push_back(static_cast<char*>(arena));
}

GC::GC(int tid) {
    GC_Arena *arena = new GC_Arena(tid);
    arenas.push_back( arena );
}

extern "C" void scope_struct_Alloc_GC(Scope_Struct *scope_struct) {
    scope_struct->gc = new GC(scope_struct->thread_id);
}



// //---------------------------------------------------------//





void protect_pool_addr(Scope_Struct *scope_struct, void *addr) {

    int tid = scope_struct->thread_id;
    char *p = static_cast<char*>(addr);

    char *arena_addr;
    // int arena =  arena_offset / GC_arena_size;

    int arena_id=-1;
    bool in_bounds;
    do {
        arena_id++;
        arena_addr = arena_base_addr[tid][arena_id]; 
        // arena =  arena_offset / GC_arena_size;
        in_bounds = (p>=arena_addr&&p<arena_addr+GC_arena_size);
    } while(!in_bounds&&arena_id<arena_base_addr[tid].size()-1);
    if (!in_bounds) {
        std::cout << "protect_pool_addr()\n------>Address " << addr << " address does not reside in any memory pool.\n";
        std::exit(0);
    }
    long arena_offset = p - arena_addr;
    int page  =  (arena_offset / GC_page_size) % pages_per_arena;

    GC_Span *span = scope_struct->gc->arenas[arena_id]->page_to_span[page];

    long obj_idx = (static_cast<char*>(addr) - static_cast<char*>(span->span_address)) / span->traits->obj_size;
    set_16_L2(span->type_metadata, obj_idx, 1u);
}

bool unprotect_pool_addr(Scope_Struct *scope_struct, void *addr) {

    int tid = scope_struct->thread_id;
    char *p = static_cast<char*>(addr);

    char *arena_addr;
    // int arena =  arena_offset / GC_arena_size;

    int arena_id=-1;
    bool in_bounds;
    do {
        arena_id++;
        arena_addr = arena_base_addr[tid][arena_id]; 
        // arena =  arena_offset / GC_arena_size;
        in_bounds = (p>=arena_addr&&p<arena_addr+GC_arena_size);
    } while(!in_bounds&&arena_id<arena_base_addr[tid].size()-1);
    if (!in_bounds) {
        std::cout << "unprotect_pool_addr()\n--------Address " << addr << " address does not reside in any memory pool.";
        return false;
    }
    long arena_offset = p - arena_addr;
    int page  =  (arena_offset / GC_page_size) % pages_per_arena;

    GC_Span *span = scope_struct->gc->arenas[arena_id]->page_to_span[page];

    long obj_idx = (static_cast<char*>(addr) - static_cast<char*>(span->span_address)) / span->traits->obj_size;
    set_16_L2(span->type_metadata, obj_idx, 0u);
    return true;
}





// //---------------------------------------------------------//

MarkSweep_Node::MarkSweep_Node(std::string type, bool marked) : type(type), marked(marked) {}
GC_Node::GC_Node(void *ptr, std::string type) : ptr(ptr), type(type) {}





