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
    }
    size = (8192*pages);
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

GC_Arena::GC_Arena() {
    // std::cout << "init arena size " << arena_size << ".\n";
    // arena = calloc(arena_size, 1);

    // arena = malloc(arena_size);
    arena = aligned_alloc(8192, arena_size);
}

GC::GC(int tid) {
    GC_Arena *arena = new GC_Arena();
    arena_base_addr[tid] = static_cast<char*>(arena->arena);
    arenas.push_back( arena );
}

extern "C" void scope_struct_Alloc_GC(Scope_Struct *scope_struct) {
    scope_struct->gc = new GC(scope_struct->thread_id);
}



// //---------------------------------------------------------//





void protect_pool_addr(Scope_Struct *scope_struct, void *addr) {
    // std::cout << "Protect address " << addr << ".\n";

    char *arena_addr = arena_base_addr[scope_struct->thread_id];

    long arena_offset = static_cast<char*>(addr) - arena_addr;
    int arena =  arena_offset / GC_arena_size;
    if (arena<0) {
        std::cout << "------>Address " << addr << " address does not reside in any memory pool.\n";
        std::cout << "Arena: " << arena << ".\n";
        std::exit(0);
    }
    int page  =  (arena_offset / GC_page_size) % pages_per_arena;
    // std::cout << "Belongs to arena: " << arena << ".\n";
    // std::cout << "Belongs to page: " << page << ".\n";

    GC_Span *span = scope_struct->gc->arenas[arena]->page_to_span[page];

    long obj_idx = (static_cast<char*>(addr) - static_cast<char*>(span->span_address)) / span->traits->obj_size;
    // std::cout << "Obj idx in span: " << obj_idx << ".\n";
    // std::cout << "span: " << span << ".\n";
    // std::cout << "span obj_size " << span->traits->obj_size << ", pages: " << span->traits->pages << ", N: " << span->traits->N << ".\n";
    set_16_L2(span->type_metadata, obj_idx, 1u);
}

bool unprotect_pool_addr(Scope_Struct *scope_struct, void *addr) {
    // std::cout << "Protect address " << addr << ".\n";

    char *arena_addr = arena_base_addr[scope_struct->thread_id];

    long arena_offset = static_cast<char*>(addr) - arena_addr;
    int arena =  arena_offset / GC_arena_size;
    if (arena<0) {
        // std::cout << " Address " << addr << " address does not reside in any memory pool.";
        return false;
    }
    int page  =  (arena_offset / GC_page_size) % pages_per_arena;
    // std::cout << "Belongs to arena: " << arena << ".\n";
    // std::cout << "Belongs to page: " << page << ".\n";

    GC_Span *span = scope_struct->gc->arenas[arena]->page_to_span[page];

    long obj_idx = (static_cast<char*>(addr) - static_cast<char*>(span->span_address)) / span->traits->obj_size;
    // std::cout << "Obj idx in span: " << obj_idx << ".\n";
    // std::cout << "span: " << span << ".\n";
    // std::cout << "span obj_size " << span->traits->obj_size << ", pages: " << span->traits->pages << ", N: " << span->traits->N << ".\n";
    set_16_L2(span->type_metadata, obj_idx, 0u);
    return true;
}





// //---------------------------------------------------------//

MarkSweep_Node::MarkSweep_Node(std::string type, bool marked) : type(type), marked(marked) {}
GC_Node::GC_Node(void *ptr, std::string type) : ptr(ptr), type(type) {}





