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





// void get_recursive_pointers(Scope_Struct *scope_struct, std::unordered_map<void *,MarkSweep_Node> &mark_sweep_dict) {
//     for (GC_Node pointer : scope_struct->gc.pointer_nodes)
//         mark_sweep_dict.emplace(pointer.ptr, MarkSweep_Node(pointer.type, false));

//     scope_struct->gc.pointer_nodes.clear();
    
//     if (scope_struct->previous_scope!=nullptr) { // clear only current thread pointers.
//         if (scope_struct->previous_scope->thread_id==scope_struct->thread_id)
//             get_recursive_pointers(scope_struct->previous_scope, mark_sweep_dict);
//     }
// }

// void GarbageCollector::sweep(Scope_Struct *scope_struct) {
//     if(scope_struct->thread_id!=0)
//         return;

//     // std::cout << "\n\nclean scope struct: " << scope_struct << "---\n\n\n";
//     std::unordered_map<void *, MarkSweep_Node> mark_sweep_dict;
//     get_recursive_roots(scope_struct, mark_sweep_dict);

//     scope_struct->gc.root_nodes.clear();
//     get_recursive_pointers(scope_struct, mark_sweep_dict);

//     Scope_Struct *inner_most_scope = get_inner_most_scope(scope_struct);
//     // std::cout << "innermost size: " << inner_most_scope->gc.size_occupied << ".\n";
//     // std::cout << "--" << "\n";

//     inner_most_scope->gc.size_occupied = 0;
//     inner_most_scope->gc.allocations = 0;

//     for (const auto &pair : mark_sweep_dict) {
//         void *ptr = pair.first;
//         MarkSweep_Node node = pair.second;
//         // std::cout << "found ptr: " << ptr << ".\n";
//         if(ptr==nullptr)
//             continue;

//         if (node.marked) {
//             inner_most_scope->gc.pointer_nodes.push_back(GC_Node(ptr, node.type));
//             // inner_most_scope->gc.allocations++;
//         } else {

//             // std::cout << "\n\nclean: " << ptr << ".\n";
//             // std::cout << "of type: " << node.type << ".\n";
//             // // std::cout << "is class: " << ClassPointers.count(node.type) << ".\n\n";

//             if (node.type=="str"||ClassPointers.count(node.type)>0)
//                 free(ptr);
//             else
//             {
//                 // std::cout << "--CLEANING: " << node.type << ".\n";
//                 // std::cout << "--CLEANING: " << node.type << "/" << ptr << ".\n";
//                 clean_up_functions[node.type](ptr); 
//                 // std::cout << "-\n\n";
//             }
            
//         }
//     }

//     // std::cout << "---" << "\n";
// }
