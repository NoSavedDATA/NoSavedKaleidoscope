#pragma once

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
#include "mark_sweep.h"




int gc_sizes[GC_obj_sizes];
uint16_t GC_size_to_class[GC_N+1];


int bin_search_gc_size(int size, int idx, int half) { 
    if(idx==0||idx==GC_obj_sizes-1)
        return gc_sizes[idx];

    if(half<1)
        half=1;

    if(size<gc_sizes[idx])
        return bin_search_gc_size(size, idx-half, half/2);

    if(size>gc_sizes[idx+1])
        return bin_search_gc_size(size, idx+half, half/2);
  

    return gc_sizes[idx+1];
}

inline int search_gc_size(int size) {
    return bin_search_gc_size(size, 8, 4);
}


GC_Span::GC_Span(GC_Arena *arena, int obj_size) : arena(arena), obj_size(obj_size) {
    N=0;
    pages=0;

    while (N<32&&pages<4) {
        pages++;
        N = size / (obj_size*pages);
    }
    N = N*pages;

    // std::cout << "\nAllocate span with\n\tObj size " << obj_size << "\n\tN: " << N << "\n\tPages: " << pages << "\n\n\n";

    

    // Get Span address
    span_address = static_cast<char*>(arena->arena) + arena->size_allocated;
    arena->size_allocated += obj_size*pages;

    // Get & initialize mark-bits
    words = (N + word_bits-1) / word_bits;
    mark_bits = (uint64_t*)malloc(words*sizeof(uint64_t));
    for (int i=0; i<words; ++i)
       mark_bits[i] = 0ULL; 
}


inline int find_free(uint64_t *mark_bits, const int words) {
    for (size_t w = 0; w < words; ++w) {
        if (~mark_bits[w]) { // if any bit is 0
            for (size_t b = 0; b < 64; ++b) {
                if (!(mark_bits[w] & (1ULL << b)))
                    return w * 64 + b;
            }
        }
    }
    return -1; // no free slot
}

inline void mark_bits_alloc(uint64_t *mark_bits, const int idx) {
    size_t w = idx / 64;
    size_t b = idx % 64;
    mark_bits[w] |= (1ULL << b);
}

// Unmark (free) a slot
inline void mark_bits_free(uint64_t *mark_bits, const int idx) {
    size_t w = idx / 64;
    size_t b = idx % 64;
    mark_bits[w] &= ~(1ULL << b);
}


void *GC_Span::Allocate() {

    int free_idx = find_free(mark_bits, words);
    // std::cout << "Found free idx: " << free_idx << ".\n";

    if (free_idx==-1)
        return nullptr;

    mark_bits_alloc(mark_bits, free_idx);
    

    return static_cast<char*>(span_address) + obj_size*free_idx;
}


GC_Arena::GC_Arena() {
    std::cout << "init arena size " << arena_size << ".\n";
    // arena = calloc(arena_size, 1);
    arena = malloc(arena_size);
}

inline bool Check_Arena_Size_Ok(const int size_allocated) {
    if(size_allocated>65536) {
        // LogErrorC(1, "Arena overflow");
        return false;
    }
    return true;
}

void *GC_Arena::Allocate(int size) {
    GC_Span *span;
    if (Spans.count(size)==0) {
        span = new GC_Span(this, size);
        if (!Check_Arena_Size_Ok(size_allocated)) {
            free(span);
            return nullptr;
        }
        Spans.emplace(size, std::vector<GC_Span*>{span});
        return span->Allocate();
    }

    int spans_count = Spans[size].size();

    void *ptr=nullptr;
    int i=0;
    while(ptr==nullptr) {
        if(i<spans_count) {
            span = Spans[size][i];
        } else {
            span = new GC_Span(this, size);
            if (!Check_Arena_Size_Ok(size_allocated)) {
                free(span);
                return nullptr;
            }
            Spans[size].push_back(span);
        }
        ptr = span->Allocate();
        ++i;
    }
    
    return ptr;
}


void *GC::Allocate(int size) {
    int obj_size = GC_size_to_class[(size+7)/8];
    // std::cout << "Allocate size: " << size << "/" << obj_size << ".\n";

    if(size>GC_max_object_size) {
        LogErrorC(-1, "Allocated object of size " + std::to_string(size) + ", but the maximum supported object size is " + std::to_string(GC_max_object_size) + ".");
        return nullptr;
    }


    void *address=nullptr;
    for (const auto &arena : arenas) {
        address = arena->Allocate(obj_size);
        std::cout << "Arena: " << arena->arena << ".\n";
        if (address!=nullptr)
            break;
    }



    if (address==nullptr) {
        LogErrorC(-1, "Failed, acquire new arena.");
        GC_Arena *new_arena = new GC_Arena();

        arenas.push_back(new_arena);
        address = new_arena->Allocate(obj_size);
    }

    // std::cout << "got addr " << address << ".\n";
    
    // address = arenas[0]->Allocate(48);
    // address = arenas[0]->Allocate(8);

    return address;
}


extern "C" void scope_struct_Alloc_GC(Scope_Struct *scope_struct) {
    scope_struct->_gc = new GC();
}

GC::GC() {
    arenas.push_back( new GC_Arena() );
}




//---------------------------------------------------------//

MarkSweep_Node::MarkSweep_Node(std::string type, bool marked) : type(type), marked(marked) {}
GC_Node::GC_Node(void *ptr, std::string type) : ptr(ptr), type(type) {}


inline void gc_list(void *ptr, const std::string &root_type, std::vector<GC_Node> &work_list, std::unordered_map<void *,MarkSweep_Node> &mark_sweep_dict) {
    if (root_type=="list") {
        DT_list *list = static_cast<DT_list*>(ptr);
        for (int i=0; i<list->size; ++i) {
            const char *type = list->data_types->at(i).c_str(); 
            if(!strcmp(type, "list")) {
                gc_list(list->get<void*>(i), "list", work_list, mark_sweep_dict);
                continue;
            }
            if(strcmp(type, "int")&&strcmp(type, "float")&&strcmp(type, "bool"))
                mark_sweep_dict.emplace(list->get<void*>(i), MarkSweep_Node(type, true));
            //     work_list.push_back(GC_Node(list->get<void*>(i), type));    
        }
    }
}


void mark_worklist_pointers(std::vector<GC_Node> &work_list, std::unordered_map<void *,MarkSweep_Node> &mark_sweep_dict) {

    for (int i=0; i<work_list.size(); ++i) {
        GC_Node node = work_list[i];
        mark_sweep_dict.emplace(node.ptr, MarkSweep_Node(node.type, true));
        // if (node.type=="tensor")
            // std::cout << "nested root: " << node.type << "/" << node.ptr << ".\n";

        if (ClassPointers.count(node.type)>0) {
            for (int j=0; j<ClassPointers[node.type].size(); ++j) {
                int offset = ClassPointers[node.type][j];
                std::string type = ClassPointersType[node.type][j];
                // std::cout << "add nested " << type << " from " << node.type << ".\n";

                void **slot = (void **)(static_cast<char*>(node.ptr)+offset);

                if(check_initialized_field(slot))
                    work_list.push_back(GC_Node(*slot, type));
            }
        }
        gc_list(node.ptr, node.type, work_list, mark_sweep_dict);
    }
}


void get_recursive_roots(Scope_Struct *scope_struct, std::unordered_map<void *,MarkSweep_Node> &mark_sweep_dict) {
    std::vector<GC_Node> work_list;
    
    for (GC_Node root : scope_struct->gc.root_nodes)
    {
        mark_sweep_dict.emplace(root.ptr, MarkSweep_Node(root.type, true));
        // if (root.type=="tensor")
            // std::cout << "root: " << root.type << "/" << root.ptr << ".\n";
        // std::cout << "root type: " << root.type << ".\n";

        if (ClassPointers.count(root.type)>0) {
            for (int i=0; i<ClassPointers[root.type].size(); ++i) {
                int offset = ClassPointers[root.type][i];
                std::string type = ClassPointersType[root.type][i];
                
                void **slot = (void **)(static_cast<char*>(root.ptr)+offset);
                
                if(check_initialized_field(slot))
                    work_list.push_back(GC_Node(*slot, type));
            }
        }

        gc_list(root.ptr, root.type, work_list, mark_sweep_dict);
    }

    mark_worklist_pointers(work_list, mark_sweep_dict);

    if (scope_struct->previous_scope!=nullptr)
        get_recursive_roots(scope_struct->previous_scope, mark_sweep_dict);
}


void get_recursive_pointers(Scope_Struct *scope_struct, std::unordered_map<void *,MarkSweep_Node> &mark_sweep_dict) {
    for (GC_Node pointer : scope_struct->gc.pointer_nodes)
        mark_sweep_dict.emplace(pointer.ptr, MarkSweep_Node(pointer.type, false));

    scope_struct->gc.pointer_nodes.clear();
    
    if (scope_struct->previous_scope!=nullptr) { // clear only current thread pointers.
        if (scope_struct->previous_scope->thread_id==scope_struct->thread_id)
            get_recursive_pointers(scope_struct->previous_scope, mark_sweep_dict);
    }
}

void GarbageCollector::sweep(Scope_Struct *scope_struct) {
    if(scope_struct->thread_id!=0)
        return;

    // std::cout << "\n\nclean scope struct: " << scope_struct << "---\n\n\n";
    std::unordered_map<void *, MarkSweep_Node> mark_sweep_dict;
    get_recursive_roots(scope_struct, mark_sweep_dict);

    scope_struct->gc.root_nodes.clear();
    get_recursive_pointers(scope_struct, mark_sweep_dict);

    Scope_Struct *inner_most_scope = get_inner_most_scope(scope_struct);
    // std::cout << "innermost size: " << inner_most_scope->gc.size_occupied << ".\n";
    // std::cout << "--" << "\n";

    inner_most_scope->gc.size_occupied = 0;
    inner_most_scope->gc.allocations = 0;

    for (const auto &pair : mark_sweep_dict) {
        void *ptr = pair.first;
        MarkSweep_Node node = pair.second;
        // std::cout << "found ptr: " << ptr << ".\n";
        if(ptr==nullptr)
            continue;

        if (node.marked) {
            inner_most_scope->gc.pointer_nodes.push_back(GC_Node(ptr, node.type));
            // inner_most_scope->gc.allocations++;
        } else {

            // std::cout << "\n\nclean: " << ptr << ".\n";
            // std::cout << "of type: " << node.type << ".\n";
            // // std::cout << "is class: " << ClassPointers.count(node.type) << ".\n\n";

            if (node.type=="str"||ClassPointers.count(node.type)>0)
                free(ptr);
            else
            {
                // std::cout << "--CLEANING: " << node.type << ".\n";
                // std::cout << "--CLEANING: " << node.type << "/" << ptr << ".\n";
                clean_up_functions[node.type](ptr); 
                // std::cout << "-\n\n";
            }
            
        }
    }

    // std::cout << "---" << "\n";
}
