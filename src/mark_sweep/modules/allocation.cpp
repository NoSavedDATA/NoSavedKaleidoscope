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





int gc_sizes[GC_obj_sizes];
uint16_t GC_size_to_class[GC_N+1];
std::unordered_map<int, GC_span_traits*> GC_span_traits_vec;

std::unordered_map<int, std::vector<char *>> arena_base_addr;






void *GC_Span::Allocate(uint16_t type_id) {
    // int free_idx = find_free(mark_bits, words);
    int free_idx = find_free_16_l2(type_metadata, type_words);
    if (free_idx==-1)
        return nullptr;

    // std::cout << "" << free_idx << "/" << traits->obj_size << "/" << traits->obj_size*free_idx << "\n";
    
    set_16_r12_mark(type_metadata, free_idx, type_id);
    return static_cast<char*>(span_address) + traits->obj_size*free_idx;
}





inline bool Check_Arena_Size_Ok(const int arena_size, const int size_allocated) {
    if(size_allocated>arena_size) {
        // LogErrorC(1, "Arena overflow");
        return false;
    }
    return true;
}


void *GC_Arena::Allocate(int size, uint16_t type_id) {
    GC_Span *span;
    GC_span_traits *traits = GC_span_traits_vec[size];
    // std::cout << "got traits of size " << traits->obj_size << "/" << traits->size << ".\n";

    if (Spans.count(size)==0) {
        if (!Check_Arena_Size_Ok(arena_size, size_allocated+traits->size))
            return nullptr;        
        // std::cout << "\n\n\n\n----NEW ARENA" << "-----.\n\n\n";
        span = new GC_Span(this, traits);
        Spans.emplace(size, std::vector<GC_Span*>{span});
        // std::exit(0);
        return span->Allocate(type_id);
    }

    int spans_count = Spans[size].size();

    void *ptr=nullptr;
    int i=0;
    while(ptr==nullptr) {
        if(i<spans_count)
            span = Spans[size][i];
        else {
            if (!Check_Arena_Size_Ok(arena_size, size_allocated+traits->size))
                return nullptr;
            span = new GC_Span(this, traits);
            Spans[size].push_back(span);
        }
        ptr = span->Allocate(type_id);
        ++i;
    }
    
    return ptr;
}


void *GC::Allocate(int size, uint16_t type_id, int tid) {
    int obj_size = GC_size_to_class[(size+7)/8];
    // std::cout << "Allocate size: " << size << "/" << obj_size << "/" << type_id << ".\n";

    if(size>GC_max_object_size) {
        LogErrorC(-1, "Allocated object of size " + std::to_string(size) + ", but the maximum supported object size is " + std::to_string(GC_max_object_size) + ".");
        return nullptr;
    }


    void *address=nullptr;
    for (const auto &arena : arenas) {
        address = arena->Allocate(obj_size, type_id);
        // std::cout << "Arena: " << arena->arena << ".\n";
        if (address!=nullptr)
            break;
    }

    if (address==nullptr) {
        GC_Arena *new_arena = new GC_Arena(tid);
        arenas.push_back(new_arena);
        address = new_arena->Allocate(obj_size, type_id);
    }
    // std::cout << "got addr " << address << ".\n";
    return address;
}

