#pragma once

#include <map>
#include <string>
#include <unordered_map>
#include <vector>


const int word_bits=64;

const int GC_obj_sizes=15;
const int GC_max_object_size = 16384;
extern int gc_sizes[GC_obj_sizes];


constexpr size_t GC_ALIGN = 8; // 8-byte granularity
constexpr size_t GC_N = GC_max_object_size / GC_ALIGN;

extern uint16_t GC_size_to_class[GC_N+1];


struct Scope_Struct;
struct GC_Arena;


struct GC_Span {
    GC_Arena *arena;
    void *span_address;
    const int size=8192;
    int obj_size, pages, N;

    int words;

    uint64_t *mark_bits;
    
    GC_Span(GC_Arena *, int);
    void *Allocate();
};


struct GC_Arena {
    // Get an arena of 64MB, and set pages size to 8 KB
    // const int arena_size=67108864, page=8192;
    const int arena_size=65536, page=8192;
    int size_allocated;
    void *arena, *metadata;
    std::unordered_map<int, std::vector<GC_Span*>> Spans;

    GC_Arena();
    void *Allocate(int);
};

struct GC {
    std::vector<GC_Arena*> arenas;
    
    GC();
    void *Allocate(int);
};



//---------------------------------------------------------//

struct MarkSweep_Node {
    std::string type;
    bool marked;

    MarkSweep_Node(std::string, bool);
};

struct GC_Node{
    void *ptr;
    std::string type;
    
    GC_Node(void *, std::string);
};


struct GarbageCollector {
    int size_occupied=0, allocations=0;
    std::vector<GC_Node> root_nodes;
    std::vector<GC_Node> pointer_nodes;

    void MergeNodes();
    void sweep(Scope_Struct *);
};
