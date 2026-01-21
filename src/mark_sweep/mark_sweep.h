#pragma once

#include <map>
#include <string>
#include <unordered_map>
#include <vector>


const int word_bits=64;

const int GC_page_size=8192;

const int sweep_after_alloc = 32 << 20;
const int GC_arena_size = 64 << 20;
// const int GC_arena_size = 8192;

const int pages_per_arena = GC_arena_size / GC_page_size;

const int GC_obj_sizes=15;
const int GC_max_object_size = 16384;
extern int gc_sizes[GC_obj_sizes];

extern std::unordered_map<int, std::vector<char *>> arena_base_addr;


constexpr size_t GC_ALIGN = 8; // 8-byte granularity
constexpr size_t GC_N = GC_max_object_size / GC_ALIGN;

extern uint16_t GC_size_to_class[GC_N+1];

struct Scope_Struct;
struct GC_span_traits;
struct GC_Arena;
struct GC_Node;

extern std::unordered_map<int, GC_span_traits*> GC_span_traits_vec;


struct GC_span_traits {
    int pages=0, N=0, size, obj_size;
    GC_span_traits(int);
};

struct GC_Span {
    GC_span_traits *traits;
    GC_Arena *arena;
    void *span_address;

    int words, type_words;

    // Interpretate type_metadata as int12
    uint64_t *mark_bits, *type_metadata;
    
    GC_Span(GC_Arena *, GC_span_traits *);
    void *Allocate(uint16_t);
};


struct GC_Arena {
    // Get an arena of 64MB, and set pages size to 8 KB
    const int arena_size=GC_arena_size, page=GC_page_size;
    // const int arena_size=65536, page=8192;
    int size_allocated=0,pages_allocated=0;
    void *arena, *metadata;
    std::unordered_map<int, std::vector<GC_Span*>> Spans;
    std::unordered_map<int, GC_Span*> page_to_span;

    GC_Arena(int);
    void *Allocate(int, uint16_t);
};

struct GC {
    int allocations=0;
    uint64_t size_occupied=0;
    std::vector<GC_Arena*> arenas;
    
    GC(int);
    void *Allocate(int, uint16_t, int);
    void Sweep(Scope_Struct *);
    void CleanUp_Unused();
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


void protect_pool_addr(Scope_Struct *scope_struct, void *addr);
bool unprotect_pool_addr(Scope_Struct *scope_struct, void *addr);
