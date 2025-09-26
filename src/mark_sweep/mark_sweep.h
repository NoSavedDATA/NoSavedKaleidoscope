#pragma once

#include <map>
#include <string>
#include <unordered_map>
#include <vector>



struct Scope_Struct;

struct MarkSweepAtom {
    std::string data_type;
    int scope_refs=0;
    int scopeless_refs=0;

    MarkSweepAtom(std::string);

    void inc_scopeful();
    void inc_scopeless();
    void dec();
    void dec_scopeful();
    void dec_scopeless();
};


struct MarkSweep {
    
    std::unordered_map<void *, MarkSweepAtom *> mark_sweep_map;

    void append(void *, std::string);
    void mark_scopeful(void *, std::string);
    void mark_scopeless(void *, std::string);
    void unmark_scopeful(void *);
    void unmark_scopeless(void *);
    void clean_up(bool);
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
    int size_occupied=0;
    std::vector<GC_Node> root_nodes;
    std::vector<GC_Node> pointer_nodes;

    void MergeNodes();
    void sweep(Scope_Struct *);
};
