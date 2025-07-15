#pragma once

#include <map>
#include <unordered_map>
#include <string>

#include "../mangler/scope_struct.h"


struct MarkSweepAtom {
    std::string data_type;
    bool marked;
    int scope_refs=0;
    int scopeless_refs=0;

    MarkSweepAtom(const std::string &, bool);

    void inc();
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