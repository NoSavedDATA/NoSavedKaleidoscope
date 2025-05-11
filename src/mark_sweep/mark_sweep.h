#pragma once

#include <map>
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
};


struct MarkSweep {
    
    std::map<void *, MarkSweepAtom *> mark_sweep_map;

    void append(void *, std::string);
    void unmark_scopeful(void *);
    void unmark_scopeless(void *);
    void clean_up();
};