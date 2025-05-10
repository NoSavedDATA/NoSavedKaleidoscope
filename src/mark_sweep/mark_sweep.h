#pragma once

#include <map>
#include <string>

#include "../mangler/scope_struct.h"


struct MarkSweepAtom {
    std::string data_type;
    bool marked;
    int references=0;

    MarkSweepAtom(const std::string &, bool);

    void inc();
    void dec();
};


struct MarkSweep {
    
    std::map<void *, MarkSweepAtom *> mark_sweep_map;

    void append(void *, std::string);
    void unmark(void *);
    void clean_up();
};