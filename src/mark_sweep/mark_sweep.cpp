#pragma once

#include <iostream>
#include <map>
#include <string>

#include "../clean_up/clean_up.h"
#include "../mangler/scope_struct.h"
#include "mark_sweep.h"


MarkSweepAtom::MarkSweepAtom(const std::string & data_type, bool marked) {
    this->data_type = data_type;
    this->marked = marked;
}

void MarkSweepAtom::inc() {
    this->references+=1;
}
void MarkSweepAtom::dec() {
    if(this->references>0)
        this->references-=1;
}



void MarkSweep::append(void *data_ptr, std::string data_type) {

    // std::cout << "Mark to sweep of " << data_type << ".\n";

    auto it = mark_sweep_map.find(data_ptr);
    if (it!=mark_sweep_map.end())
    {
        // std::cout << "DECREMENT" << ".\n";
        it->second->dec();
    }
    else
        mark_sweep_map[data_ptr] = new MarkSweepAtom(data_type, true);
}


void MarkSweep::unmark(void *data_ptr) {

    auto it = mark_sweep_map.find(data_ptr);

    if (it!=mark_sweep_map.end())
        it->second->inc();
}


void MarkSweep::clean_up() {
    // std::cout << "clean_up" << ".\n";


    for (auto &it : mark_sweep_map)
    {
        // if(it.second->data_type!="str")
        //     std::cout << "IT TYPE IS " << it.second->data_type << ".\n";
        if(it.second->references==0)
            clean_up_functions[it.second->data_type](it.first);
        free(it.second);
    }
}



extern "C" void MarkToSweep_Mark(Scope_Struct *scope_struct, void *value, char *data_type) {
    if (value==nullptr)
        return;
    // std::cout << "Mark to sweep of " << data_type << ".\n";
    scope_struct->mark_sweep_map->append(value, data_type);
}


extern "C" void MarkToSweep_Unmark(Scope_Struct *scope_struct, void *value) {
    if (value==nullptr)
        return;
    // std::cout << "Mark to sweep of " << data_type << ".\n";
    scope_struct->mark_sweep_map->unmark(value);
}