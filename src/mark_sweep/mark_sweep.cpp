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
    this->scope_refs+=1;
}
void MarkSweepAtom::inc_scopeless() {
    this->scope_refs+=1;
    this->scopeless_refs+=1;
}
void MarkSweepAtom::dec() {
    if(this->scopeless_refs>0)
        this->scopeless_refs-=1;
    if(this->scope_refs>0)
        this->scope_refs-=1;
}



void MarkSweep::append(void *data_ptr, std::string data_type) {


    auto it = mark_sweep_map.find(data_ptr);
    if (it!=mark_sweep_map.end())
    {
        // std::cout << "DECREMENT" << ".\n";
        it->second->dec();
    }
    else
        mark_sweep_map[data_ptr] = new MarkSweepAtom(data_type, true);
}


void MarkSweep::unmark_scopeful(void *data_ptr) {

    auto it = mark_sweep_map.find(data_ptr);

    if (it!=mark_sweep_map.end())
    {

        it->second->inc();
    } else {
        // std::cout << "UNMARK NOT FOUND" << ".\n";
        // auto [it, inserted] = mark_sweep_map.try_emplace(data_ptr, new MarkSweepAtom(data_type, true));
    }
}

void MarkSweep::unmark_scopeless(void *data_ptr) {

    auto it = mark_sweep_map.find(data_ptr);

    if (it!=mark_sweep_map.end())
    {
        // std::cout << "INCREMENT" << ".\n";
        it->second->inc_scopeless();
    } else {
        // std::cout << "UNMARK NOT FOUND" << ".\n";
    }
}


void MarkSweep::clean_up() {
    // std::cout << "clean_up" << ".\n";;

    for (auto it = mark_sweep_map.begin(); it != mark_sweep_map.end(); ) {
        // std::cout << "cleaning" << ".\n";
        MarkSweepAtom *atom = it->second;

        if (atom->scope_refs==0 && atom->scopeless_refs==0) {
            // std::cout << "delete " << it->first << " fn: " << atom->data_type << ".\n";
            clean_up_functions[atom->data_type](it->first);
            delete atom;
            it = mark_sweep_map.erase(it); // erase returns the next valid iterator
        } else {
            ++it;
        }
    }
}



extern "C" void MarkToSweep_Mark(Scope_Struct *scope_struct, void *value, char *data_type) {
    // std::cout << "MARK OF " << data_type << ".\n";
    if (value==nullptr)
        return;
    // // std::cout << "Mark to sweep of " << data_type << ".\n";
    scope_struct->mark_sweep_map->append(value, data_type);
}


extern "C" void MarkToSweep_Unmark_Scopeful(Scope_Struct *scope_struct, void *value) {
    if (value==nullptr)
        return;
    // std::cout << "Unmark " << value << ".\n";
    scope_struct->mark_sweep_map->unmark_scopeful(value);
}

extern "C" void MarkToSweep_Unmark_Scopeless(Scope_Struct *scope_struct, void *value) {
    if (value==nullptr)
        return;
    // std::cout << "Unmark " << value << ".\n";
    scope_struct->mark_sweep_map->unmark_scopeless(value);
}