#pragma once

#include <iostream>
#include <map>
#include <string>

#include "../compiler_frontend/logging.h"
#include "../clean_up/clean_up.h"
#include "../mangler/scope_struct.h"
#include "mark_sweep.h"



MarkSweepAtom::MarkSweepAtom(std::string data_type) : data_type(data_type) {
}

void MarkSweepAtom::inc_scopeful() {
    this->scope_refs+=1;
}
void MarkSweepAtom::inc_scopeless() {
    this->scopeless_refs+=1;
}

void MarkSweepAtom::dec_scopeful() {
    if(this->scope_refs>0)
        this->scope_refs-=1;
}
void MarkSweepAtom::dec_scopeless() {
    if(this->scopeless_refs>0)
        this->scopeless_refs-=1;
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
    {}
    else
        mark_sweep_map[data_ptr] = new MarkSweepAtom(data_type);
}


void MarkSweep::mark_scopeful(void *data_ptr, std::string data_type) {

    auto it = mark_sweep_map.find(data_ptr);
    if (it!=mark_sweep_map.end())
    {
        it->second->dec_scopeful();
        // mark_sweep_map[data_ptr] = it->second;
        if(it->second->scope_refs+it->second->scope_refs&&data_type=="tensor")
            clean_up_functions[data_type](data_ptr);
    }
    // else
    //     mark_sweep_map[data_ptr] = new MarkSweepAtom(data_type, true);
}


void MarkSweep::mark_scopeless(void *data_ptr, std::string data_type) {

    auto it = mark_sweep_map.find(data_ptr);
    if (it!=mark_sweep_map.end())
    {
        it->second->dec_scopeless();
        // mark_sweep_map[data_ptr] = it->second;

    }
    // else
    //     mark_sweep_map[data_ptr] = new MarkSweepAtom(data_type, true);

}


void MarkSweep::unmark_scopeful(void *data_ptr) {

    auto it = mark_sweep_map.find(data_ptr);

    if (it!=mark_sweep_map.end())
    {
        it->second->inc_scopeful();
    } else {
        // std::cout << "UNMARK NOT FOUND" << ".\n";
    }
}

void MarkSweep::unmark_scopeless(void *data_ptr) {

    auto it = mark_sweep_map.find(data_ptr);

    if (it!=mark_sweep_map.end())
    {
        it->second->inc_scopeless();
    } else {
        // std::cout << "UNMARK NOT FOUND" << ".\n";
    }
}


void MarkSweep::clean_up(bool clean_scopeful) {
    // std::cout << "\n\n-----clean_up" << ".\n";;



    for (auto it = mark_sweep_map.begin(); it != mark_sweep_map.end(); ) {
        MarkSweepAtom *atom = it->second;
        // std::cout << "cleaning " << it->first << " - " << atom->data_type <<  ".\n";
        // std::cout << "refs " << atom->scope_refs << "/" << atom->scopeless_refs << ".\n";

        
        if ((atom->scope_refs==0||clean_scopeful) && atom->scopeless_refs==0) {
            
            // std::cout << "delete " << it->first << " fn: " << atom->data_type << ".\n";
            clean_up_functions[atom->data_type](it->first);
            free(atom);
            it = mark_sweep_map.erase(it); // erase returns the next valid iterator
        } else {
            ++it;
        }
    }


    // std::cout << "-----" << "\n\n\n";
}





extern "C" void MarkToSweep_Mark(Scope_Struct *scope_struct, void *value, char *data_type) {
    // std::cout << "MARK OF " << data_type << ".\n";
    if (value==nullptr)
        return;
    // std::cout << "Mark to sweep of " << data_type << ".\n";
    scope_struct->mark_sweep_map->append(value, data_type);
}

extern "C" void MarkToSweep_Mark_Scopeful(Scope_Struct *scope_struct, void *value, char *data_type) {
    // std::cout << "MARK OF " << data_type << ".\n";
    if (value==nullptr)
        return;
    // std::cout << "Mark to sweep of " << data_type << ".\n";
    scope_struct->mark_sweep_map->mark_scopeful(value, data_type);
}

extern "C" void MarkToSweep_Mark_Scopeless(Scope_Struct *scope_struct, void *value, char *data_type) {
    // std::cout << "MARK OF " << data_type << ".\n";
    if (value==nullptr)
        return;
    // std::cout << "Mark to sweep of " << data_type << ".\n";
    scope_struct->mark_sweep_map->mark_scopeless(value, data_type);
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