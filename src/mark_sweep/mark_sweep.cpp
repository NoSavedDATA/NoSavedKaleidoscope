#pragma once

#include <iostream>
#include <map>
#include <string>
#include <unordered_map>

#include "../compiler_frontend/global_vars.h"
#include "../compiler_frontend/logging.h"
#include "../clean_up/clean_up.h"
#include "../data_types/list.h"
#include "../mangler/scope_struct.h"
#include "../pool/pool.h"
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
    for (auto it = mark_sweep_map.begin(); it != mark_sweep_map.end(); ) {
        MarkSweepAtom *atom = it->second;

        
        if ((atom->scope_refs==0||clean_scopeful) && atom->scopeless_refs==0) {
            
            // clean_up_functions[atom->data_type](it->first);
            // free(atom);
            // it = mark_sweep_map.erase(it); // erase returns the next valid iterator
        } else {
            ++it;
        }
    }
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



//---------------------------------------------------------//

MarkSweep_Node::MarkSweep_Node(std::string type, bool marked) : type(type), marked(marked) {}
GC_Node::GC_Node(void *ptr, std::string type) : ptr(ptr), type(type) {}


inline void gc_list(void *ptr, const std::string &root_type, std::vector<GC_Node> &work_list, std::unordered_map<void *,MarkSweep_Node> &mark_sweep_dict) {
    if (root_type=="list") {
        DT_list *list = static_cast<DT_list*>(ptr);
        for (int i=0; i<list->size; ++i) {
            const char *type = list->data_types->at(i).c_str(); 
            if(!strcmp(type, "list")) {
                gc_list(list->get<void*>(i), "list", work_list, mark_sweep_dict);
                continue;
            }
            if(strcmp(type, "int")&&strcmp(type, "float")&&strcmp(type, "bool"))
                mark_sweep_dict.emplace(list->get<void*>(i), MarkSweep_Node(type, true));
            //     work_list.push_back(GC_Node(list->get<void*>(i), type));    
        }
    }
}


void mark_worklist_pointers(std::vector<GC_Node> &work_list, std::unordered_map<void *,MarkSweep_Node> &mark_sweep_dict) {

    for (int i=0; i<work_list.size(); ++i) {
        GC_Node node = work_list[i];
        mark_sweep_dict.emplace(node.ptr, MarkSweep_Node(node.type, true));
        // if (node.type=="tensor")
            // std::cout << "nested root: " << node.type << "/" << node.ptr << ".\n";

        if (ClassPointers.count(node.type)>0) {
            for (int j=0; j<ClassPointers[node.type].size(); ++j) {
                int offset = ClassPointers[node.type][j];
                std::string type = ClassPointersType[node.type][j];
                // std::cout << "add nested " << type << " from " << node.type << ".\n";

                void **slot = (void **)(static_cast<char*>(node.ptr)+offset);

                if(check_initialized_field(slot))
                    work_list.push_back(GC_Node(*slot, type));
            }
        }
        gc_list(node.ptr, node.type, work_list, mark_sweep_dict);
    }
}


void get_recursive_roots(Scope_Struct *scope_struct, std::unordered_map<void *,MarkSweep_Node> &mark_sweep_dict) {
    std::vector<GC_Node> work_list;
    
    for (GC_Node root : scope_struct->gc.root_nodes)
    {
        mark_sweep_dict.emplace(root.ptr, MarkSweep_Node(root.type, true));
        // if (root.type=="tensor")
            // std::cout << "root: " << root.type << "/" << root.ptr << ".\n";
        // std::cout << "root type: " << root.type << ".\n";

        if (ClassPointers.count(root.type)>0) {
            for (int i=0; i<ClassPointers[root.type].size(); ++i) {
                int offset = ClassPointers[root.type][i];
                std::string type = ClassPointersType[root.type][i];
                
                void **slot = (void **)(static_cast<char*>(root.ptr)+offset);
                
                if(check_initialized_field(slot))
                    work_list.push_back(GC_Node(*slot, type));
            }
        }

        gc_list(root.ptr, root.type, work_list, mark_sweep_dict);
    }

    mark_worklist_pointers(work_list, mark_sweep_dict);

    if (scope_struct->previous_scope!=nullptr)
        get_recursive_roots(scope_struct->previous_scope, mark_sweep_dict);
}


void get_recursive_pointers(Scope_Struct *scope_struct, std::unordered_map<void *,MarkSweep_Node> &mark_sweep_dict) {
    for (GC_Node pointer : scope_struct->gc.pointer_nodes)
        mark_sweep_dict.emplace(pointer.ptr, MarkSweep_Node(pointer.type, false));

    scope_struct->gc.pointer_nodes.clear();
    
    if (scope_struct->previous_scope!=nullptr) { // clear only current thread pointers.
        if (scope_struct->previous_scope->thread_id==scope_struct->thread_id)
            get_recursive_pointers(scope_struct->previous_scope, mark_sweep_dict);
    }
}

void GarbageCollector::sweep(Scope_Struct *scope_struct) {
    if(scope_struct->thread_id!=0)
        return;

    // std::cout << "\n\nclean scope struct: " << scope_struct << "---\n\n\n";
    std::unordered_map<void *, MarkSweep_Node> mark_sweep_dict;
    get_recursive_roots(scope_struct, mark_sweep_dict);

    scope_struct->gc.root_nodes.clear();
    get_recursive_pointers(scope_struct, mark_sweep_dict);

    Scope_Struct *inner_most_scope = get_inner_most_scope(scope_struct);
    // std::cout << "innermost size: " << inner_most_scope->gc.size_occupied << ".\n";
    // std::cout << "--" << "\n";

    inner_most_scope->gc.size_occupied = 0;
    inner_most_scope->gc.allocations = 0;

    for (const auto &pair : mark_sweep_dict) {
        void *ptr = pair.first;
        MarkSweep_Node node = pair.second;
        // std::cout << "found ptr: " << ptr << ".\n";
        if(ptr==nullptr)
            continue;

        if (node.marked) {
            inner_most_scope->gc.pointer_nodes.push_back(GC_Node(ptr, node.type));
            // inner_most_scope->gc.allocations++;
        } else {

            // std::cout << "\n\nclean: " << ptr << ".\n";
            // std::cout << "of type: " << node.type << ".\n";
            // // std::cout << "is class: " << ClassPointers.count(node.type) << ".\n\n";

            if (node.type=="str"||ClassPointers.count(node.type)>0)
                free(ptr);
            else
            {
                // std::cout << "--CLEANING: " << node.type << ".\n";
                // std::cout << "--CLEANING: " << node.type << "/" << ptr << ".\n";
                clean_up_functions[node.type](ptr); 
                // std::cout << "-\n\n";
            }
            
        }
    }

    // std::cout << "---" << "\n";
}
