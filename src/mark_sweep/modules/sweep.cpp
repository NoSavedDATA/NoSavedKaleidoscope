#include <cmath>
#include <iostream>
#include <limits>
#include <map>
#include <string>
#include <unordered_map>

#include "../../common/extension_functions.h"
#include "../../compiler_frontend/global_vars.h"
#include "../../compiler_frontend/logging_v.h"
#include "../../compiler_frontend/logging.h"
#include "../../compiler_frontend/tokenizer.h"
#include "../../clean_up/clean_up.h"
#include "../../data_types/array.h"
#include "../../data_types/list.h"
#include "../../mangler/scope_struct.h"
#include "../../pool/pool.h"
#include "../include.h"







void Reset_Pools(const std::vector<GC_Arena *> &arenas) {
    for (const auto &arena : arenas) {
        for (const auto &span_pair : arena->Spans) {
            for (const auto &span : span_pair.second) {
                GC_span_traits *traits = span->traits;
                
                for (int i=0; i<traits->N; ++i) {
                    mark_bits_free(span->mark_bits, i);
                    set_16_L1(span->type_metadata, i, 0u);
                }
            }
        }
    }
}




inline void gc_list(void *ptr, const std::string &root_type, std::vector<GC_Node> &work_list, std::vector<void *> &root_nodes) {
    if (root_type=="list") {
        DT_list *list = static_cast<DT_list*>(ptr);
        for (int i=0; i<list->size; ++i) {
            const char *type = list->data_types->at(i).c_str(); 
            if(!strcmp(type, "list")) {
                gc_list(list->get<void*>(i), "list", work_list, root_nodes);
                continue;
            }
            if(!strcmp(type, "str")) {
                root_nodes.push_back(static_cast<void*>(list->get<char*>(i)));
                continue;
            }
            if(strcmp(type, "int")&&strcmp(type, "float")&&strcmp(type, "bool")) // not a primary
                root_nodes.push_back(list->get<void*>(i));
            //     work_list.push_back(GC_Node(list->get<void*>(i), type));
        }
    }
    if (root_type=="array") {
        DT_array *array = static_cast<DT_array*>(ptr);
        void **data = static_cast<void **>(array->data);
        
        if (in_str(array->type, compound_tokens)) {
            for (int i=0; i<array->virtual_size; ++i) {
                root_nodes.push_back(data[i]);
                gc_list(data[i], array->type, work_list, root_nodes);
            }
        }
        else if(!in_str(array->type, primary_data_tokens)) {
            for (int i=0; i<array->virtual_size; ++i)
                root_nodes.push_back(data[i]);
        } 
    }
}


void mark_worklist_pointers(std::vector<GC_Node> &work_list, std::vector<void *> &root_nodes) {
    for (int i=0; i<work_list.size(); ++i) {
        GC_Node &node = work_list[i];
        root_nodes.push_back(node.ptr);
        // std::cout << "push obj attr of type: " << node.type << "/" << node.ptr << ".\n";

        if (ClassPointers.count(node.type)>0) {
            for (int j=0; j<ClassPointers[node.type].size(); ++j) {
                int offset = ClassPointers[node.type][j];
                std::string type = ClassPointersType[node.type][j];

                void **slot = (void **)(static_cast<char*>(node.ptr)+offset);

                if(check_initialized_field(slot))
                    work_list.push_back(GC_Node(*slot, type));
            }
        }
        gc_list(node.ptr, node.type, work_list, root_nodes);
    }
}


void check_roots_worklist(Scope_Struct *scope_struct, std::vector<void *> &root_nodes) {
    std::vector<GC_Node> work_list;

    // std::cout << "stack top: " << scope_struct->stack_top << ".\n";
    for (int i=0; i<scope_struct->stack_top; ++i) {
        void *root_ptr = scope_struct->pointers_stack[i];
        root_nodes.push_back(root_ptr);

        // std::cout << "PUSH BACK ROOT: " << i << "/" << scope_struct->stack_top << "/" <<  root_ptr << ".\n";
        std::string root_type = get_pool_obj_type(scope_struct, root_ptr);
        // std::cout << "PUSH BACK ROOT: " << root_type << "/" << root_ptr << ".\n";
        
        if (ClassPointers.count(root_type)>0) {
            for (int i=0; i<ClassPointers[root_type].size(); ++i) {
                int offset = ClassPointers[root_type][i];
                std::string type = ClassPointersType[root_type][i];
                
                void **slot = (void **)(static_cast<char*>(root_ptr)+offset);
                
                if(check_initialized_field(slot))
                    work_list.push_back(GC_Node(*slot, type));
            }
        }
        gc_list(root_ptr, root_type, work_list, root_nodes);
    }
    mark_worklist_pointers(work_list, root_nodes);

    // for (void &root : scope_struct->root_nodes)
    // {
    //     root_nodes.push_back(GC_Node(root.ptr, root.type));
    //     if (ClassPointers.count(root.type)>0) {
    //         for (int i=0; i<ClassPointers[root.type].size(); ++i) {
    //             int offset = ClassPointers[root.type][i];
    //             std::string type = ClassPointersType[root.type][i];
                
    //             void **slot = (void **)(static_cast<char*>(root.ptr)+offset);
                
    //             if(check_initialized_field(slot))
    //                 work_list.push_back(GC_Node(*slot, type));
    //         }
    //     }
    //     gc_list(root.ptr, root.type, work_list, root_nodes);
    // }
    // mark_worklist_pointers(work_list, root_nodes);
}




void GC::Sweep(Scope_Struct *scope_struct) {
    Reset_Pools(arenas);  

    // std::cout << "sweep sweep sweep thread " << scope_struct->thread_id << ".\n";
    // std::cout << "sweep thread: " << scope_struct->thread_id << ".\n";
    // std::cout << "Has " << arenas.size() << " arenas.\n";

    // std::cout << "\n\n\n---------------------------------------------" << "\n\n\n";

    std::vector<void *> root_nodes;

    check_roots_worklist(scope_struct, root_nodes);
    char *arena_addr = arena_base_addr[scope_struct->thread_id];

    for (void * node_ptr : root_nodes) {
        long arena_offset = static_cast<char*>(node_ptr) - arena_addr;
        int arena =  arena_offset / GC_arena_size;
        if (arena<0) {
            // LogErrorC(scope_struct->code_line, "Variable of type " + node.type + " address does not reside in any memory pool.");
            std::cout << "Variable of type " << node_ptr << " address does not reside in any memory pool..\n";
            std::cout << node_ptr << ", arena: " << arena << ".\n";
            std::exit(0);
        }

        int page  =  (arena_offset / GC_page_size) % pages_per_arena;
        // std::cout << "Belongs to arena: " << arena << ".\n";
        // std::cout << "Belongs to page: " << page << ".\n";
 
        GC_Span *span = arenas[arena]->page_to_span[page];

        long obj_idx = (static_cast<char*>(node_ptr) - static_cast<char*>(span->span_address)) / span->traits->obj_size;
        // std::cout << "Obj idx in span: " << obj_idx << ".\n";
        // std::cout << "span: " << span << ".\n";
        // std::cout << "span obj_size " << span->traits->obj_size << ", pages: " << span->traits->pages << ", N: " << span->traits->N << ".\n";
        set_16_L1(span->type_metadata, obj_idx, 1u);
    }


    CleanUp_Unused(); // Trigger clean_up functions
    allocations=0;
    size_occupied=0;
}


void GC::CleanUp_Unused() {
    
    for (const auto &arena : arenas) {
        for (const auto &span_pair : arena->Spans) {
            for (GC_Span *span : span_pair.second) {
                GC_span_traits *traits = span->traits;
                int free_idx = find_free_16_l2(span->type_metadata, span->type_words);

                for (int i=0; i<traits->N; ++i) {
                    if (get_16_l2(span->type_metadata, i)==0) {
                        uint16_t u_type = get_16_r12(span->type_metadata, i);

                        if(u_type!=0) {
                            std::string obj_type = data_type_to_name[u_type]; 
                            if(obj_type!="str"&&ClassPointers.count(obj_type)==0) {
                                void *obj_addr = static_cast<char*>(span->span_address) + i*traits->obj_size;
                                if (obj_type!="list"&&obj_type!="tensor")
                                std::cout << "CLEAN: addr " << obj_addr << " got object: " << u_type << "/" << obj_type << ".\n";
                                clean_up_functions[obj_type](obj_addr);
                            }
                        }
                    }
                }
            }
        }
    }
}
