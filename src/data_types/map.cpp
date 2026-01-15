#include <stdio.h>
#include <string>

#include "../compiler_frontend/logging_v.h"
#include "../pool/include.h"
#include "../pool/pool.h"

#include "data_tree.h"
#include "map.h"
#include "list.h"

DT_map_node::DT_map_node(int key_size, int value_size) {
    key = malloc(key_size);
    value = malloc(value_size);
}


DT_map::DT_map() {}


void DT_map::New(int size, int key_size, int value_size) {
    this->size = size;

    capacity = ((size + 7) / 8) * 8;
    if (capacity<8)
        capacity=8;

    nodes = (DT_map_node**)malloc(capacity*8); // 8 == size of one void *
    for (int i=0; i<capacity; ++i)
        nodes[i] = nullptr;
}

extern "C" DT_map *map_Create(Scope_Struct *scope_struct, char *name, char *scopeless_name, DT_map *init_val,
                                  DT_list *notes_vector, Data_Tree dt) {
    if(init_val!=nullptr)
        return init_val;
    
    if (dt.Nested_Data.size()<2)
        LogErrorC(scope_struct->code_line, "map requires key and value info");

    std::string key_type = dt.Nested_Data[0].Type;
    int key_size;
    if(data_name_to_size.count(key_type)>0)
        key_size = data_name_to_size[key_type];
    else
        key_size = 8;

    std::string value_type = dt.Nested_Data[1].Type;
    int value_size;
    if(data_name_to_size.count(value_type)>0)
        value_size = data_name_to_size[value_type];
    else
        value_size = 8;

    DT_map *map = newT<DT_map>(scope_struct, "map");
    map->New(8, key_size, value_size); 


    DT_map_node *node = new DT_map_node(key_size, value_size);
    char *key = allocate<char>(scope_struct, 2, "str");
    key[0] = 'x';
    key[1] = '\0';
    node->key = key;
    node->value = nullptr;
    map->nodes[7] = node;


    node = new DT_map_node(key_size, value_size);
    key = allocate<char>(scope_struct, 2, "str");
    key[0] = 'y';
    key[1] = '\0';
    node->key = key;
    node->value = nullptr;
    map->nodes[4] = node;
    

    return map;
}


extern "C" void print_str(char *str) {
    std::cout << "print_str: " << str << ".\n";
}

extern "C" void map_print(Scope_Struct *scope_struct, DT_map *map) {

    std::cout << "\n";
    for (int i=0; i<map->capacity; ++i) {
        DT_map_node *node = map->nodes[i];
        while (node!=nullptr) {
            char *key = static_cast<char*>(node->key);
            std::cout << "key[" << i << "]: " << key << "\n";
            std::cout << "val: " << node->value << "\n\n";
            node = node->next;
        }
    }
}

extern "C" void map_bad_key(Scope_Struct *scope_struct, char *key) {
    std::string key_str = key;
    LogErrorC(scope_struct->code_line, "Map does not contain key: " + key_str);
}

