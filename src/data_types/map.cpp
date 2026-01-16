#include <cstdint>
#include <stdio.h>
#include <string>

#include "../compiler_frontend/logging_v.h"
#include "../pool/include.h"
#include "../pool/pool.h"

#include "data_tree.h"
#include "map.h"
#include "list.h"

unsigned int str_hash(const char *str) {
    unsigned int hash = 2166136261u; // FNV-1a offset basis

    while (*str) {
        hash ^= static_cast<unsigned char>(*str);
        hash *= 16777619u;
        ++str;
    }
    return hash;
}
uint32_t float_hash(float x) {
    uint32_t bits;
    static_assert(sizeof(float) == sizeof(uint32_t));
    std::memcpy(&bits, &x, sizeof(bits));

    // Mix bits (Murmur3 finalizer-style)
    bits ^= bits >> 16;
    bits *= 0x85ebca6bU;
    bits ^= bits >> 13;
    bits *= 0xc2b2ae35U;
    bits ^= bits >> 16;

    return bits;
}


DT_map_node::DT_map_node(int key_size, int value_size) {
    key = malloc(key_size);
    value = malloc(value_size);
}

DT_map::DT_map() {}

void DT_map::New(int size, int key_size, int value_size, std::string key_type, std::string value_type) {
    this->size = size;
    this->key_size = key_size;
    this->val_size = value_size;
    this->key_type = key_type;
    this->val_type = value_type;

    capacity = ((size + 7) / 8) * 8;
    if (capacity<8)
        capacity=8;

    // expand_at = capacity*4;
    expand_at = 3;

    nodes = (DT_map_node**)malloc(capacity*8); // 8 == size of one void *
    for (int i=0; i<capacity; ++i)
        nodes[i] = nullptr;
    std::cout << "create map: " << key_size << "/" << value_size  << "/" << key_type << ".\n";
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
    map->New(0, key_size, value_size, key_type, value_type); 

    
    return map;
}


void DT_map::Insert(int hash_pos, DT_map_node *node, DT_map_node **nodes) {
    DT_map_node *cur_node = nodes[hash_pos];

    if (cur_node==nullptr)
        nodes[hash_pos] = node;
    else {
        while(cur_node->next!=nullptr)
            cur_node = cur_node->next;
        cur_node->next = node;
    }
}

extern "C" void map_expand(Scope_Struct *scope_struct, DT_map *map) {
    int capacity = map->capacity*4;

    DT_map_node **nodes = (DT_map_node**)malloc(capacity*8); // 8 == size of one void *
    for (int i=0; i<capacity; i++)
        nodes[i] = nullptr;
    
    for (int i=0; i<map->capacity; ++i) {
        DT_map_node *node = map->nodes[i];
        while (node!=nullptr) {
            unsigned int hashed;
            if (map->key_type=="str") {
                char *key = static_cast<char*>(node->key);
                hashed = str_hash(key);
            } else if (map->key_type=="int") {
                int *key = static_cast<int*>(node->key);
                hashed = *key;
            } else if (map->key_type=="float") {
                float *key = static_cast<float*>(node->key);
                hashed = float_hash(*key);
            }

            unsigned int bucket_pos = hashed % capacity;
            
            map->Insert(bucket_pos, node, nodes);
            
            DT_map_node *prev = node;
            node = node->next;
            prev->next = nullptr;
        }
    }

    free(map->nodes);
    map->nodes = nodes;
    
    map->capacity = capacity;
    map->expand_at = capacity*4;
}

extern "C" void print_str(char *str) {
    std::cout << "print_str: " << str << ".\n";
}

extern "C" void map_print(Scope_Struct *scope_struct, DT_map *map) {

    std::cout << "\n";
    for (int i=0; i<map->capacity; ++i) {
        DT_map_node *node = map->nodes[i];
        while (node!=nullptr) {
            if (map->key_type=="str") {
                char *key = static_cast<char*>(node->key);
                std::cout << "key[" << i << "]: " << key << "\n";
            } else if (map->key_type=="int") {
                int *key = static_cast<int*>(node->key);
                std::cout << "key[" << i << "]: " << *key << "\n";
            } else if (map->key_type=="float") {
                float *key = static_cast<float*>(node->key);
                std::cout << "key[" << i << "]: " << *key << "\n";
            }

            std::cout << "val: " << node->value << "\n\n";
            node = node->next;
        }
    }
}

extern "C" void map_bad_key(Scope_Struct *scope_struct, char *key) {
    std::string key_str = key;
    LogErrorC(scope_struct->code_line, "Map does not contain key: " + key_str);
}

