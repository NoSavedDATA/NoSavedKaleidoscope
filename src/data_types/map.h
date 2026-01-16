#pragma once

#include <string>

struct DT_map_node {
    void *key, *value;
    DT_map_node *next=nullptr;

    DT_map_node(int, int);
};

struct DT_map {
    int size;
    int capacity;
    int expand_at;
    int key_size;    // sizeof(key)
    int val_size;    // sizeof(value)
    DT_map_node **nodes;
    std::string key_type;    // sizeof(key)
    std::string val_type;    // sizeof(value)

    DT_map();
    void New(int, int, int, std::string, std::string);
    void Insert(int hash_pos, DT_map_node *node, DT_map_node **nodes);
};


