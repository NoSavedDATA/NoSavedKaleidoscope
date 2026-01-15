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
    int key_size;    // sizeof(key)
    int val_size;    // sizeof(value)
    DT_map_node **nodes;

    DT_map();
    void New(int, int, int);
};


