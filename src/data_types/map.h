#pragma once

#include <string>

struct DT_map {
    int size;
    int capacity;
    int key_size;    // sizeof(key)
    int val_size;    // sizeof(value)
    void *keys;
    void *values;
    uint8_t *states; // 0=empty, 1=filled, 2=tombstone

    DT_map();
    void New(int, int, int);
};


