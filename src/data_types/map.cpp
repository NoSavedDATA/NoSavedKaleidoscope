#include <stdio.h>
#include <string>

#include "../compiler_frontend/logging_v.h"
#include "../pool/include.h"

#include "data_tree.h"
#include "map.h"
#include "list.h"

DT_map::DT_map() {}

void DT_map::New(int size, int key_size, int value_size) {
    this->size = size;

    capacity = ((size + 7) / 8) * 8;


    keys = malloc(capacity*sizeof(key_size));
    values = malloc(capacity*sizeof(value_size));
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
    map->New(4, key_size, value_size);
    

    char *c = allocate<char>(scope_struct, 2, "str");
    c[0] = 'x';
    c[1] = '\0';

    char *c1 = allocate<char>(scope_struct, 2, "str");
    c1[0] = 'y';
    c1[1] = '\0';

    char *c2 = allocate<char>(scope_struct, 14, "str");
    c2[0] = 'a';
    c2[1] = ' ';
    c2[2] = 'l';
    c2[3] = 'o';
    c2[4] = 'n';
    c2[5] = 'g';
    c2[6] = ' ';
    c2[7] = 's';
    c2[8] = 't';
    c2[9] = 'r';
    c2[10] = 'i';
    c2[11] = 'n';
    c2[12] = 'g';
    c2[13] = '\0';

    char *c3 = allocate<char>(scope_struct, 2, "str");
    c3[0] = 'z';
    c3[1] = '\0';

    char **keys = static_cast<char**>(map->keys);
    keys[0] = c;
    keys[1] = c2;
    keys[2] = c1;
    keys[3] = c3;

    
    void **values = static_cast<void**>(map->values);
    values[0] = nullptr;
    values[1] = nullptr;
    values[2] = nullptr;
    values[3] = nullptr;

    std::cout << "returning map with size: " << map->size << ".\n";
    return map;
}


extern "C" void print_str(char *str) {
    std::cout << "print_str: " << str << ".\n";
}

extern "C" void map_print(Scope_Struct *scope_struct, DT_map *map) {

    std::cout << "\n";
    for (int i=0; i<map->size; ++i) {
        char **keys = static_cast<char**>(map->keys);
        void **values = static_cast<void**>(map->values);
        std::cout << "key[" << i << "]: " << keys[i] << "\n";
        std::cout << "val: " << values[i] << "\n\n";
    }
}
