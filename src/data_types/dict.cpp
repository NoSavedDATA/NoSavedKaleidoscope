#include <any>
#include <cstring>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include "../char_pool/char_pool.h"
#include "../compiler_frontend/logging.h"
#include "../clean_up/clean_up.h"
#include "../mangler/scope_struct.h"

#include "any_map.h"
#include "list.h"







extern "C" DT_dict *dict_Create(Scope_Struct *scope_struct) {

    return nullptr;
}





extern "C" DT_dict *dict_New(Scope_Struct *scope_struct, char *key, ...)
{
    char *type;
    va_list args;
    va_start(args, key);

    if(key=="TERMINATE_VARARG")
        return nullptr;

    DT_dict *dict = new DT_dict();


    bool is_type = false;
    
    do
    {
        type = va_arg(args, char *);
        
        if (!strcmp(type, "float"))
        {
            // std::cout << "appending float" << ".\n";
            float value = va_arg(args, float);
            dict->append(key, value, type);
        } else if (!strcmp(type, "int")) {
            // std::cout << "appending float" << ".\n";
            int value = va_arg(args, int);
            dict->append(key, value, type);
        } else {
            // std::cout << "appending void *: " << type << ".\n";
            void *value = va_arg(args, void *);
            // std::cout << "decoded"  << ".\n";
            dict->append(key, std::any(value), type);
            // std::cout << "appended"  << ".\n";
        }
        
        key = va_arg(args, char *);
    } while(strcmp(key,"TERMINATE_VARARG")!=0);

    va_end(args);
    
    // std::cout << "" << ".\n";
    // dict->print();
    
    return dict;
}





extern "C" void dict_Store_Key(Scope_Struct *scope_struct, DT_dict *dict, char *key, void *value, char *type) {
    std::cout << "Store " << key << " of type " << type << "\n";
    
    dict->append(key, std::any(value), type);

}


extern "C" void dict_Store_Key_int(Scope_Struct *scope_struct, DT_dict *dict, char *key, int value) {
    dict->append(key, value, "int");
}


extern "C" void dict_Store_Key_float(Scope_Struct *scope_struct, DT_dict *dict, char *key, float value) { 
    dict->append(key, value, "float");
}


extern "C" float dict_print(Scope_Struct *scope_struct, DT_dict *dict) {
    dict->print();
    return 0;
}




extern "C" void *dict_Query(Scope_Struct *scope_struct, DT_dict *dict, char *query)
{
// std::cout << "INDEX AT " << query << ".\n";



    auto it = dict->data_types->find(query);
    if (it != dict->data_types->end()) {
        std::string type = it->second;
        

        if (type=="float")
        {
            float* float_ptr = new float(dict->get<float>(query));
            return (void*)float_ptr;
        }
        if (type=="int")
        {
            int* ptr = new int(dict->get<int>(query));
            return static_cast<void*>(ptr);
        }

        return std::any_cast<void *>((*dict->data)[query]);
    }
    
    std::string q = query;
    LogErrorEE(scope_struct->code_line, "The query " + q + " was not found at the dictionary.");

    return nullptr;
}

void dict_Clean_Up(void *data_ptr) {
    if (data_ptr==nullptr)
        return;

    // DT_dict *dict = static_cast<DT_dict *>(data_ptr);
    // delete dict;
}
