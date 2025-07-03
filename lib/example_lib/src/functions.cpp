#include <iostream>
#include <string>


#include "include.h"
#include "testing_struct.h"


extern "C" DT_placeholder *placeholder_Create(Scope_Struct *scope_struct, char *name,
                                                char *scopeless_name, DT_placeholder *init_val, DT_list *notes_vector) {


    int x, y;
    x = notes_vector->get<int>(0);
    y = notes_vector->get<int>(1);


    DT_placeholder *placeholder = new DT_placeholder(name, x, y);

    return placeholder; // Return it to save it on the stack and pass to the gc
}



extern "C" DT_placeholder *placeholder_placeholder_add(Scope_Struct *scope_struct, DT_placeholder *px, DT_placeholder *py) {

    int x = px->x + py->x;
    int y = px->y + py->y;

       
    DT_placeholder *placeholder = new DT_placeholder("", x, y);

    return placeholder;
}


extern "C" DT_placeholder *placeholder_int_add(Scope_Struct *scope_struct, DT_placeholder *px, int z) {

    int x = px->x + z;
    int y = px->y + z;

       
    DT_placeholder *placeholder = new DT_placeholder("", x, y);

    return placeholder;
}


extern "C" void placeholder_Clean_Up(void *data_ptr) {
    std::cout << "placeholder_Clean_Up" << data_ptr << ".\n";

    free(data_ptr);
}


extern "C" float placeholder_print(Scope_Struct *scope_struct, DT_placeholder *placeholder) {

    std::cout << "Placeholder " << placeholder->name << ": [" << placeholder->x << ", " << placeholder->y << "].\n";

    return 0;
}


