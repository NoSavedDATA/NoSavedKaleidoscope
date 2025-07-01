#include "include.h"



extern "C" float testing_fn(Scope_Struct *scope_struct, float y) {
    
    std::cout << "testing_fn" << ".\n";
    
    return 0;
}


extern "C" DT_tensor *opa_gangnamstyel(Scope_Struct *scope_struct, DT_tensor *x, float y, DT_tensor *z) {
    
    std::cout << "testing_fn" << ".\n";
    
    return 0;
}