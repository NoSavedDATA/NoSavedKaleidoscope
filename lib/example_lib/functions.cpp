#include "include.h"



extern "C" float testing_argless() {
    
    std::cout << "--Execution of: testing_fn argless" << ".\n";
    
    return 0;
}

extern "C" float testing_fn(Scope_Struct *scope_struct, float y) {
    
    std::cout << "testing_fn: " << y << " ULULULLULULUULU.\n";
    
    return 0;
}

