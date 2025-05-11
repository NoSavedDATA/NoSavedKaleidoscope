#include <cstdlib>

#include "../mangler/scope_struct.h"

extern "C" float _exit(Scope_Struct *scope_struct) {
    std::exit(0);
    return 0;
}