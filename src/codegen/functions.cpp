#include <cstdlib>

#include "../mangler/scope_struct.h"

extern "C" float _quit_(Scope_Struct *scope_struct) {
    std::exit(0);
    return 0;
}