#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <vector>
#include <string>

#include "../codegen/string.h"
#include "../common/extension_functions.h"
#include "../mangler/scope_struct.h"
#include "../pool/include.h"
#include "include.h"



extern "C" int read_int(Scope_Struct *scope_struct) {
    int value;
    if (scanf("%d", &value) != 1) {
        fprintf(stderr, "Failed to read int\n");
        return 0; // default on failure
    }
    return value;
}


extern "C" char *int_to_str(Scope_Struct *scope_struct, int x) {
    // Enough for 32-bit int including sign and null terminator
    char buffer[32];  
    int len = std::snprintf(buffer, sizeof(buffer), "%d", x);

    if (len < 0) return nullptr;  // snprintf error

    char *result = allocate<char>(scope_struct, len+1, "str");
    if (!result) return nullptr;  // malloc failed

    std::memcpy(result, buffer, len + 1); // copy with '\0'
    return result;
}
