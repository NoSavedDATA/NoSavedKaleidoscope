#include <cmath>
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





extern "C" float read_float(Scope_Struct *scope_struct) {
    float value;
    if (scanf("%f", &value) != 1) {
        fprintf(stderr, "Failed to read float\n");
        return 0.0f; // default on failure
    }
    return value;
}



extern "C" char *float_to_str(Scope_Struct *scope_struct, float x) {
    // Enough for 32-bit int including sign and null terminator
    char buffer[32];  
    int len = std::snprintf(buffer, sizeof(buffer), "%f", x);

    if (len < 0) return nullptr;  // snprintf error

    char *result = allocate<char>(scope_struct, len+1, "str");
    if (!result) return nullptr;  // malloc failed

    std::memcpy(result, buffer, len + 1); // copy with '\0'
    return result;
}


extern "C" float nsk_pow(Scope_Struct *scope_struct, float x, float exponent) {
    std::cout << "pow with " << x  << " & " << exponent<< ".\n";
    return std::pow(x, exponent);
}
extern "C" float nsk_sqrt(Scope_Struct *scope_struct, float x) {
    return std::sqrt(x);
}