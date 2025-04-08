#include"../char_pool/include.h"

#include "scope_mangler.h"


Scope_Mangler::Scope_Mangler() {
    first_arg = get_from_char_pool(1,"Scope mangler first_arg");
    scope = get_from_char_pool(1,"Scope mangler scope");
    previous_scope = get_from_char_pool(1,"Scope mangler previous scope");

    first_arg[0] = '\0';
    scope[0] = '\0';
    previous_scope[0] = '\0';
}


extern "C" Scope_Mangler *scope_mangler_Create() {
    Scope_Mangler *scope_mangler = new Scope_Mangler();
    std::cout << "Created scope mangler" << ".\n";
    return scope_mangler;
}