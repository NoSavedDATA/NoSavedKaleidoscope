#pragma once

#include <cstdarg>


#include "../mangler/scope_struct.h"
#include "../tensor/tensor_struct.h"





extern "C" float tuple_New(Scope_Struct *, char *, ...);

extern "C" float tuple_Create(char *name, char *scopeless_name, float init_val, AnyVector *notes_vector, Scope_Struct *scope_struct);