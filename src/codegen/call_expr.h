#pragma once

#include "../mangler/scope_struct.h"
#include"include.h"


extern "C" char * FirstArgOnDemand(Scope_Struct *, char *pre_dotc, char *_class, char *method, int nested_function, int isSelf, int isAttribute);
