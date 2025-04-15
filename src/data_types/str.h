#pragma once

#include "../mangler/scope_struct.h"

extern "C" float str_Create(char *name, char *scopeless_name, char *init_val, AnyVector *notes_vector, Scope_Struct *);
extern "C" void *str_Load(char *, Scope_Struct *);
extern "C" float str_Store(char *, char *, Scope_Struct *);




extern "C" float PrintStr(char* value);


extern "C" float *split_str_to_float(char *in_string, int gather_position);



extern "C" void *cat_str_float(char *c, float v);




extern "C" void *SplitString(char *self, char *pattern);

extern "C" char *SplitStringIndexate(char *name, char *pattern, float idx);


extern "C" float StrToFloat(char *in_str);
