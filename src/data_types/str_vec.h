#pragma once

#include <vector>

#include "../mangler/scope_struct.h"


extern "C" float str_vec_Create(char *name, char *scopeless_name, float init_val, AnyVector *notes_vector, Scope_Struct *);
extern "C" void *str_vec_Load(char *, Scope_Struct *);
extern "C" void str_vec_Store(char *, std::vector<char *>, Scope_Struct *);



extern "C" float PrintStrVec(std::vector<char*> vec);


extern "C" float LenStrVec(std::vector<char*> vec);




extern "C" void * ShuffleStrVec(std::vector<char*> vec);



//deprecated
extern "C" char * shuffle_str(char *string_list);


extern "C" void * _glob_b_(char *pattern);
