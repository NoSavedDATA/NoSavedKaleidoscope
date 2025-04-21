#pragma once

#include <iostream>
#include <vector>


#include "../mangler/scope_struct.h"


extern "C" float float_vec_Create(char *name, char *scopeless_name, float init_val, AnyVector *notes_vector, Scope_Struct *);
extern "C" void *float_vec_Load(char *, Scope_Struct *);



extern "C" float float_vec_Store(char *, std::vector<float>, Scope_Struct *);

extern "C" float float_vec_Store_Idx(char *, float, float, Scope_Struct *);





extern "C" float PrintFloatVec(std::vector<float> vec);

extern "C" void * zeros_vec(Scope_Struct *, float size);
extern "C" void * ones_vec(Scope_Struct *, float size);



extern "C" float IndexClassFloatVec(char *vec_name, float _idx);



extern "C" float float_vec_CalculateIdx(char *, float, ...); 


extern "C" float float_vec_first_nonzero(Scope_Struct *, std::vector<float>);