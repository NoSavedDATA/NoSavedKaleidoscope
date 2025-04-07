#pragma once

#include <iostream>
#include <vector>



extern "C" float float_vec_Create(char *name, char *scopeless_name, float init_val, AnyVector *notes_vector, int thread_id, char *scope);
extern "C" void *float_vec_Load(char *, int);



extern "C" float StoreFloatVecOnDemand(char *name, std::vector<float> value);

extern "C" float StoreFloatVecOnDemandOnIdx(char *name, float idx, float value);





extern "C" float PrintFloatVec(std::vector<float> vec);

extern "C" void * zeros_vec(float size);

extern "C" void * ones_vec(float size);
