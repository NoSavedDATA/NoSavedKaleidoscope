#pragma once



extern "C" float float_Create(char *name, char *scopeless_name, float init_val, AnyVector *notes_vector, int thread_id, char *scope);
extern "C" float float_Load(char *, int);

extern "C" void StoreOnDemandNoFree(char *name, float value);

extern "C" float LoadOnDemandNoFree(char *object_var_name);


extern "C" void *to_string(float v);

extern "C" void PrintFloat(float value);

extern "C" float UnbugFloat(float value);
