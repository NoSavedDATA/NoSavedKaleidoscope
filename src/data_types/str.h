#pragma once


extern "C" float str_Create(char *name, char *scopeless_name, char *init_val, AnyVector *notes_vector, int thread_id, char *scope);

extern "C" float StoreStrOnDemand(char *name, char *value);

extern "C" void *LoadStrOnDemand(char *name);

extern "C" float PrintStr(char* value);


extern "C" float *split_str_to_float(char *in_string, int gather_position);



extern "C" void *cat_str_float(char *c, float v);
