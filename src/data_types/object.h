#pragma once


extern "C" void InstantiateObject(char *scope, char *obj_name);

extern "C" char *objHash(char *scope, char *obj_name);

extern "C" char *LoadObject(char *obj_name);


extern "C" float InitObjectVecWithNull(char *name, float vec_size);

extern "C" float is_null(char *name);



extern "C" void objAttr_var_from_var(char *LName, char *RName);

extern "C" void objAttr_var_from_vec(char *LName, char *RName);

extern "C" void objAttr_vec_from_var(char *LName, char *RName);

extern "C" void objAttr_vec_from_vec(char *LName, char *RName);

extern "C" float append(char *self, char *obj_name);

extern "C" char *LoadObjectScopeName(char *self);
