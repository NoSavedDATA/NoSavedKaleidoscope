#pragma once


extern "C" void InstantiateObject(char *scope, char *obj_name);

extern "C" char *objHash(char *scope, char *obj_name);

extern "C" char *LoadObject(char *obj_name);
