#pragma once

#include <string>

#include "../tensor/tensor_struct.h"
#include"include.h"



extern std::vector<std::string> scopes;


extern "C" char * ConcatScopeStr(char *lc, char *rc);


extern "C" char * ConcatScopeAtCallExpr(char *lc, char *rc);


extern "C" void AddFloatToScopeCleanList(char *scope, char *name);


extern "C" void AddToScopeCleanList(char *scope, char *name);


extern "C" void CleanScopeVars(char *scope, int thread_id);


extern "C" float RemoveTensorScope(char *tensor_name, char *scope, char *tgt_tensorc, char *previous_scope, int thread_id);


extern "C" float print_scope(char *scope, char *previous_scope, int thread_id);

