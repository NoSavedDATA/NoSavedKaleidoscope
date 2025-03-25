#pragma once

#include "tokenizer.h"
#include "expressions.h"
#include "logging.h"
#include "parser.h"

#include "../KaleidoscopeJIT.h"


extern std::vector<std::string> return_tensor_functions, return_tensor_methods, return_tensor_fn, native_modules,
return_pinned_methods, vararg_methods, string_methods, native_methods, native_functions, native_fn, tensor_inits,
return_string_fn, threaded_tensor_functions, require_scope_functions, notators_str;


extern std::vector<std::string> tensorVars;
extern std::vector<std::string> pinnedTensorVars;
extern std::vector<std::string> floatVars;
extern std::vector<std::string> strVars;
extern std::vector<std::string> objectVars;
extern std::vector<std::string> globalVars;
extern std::map<std::string, std::string> functionVars;
extern std::map<std::string, std::string> floatFunctions;
extern std::map<std::string, std::string> stringMethods;
extern std::vector<std::string> str_vecVars;
extern std::vector<std::string> float_vecVars;
extern std::map<std::string, pthread_mutex_t *> lockVars;



extern std::vector<std::string> Classes;
extern std::map<std::string, std::string> Object_toClass;
extern std::map<std::string, std::string> Object_toClassVec;


extern float TERMINATE_VARARG;


extern std::map<std::string, std::unique_ptr<PrototypeAST>> FunctionProtos;
extern std::unique_ptr<llvm::orc::KaleidoscopeJIT> TheJIT;
extern ExitOnError ExitOnErr;