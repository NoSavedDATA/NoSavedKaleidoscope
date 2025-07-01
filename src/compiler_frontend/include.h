#pragma once


#include <map>
#include <string>
#include <vector>

#include "codegen.h"
#include "expressions.h"
#include "libs_parser.h"
#include "logging.h"
#include "modules.h"
#include "name_solver.h"
#include "parser.h"
#include "tokenizer.h"

#include "../threads/include.h"
#include "../KaleidoscopeJIT.h"


extern std::vector<std::string> return_tensor_functions, return_tensor_methods, return_tensor_fn, native_modules,
return_pinned_methods, vararg_methods, string_methods, native_methods, native_functions, native_fn, tensor_inits,
return_string_fn, threaded_tensor_functions, require_scope_functions, notators_str, user_cpp_functions;


extern std::map<std::string, std::string> functions_return_type, reverse_ops;



extern std::map<std::string, std::vector<std::string>> data_typeVars;
extern std::map<std::string, std::string> typeVars;

extern std::vector<std::string> objectVars;
extern std::vector<std::string> globalVars;
extern std::map<std::string, std::string> functionVars;
extern std::map<std::string, std::string> floatFunctions;
extern std::map<std::string, std::string> stringMethods;


extern std::vector<std::string> Classes;
extern std::map<std::string, std::string> Object_toClass;
extern std::map<std::string, std::string> Object_toClassVec;


extern int TERMINATE_VARARG;


extern std::map<std::string, std::unique_ptr<PrototypeAST>> FunctionProtos;
extern std::unique_ptr<llvm::orc::KaleidoscopeJIT> TheJIT;
extern ExitOnError ExitOnErr;

extern PointerType *floatPtrTy, *int8PtrTy;