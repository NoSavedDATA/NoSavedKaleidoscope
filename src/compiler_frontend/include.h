#pragma once


#include <map>
#include <string>
#include <vector>

#include "codegen.h"
#include "expressions.h"
#include "function_ast.h"
#include "global_vars.h"
#include "libs_parser.h"
#include "logging.h"
#include "logging_execution.h"
#include "modules.h"
#include "name_solver.h"
#include "parser.h"
#include "tokenizer.h"

#include "../threads/include.h"
#include "../KaleidoscopeJIT.h"


extern std::vector<std::string> return_tensor_functions, return_tensor_methods, return_tensor_fn, native_modules,
return_pinned_methods, vararg_methods, string_methods, native_methods, native_functions, native_fn, tensor_inits,
return_string_fn, threaded_tensor_functions, require_scope_functions, notators_str, user_cpp_functions;


extern std::map<std::string, int> Function_Arg_Count;
extern std::map<std::string, int> Function_Required_Arg_Count;
extern std::map<std::string, std::map<std::string, std::string>> Function_Arg_Types;
extern std::map<std::string, std::map<std::string, Data_Tree>> Function_Arg_DataTypes;
extern std::map<std::string, std::vector<std::string>> Function_Arg_Names;
extern std::map<std::string, std::map<std::string, std::unique_ptr<ExprAST>>> ArgsInit;


extern std::map<std::string, Data_Tree> Idx_Fn_Return;


extern std::vector<std::string> Sys_Arguments;
 
extern std::map<std::string, std::vector<std::string>> Equivalent_Types;


extern std::vector<std::string> imported_libs;
extern std::map<std::string, std::vector<std::string>> lib_submodules;

extern std::map<std::string, std::string> lib_function_remaps;





extern std::map<std::string, std::map<std::string, Data_Tree>> data_typeVars;
extern std::map<std::string, std::map<std::string, std::string>> typeVars;

extern std::map<std::string, std::string> floatFunctions;
extern std::map<std::string, std::string> stringMethods;


extern std::vector<std::string> Classes;




extern std::unique_ptr<llvm::orc::KaleidoscopeJIT> TheJIT;
extern ExitOnError ExitOnErr;

extern PointerType *floatPtrTy, *int8PtrTy;
extern llvm::Type *floatTy, *intTy, *boolTy;

extern Value *stack, *stack_top_value;
