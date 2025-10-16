#pragma once

#include <string>
#include <vector>

#include "llvm/IR/Value.h"



#include "../data_types/include.h"
#include "../notators/include.h"
#include "../KaleidoscopeJIT.h"
#include "include.h"
#include "modules.h"
#include "expressions.h"



using namespace llvm;



extern std::vector<Value *> thread_pointers;
extern std::map<std::string, std::map<std::string, AllocaInst *>> function_allocas;
extern std::string current_codegen_function;



extern bool seen_var_attr;


Value * VoidPtr_toValue(void *vec);
Value* FloatPtr_toValue(float* vec);

Function *getFunction(std::string Name);


/// CreateEntryBlockAlloca - Create an alloca instruction in the entry block of
/// the function.  This is used for mutable variables etc.
AllocaInst *CreateEntryBlockAlloca(Function *TheFunction,
                                          StringRef VarName, llvm::Type *);





Value *load_alloca(std::string name, std::string type, std::string from_function);


Type *get_type_from_str(std::string type);

std::string Get_Nested_Name(std::vector<std::string>, Parser_Struct, bool);


bool Check_Is_Compatible_Data_Type(Data_Tree LType, Data_Tree RType, Parser_Struct parser_struct);


bool CheckIsEquivalent(std::string LType, std::string RType);