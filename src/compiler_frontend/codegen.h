#pragma once

#include <string>

#include "llvm/IR/Value.h"



#include "../data_types/include.h"
#include "../notators/include.h"
#include "../tensor/include.h"
#include "../KaleidoscopeJIT.h"
#include "include.h"
#include "modules.h"
#include "expressions.h"



using namespace llvm;



extern std::vector<Value *> thread_pointers;



extern bool seen_var_attr;


Value * VoidPtr_toValue(void *vec);
Value* FloatPtr_toValue(float* vec);

Function *getFunction(std::string Name);


/// CreateEntryBlockAlloca - Create an alloca instruction in the entry block of
/// the function.  This is used for mutable variables etc.
AllocaInst *CreateEntryBlockAlloca(Function *TheFunction,
                                          StringRef VarName);



