#pragma once


#include "llvm/IR/Value.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/LLVMContext.h"

#include <memory>
#include <string>


#include "../KaleidoscopeJIT.h"



using namespace llvm;
using namespace orc;

extern std::unique_ptr<KaleidoscopeJIT> TheJIT;
extern std::unique_ptr<LLVMContext> TheContext;
extern std::unique_ptr<LLVMContext> GlobalContext;

extern std::unique_ptr<IRBuilder<>> Builder;
extern std::unique_ptr<Module> TheModule;
extern std::unique_ptr<Module> GlobalModule;