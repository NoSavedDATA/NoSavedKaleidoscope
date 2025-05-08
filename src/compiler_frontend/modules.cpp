#pragma once

#include <string>

#include "llvm/IR/Value.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/LLVMContext.h"

#include <memory>


#include "../KaleidoscopeJIT.h"

using namespace llvm;
using namespace orc;

std::unique_ptr<KaleidoscopeJIT> TheJIT;
std::unique_ptr<LLVMContext> TheContext;
std::unique_ptr<LLVMContext> GlobalContext = std::make_unique<LLVMContext>();


std::unique_ptr<IRBuilder<>> Builder;
std::unique_ptr<Module> TheModule;
std::unique_ptr<Module> GlobalModule;


