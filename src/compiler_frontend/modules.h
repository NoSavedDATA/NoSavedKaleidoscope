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



inline void call(std::string fn, std::vector<Value *> args) {
    Builder->CreateCall(TheModule->getFunction(fn), args);
}
inline Value *callret(std::string fn, std::vector<Value *> args) {
    return Builder->CreateCall(TheModule->getFunction(fn), args);
}


inline Value *global_str(std::string _string) {
    return Builder->CreateGlobalString(_string);
}