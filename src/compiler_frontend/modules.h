#pragma once


#include "llvm/IR/Value.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/LLVMContext.h"

#include <memory>
#include <string>


#include "../KaleidoscopeJIT.h"
#include "logging.h"



using namespace llvm;
using namespace orc;

extern std::unique_ptr<KaleidoscopeJIT> TheJIT;
extern std::unique_ptr<LLVMContext> TheContext;
extern std::unique_ptr<LLVMContext> GlobalContext;

extern std::unique_ptr<IRBuilder<>> Builder;
extern std::unique_ptr<Module> TheModule;
extern std::unique_ptr<Module> GlobalModule;


inline Function *getFunctionCheck(std::string Name) {
  // First, see if the function has already been added to the current module.
  if (auto *F = TheModule->getFunction(Name))
    return F;

  LogError("The function " + Name + " was not found.");
  // If no existing prototype exists, return null.
  return nullptr;
}

inline void call(std::string fn, std::vector<Value *> args) {
    Builder->CreateCall(getFunctionCheck(fn), args);
}
inline Value *callret(std::string fn, std::vector<Value *> args) {
    return Builder->CreateCall(getFunctionCheck(fn), args);
}


inline Value *global_str(std::string _string) {
    return Builder->CreateGlobalString(_string);
}


inline Value *const_int(int val) {
    return ConstantInt::get(Type::getInt32Ty(*TheContext), val);
}
inline Value *const_float(float val) {
    return ConstantFP::get(*TheContext, APFloat(val));
}


inline AllocaInst *int_alloca() {
    return Builder->CreateAlloca(Type::getInt32Ty(*TheContext), nullptr);
}
inline AllocaInst *float_alloca() {
    return Builder->CreateAlloca(Type::getFloatTy(*TheContext), nullptr);
}

inline void store_alloca(AllocaInst *alloca, Value *val) {
    Builder->CreateStore(val, alloca);
}

inline Value *load_int(AllocaInst *alloca) {
    Builder->CreateLoad(Type::getInt32Ty(*TheContext), alloca, "loaded");
}
inline Value *load_float(AllocaInst *alloca) {
    Builder->CreateLoad(Type::getFloatTy(*TheContext), alloca, "loaded");
}