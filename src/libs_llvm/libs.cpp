#include "llvm/IR/Value.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/LLVMContext.h"

#include <iostream>
#include <memory>

using namespace llvm;

#include "../compiler_frontend/modules.h"

void Generate_LLVM_Functions() {
    PointerType *floatPtrTy, *int8PtrTy;

    floatPtrTy = Type::getFloatTy(*TheContext)->getPointerTo();
    int8PtrTy = Type::getInt8Ty(*TheContext)->getPointerTo();
    
}