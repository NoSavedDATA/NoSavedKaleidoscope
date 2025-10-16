#pragma once


#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Value.h"
#include "llvm/IR/Verifier.h"

#include <memory>
#include <string>


#include "../KaleidoscopeJIT.h"
#include "global_vars.h"
#include "logging.h"



using namespace llvm;
using namespace orc;

extern std::unique_ptr<KaleidoscopeJIT> TheJIT;
extern std::unique_ptr<LLVMContext> TheContext;
extern std::unique_ptr<LLVMContext> GlobalContext;

extern std::unique_ptr<IRBuilder<>> Builder;
extern std::unique_ptr<Module> TheModule;
extern std::unique_ptr<Module> GlobalModule;


extern std::map<std::string, StructType*> struct_types;

extern std::map<std::string, std::unique_ptr<PrototypeAST>> FunctionProtos;

inline Value *const_int(int val) {
    return ConstantInt::get(Type::getInt32Ty(*TheContext), val);
}
inline Value *const_int64(int val) {
    return ConstantInt::get(Type::getInt64Ty(*TheContext), val);
}
inline Value *const_float(float val) {
    return ConstantFP::get(*TheContext, APFloat(val));
}
inline Value *const_bool(bool val) {
    return ConstantInt::get(Type::getInt1Ty(*TheContext), val);
}



inline Function *getFunctionCheck(std::string Name) {
  // First, see if the function has already been added to the current module.


  if (auto *F = TheModule->getFunction(Name))
    return F;

  auto FI = FunctionProtos.find(Name);
  if (FI != FunctionProtos.end())
    return FI->second->codegen();


    

  LogError(-1, "The function " + Name + " was not found.");
  // If no existing prototype exists, return null.
  return nullptr;
}

inline void call(std::string fn, const std::vector<Value *> &args) {
    if(!Shall_Exit)
        Builder->CreateCall(getFunctionCheck(fn), args);
}
inline Value *callret(std::string fn, const std::vector<Value *> &args) { 
    if(!Shall_Exit)
        return Builder->CreateCall(getFunctionCheck(fn), args);
    return const_float(0);
}



inline Value *robust_str(std::string _string) {
    Constant *fnStrConstant = ConstantDataArray::getString(*TheContext, _string, true);
    GlobalVariable *fnStrGV = new GlobalVariable(
        *TheModule,
        fnStrConstant->getType(),
        true, // isConstant
        GlobalValue::PrivateLinkage,
        fnStrConstant,
        ".str");

    fnStrGV->setUnnamedAddr(GlobalValue::UnnamedAddr::Global);
    fnStrGV->setAlignment(Align(1));

    // get a pointer to the first element (i8*)
    Value *v = Builder->CreateInBoundsGEP(
        fnStrConstant->getType(),
        fnStrGV,
        {Builder->getInt32(0), Builder->getInt32(0)},
        "fn_name_gep");
    return v;
}

inline Value *global_str(std::string _string) {
    return Builder->CreateGlobalString(_string);
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



inline void check_llvm_err(){
    std::string err;
    llvm::raw_string_ostream os(err);
    if (llvm::verifyModule(*TheModule, &os)) {
        errs() << "Module broken:\n" << os.str();
        TheModule->print(llvm::errs(), nullptr);
        // abort();
    }
}
