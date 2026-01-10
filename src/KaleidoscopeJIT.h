//===- KaleidoscopeJIT.h - A simple JIT for Kaleidoscope --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Contains a simple JIT definition for use in the kaleidoscope tutorials.
//
//===----------------------------------------------------------------------===//
#pragma once

#include "llvm/ADT/StringRef.h"
#include "llvm/ExecutionEngine/Orc/CompileOnDemandLayer.h"
#include "llvm/ExecutionEngine/Orc/CompileUtils.h"
#include "llvm/ExecutionEngine/Orc/Core.h"
#include "llvm/ExecutionEngine/Orc/EPCIndirectionUtils.h"
#include "llvm/ExecutionEngine/Orc/ExecutionUtils.h"
#include "llvm/ExecutionEngine/Orc/ExecutorProcessControl.h"
#include "llvm/ExecutionEngine/Orc/IRCompileLayer.h"
#include "llvm/ExecutionEngine/Orc/IRTransformLayer.h"
#include "llvm/ExecutionEngine/Orc/JITTargetMachineBuilder.h"
#include "llvm/ExecutionEngine/Orc/RTDyldObjectLinkingLayer.h"
#include "llvm/ExecutionEngine/Orc/Shared/ExecutorSymbolDef.h"
#include "llvm/ExecutionEngine/SectionMemoryManager.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/Transforms/InstCombine/InstCombine.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Scalar/GVN.h"
#include <memory>

#include "../lsp/json.hpp"

class PrototypeAST;
class ExprAST;

/// FunctionAST - This class represents a function definition itself.
class FunctionAST {
  std::unique_ptr<PrototypeAST> Proto;
  //std::vector<ExprAST> Body;
  std::vector<std::unique_ptr<ExprAST>> Body;

  public:
    FunctionAST(std::unique_ptr<PrototypeAST> Proto,
                std::vector<std::unique_ptr<ExprAST>> Body);
  
  const PrototypeAST& getProto() const;
  const std::string& getName() const;
  llvm::Function *codegen();
  nlohmann::json toJSON();
};




/// This will compile FnAST to IR, rename the function to add the given
/// suffix (needed to prevent a name-clash with the function's stub),
/// and then take ownership of the module that the function was compiled
/// into.
llvm::orc::ThreadSafeModule irgenAndTakeOwnership(FunctionAST &FnAST,
                                                  const std::string &Suffix);

namespace llvm {
namespace orc {

class KaleidoscopeASTLayer;
class KaleidoscopeJIT;


class KaleidoscopeASTMaterializationUnit : public MaterializationUnit {
  public:
    KaleidoscopeASTMaterializationUnit(KaleidoscopeASTLayer &L,
                                      std::unique_ptr<FunctionAST> F);

  StringRef getName() const override; 

  void materialize(std::unique_ptr<MaterializationResponsibility> R) override;

  private:
    void discard(const JITDylib &JD, const SymbolStringPtr &Sym) override; 

  KaleidoscopeASTLayer &L;
  std::unique_ptr<FunctionAST> F;
};


class KaleidoscopeASTLayer {
public:
  KaleidoscopeASTLayer(IRLayer &BaseLayer, const DataLayout &DL);

  Error add(ResourceTrackerSP RT, std::unique_ptr<FunctionAST> F); 

  void emit(std::unique_ptr<MaterializationResponsibility> MR,
            std::unique_ptr<FunctionAST> F); 
  

  MaterializationUnit::Interface getInterface(FunctionAST &F); 

  private:
    IRLayer &BaseLayer;
    const DataLayout &DL;
};






class KaleidoscopeJIT {
private:
  std::unique_ptr<ExecutionSession> ES;
  std::unique_ptr<EPCIndirectionUtils> EPCIU;

  DataLayout DL;

  RTDyldObjectLinkingLayer ObjectLayer;
  IRCompileLayer CompileLayer;
  IRTransformLayer OptimizeLayer;
  KaleidoscopeASTLayer ASTLayer;

  JITDylib &MainJD;

  static void handleLazyCallThroughError(); 

public:
  MangleAndInterner Mangle;
  KaleidoscopeJIT(std::unique_ptr<ExecutionSession> ES,
                  std::unique_ptr<EPCIndirectionUtils> EPCIU,
                  JITTargetMachineBuilder JTMB, DataLayout DL);

  ~KaleidoscopeJIT(); 

  static Expected<std::unique_ptr<KaleidoscopeJIT>> Create(); 

  const DataLayout &getDataLayout() const; 

  JITDylib &getMainJITDylib();

  Error addModule(ThreadSafeModule TSM, ResourceTrackerSP RT = nullptr);

  Error addAST(std::unique_ptr<FunctionAST> F, ResourceTrackerSP RT = nullptr); 

  Expected<ExecutorSymbolDef> lookup(StringRef Name); 

private:
  static Expected<ThreadSafeModule>
  optimizeModule(ThreadSafeModule TSM, const MaterializationResponsibility &R); 
};

} // end namespace orc
} // end namespace llvm
