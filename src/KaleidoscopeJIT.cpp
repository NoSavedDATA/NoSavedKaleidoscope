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
#include "KaleidoscopeJIT.h"
#include "compiler_frontend/expressions.h"

class PrototypeAST;
class ExprAST;

/// FunctionAST - This class represents a function definition itself.
FunctionAST::FunctionAST(std::unique_ptr<PrototypeAST> Proto,
                std::vector<std::unique_ptr<ExprAST>> Body)
        : Proto(std::move(Proto)), Body(std::move(Body)) {}
  



/// This will compile FnAST to IR, rename the function to add the given
/// suffix (needed to prevent a name-clash with the function's stub),
/// and then take ownership of the module that the function was compiled
/// into.

namespace llvm {
namespace orc {

class KaleidoscopeASTLayer;
class KaleidoscopeJIT;


StringRef KaleidoscopeASTMaterializationUnit::getName() const  {
  return "KaleidoscopeASTMaterializationUnit";
}


void KaleidoscopeASTMaterializationUnit::discard(const JITDylib &JD, const SymbolStringPtr &Sym)  {
  llvm_unreachable("Kaleidoscope functions are not overridable");
}


KaleidoscopeASTLayer::KaleidoscopeASTLayer(IRLayer &BaseLayer, const DataLayout &DL)
    : BaseLayer(BaseLayer), DL(DL) {}

Error KaleidoscopeASTLayer::add(ResourceTrackerSP RT, std::unique_ptr<FunctionAST> F) {
  return RT->getJITDylib().define(
      std::make_unique<KaleidoscopeASTMaterializationUnit>(*this,
                                                            std::move(F)),
      RT);
}

void KaleidoscopeASTLayer::emit(std::unique_ptr<MaterializationResponsibility> MR,
          std::unique_ptr<FunctionAST> F) {
  BaseLayer.emit(std::move(MR), irgenAndTakeOwnership(*F, ""));
}
  

MaterializationUnit::Interface KaleidoscopeASTLayer::getInterface(FunctionAST &F) {
  MangleAndInterner Mangle(BaseLayer.getExecutionSession(), DL);
  SymbolFlagsMap Symbols;
  Symbols[Mangle(F.getName())] =
      JITSymbolFlags(JITSymbolFlags::Exported | JITSymbolFlags::Callable);
  return MaterializationUnit::Interface(std::move(Symbols), nullptr);
}



KaleidoscopeASTMaterializationUnit::KaleidoscopeASTMaterializationUnit(
    KaleidoscopeASTLayer &L, std::unique_ptr<FunctionAST> F)
    : MaterializationUnit(L.getInterface(*F)), L(L), F(std::move(F)) {}


void KaleidoscopeASTMaterializationUnit::materialize(
    std::unique_ptr<MaterializationResponsibility> R)
    { L.emit(std::move(R), std::move(F)); }


void KaleidoscopeJIT::handleLazyCallThroughError() {
  errs() << "LazyCallThrough error: Could not find function body";
  exit(1);
}

KaleidoscopeJIT::KaleidoscopeJIT(std::unique_ptr<ExecutionSession> ES,
                std::unique_ptr<EPCIndirectionUtils> EPCIU,
                JITTargetMachineBuilder JTMB, DataLayout DL)
    : ES(std::move(ES)), EPCIU(std::move(EPCIU)), DL(std::move(DL)),
      Mangle(*this->ES, this->DL),
      ObjectLayer(*this->ES,
                  []() { return std::make_unique<SectionMemoryManager>(); }),
      CompileLayer(*this->ES, ObjectLayer,
                    std::make_unique<ConcurrentIRCompiler>(std::move(JTMB))),
      OptimizeLayer(*this->ES, CompileLayer, optimizeModule),
      ASTLayer(OptimizeLayer, this->DL),
      MainJD(this->ES->createBareJITDylib("<main>")) {
  MainJD.addGenerator(
      cantFail(DynamicLibrarySearchGenerator::GetForCurrentProcess(
          DL.getGlobalPrefix())));
}

KaleidoscopeJIT::~KaleidoscopeJIT() {
  if (auto Err = ES->endSession())
    ES->reportError(std::move(Err));
  if (auto Err = EPCIU->cleanup())
    ES->reportError(std::move(Err));
}

Expected<std::unique_ptr<KaleidoscopeJIT>> KaleidoscopeJIT::Create() {
  auto EPC = SelfExecutorProcessControl::Create();
  if (!EPC)
    return EPC.takeError();

  auto ES = std::make_unique<ExecutionSession>(std::move(*EPC));

  auto EPCIU = EPCIndirectionUtils::Create(*ES);
  if (!EPCIU)
    return EPCIU.takeError();

  (*EPCIU)->createLazyCallThroughManager(
      *ES, ExecutorAddr::fromPtr(&handleLazyCallThroughError));

  if (auto Err = setUpInProcessLCTMReentryViaEPCIU(**EPCIU))
    return std::move(Err);

  JITTargetMachineBuilder JTMB(
      ES->getExecutorProcessControl().getTargetTriple());

  JTMB.setCodeGenOptLevel(llvm::CodeGenOptLevel::Aggressive);

  auto DL = JTMB.getDefaultDataLayoutForTarget();
  if (!DL)
    return DL.takeError();

  return std::make_unique<KaleidoscopeJIT>(std::move(ES), std::move(*EPCIU),
                                            std::move(JTMB), std::move(*DL));
}

const DataLayout &KaleidoscopeJIT::getDataLayout() const { return DL; }

JITDylib &KaleidoscopeJIT::getMainJITDylib() { return MainJD; }

Error KaleidoscopeJIT::addModule(ThreadSafeModule TSM, ResourceTrackerSP RT) {
  if (!RT)
    RT = MainJD.getDefaultResourceTracker();

  return OptimizeLayer.add(RT, std::move(TSM));
}

Error KaleidoscopeJIT::addAST(std::unique_ptr<FunctionAST> F, ResourceTrackerSP RT) {
  if (!RT)
    RT = MainJD.getDefaultResourceTracker();
  return ASTLayer.add(RT, std::move(F));
}

Expected<ExecutorSymbolDef> KaleidoscopeJIT::lookup(StringRef Name) {
  return ES->lookup({&MainJD}, Mangle(Name.str()));
}

Expected<ThreadSafeModule>
KaleidoscopeJIT::optimizeModule(ThreadSafeModule TSM, const MaterializationResponsibility &R) {
  TSM.withModuleDo([](Module &M) {
    // Create a function pass manager.
    auto FPM = std::make_unique<legacy::FunctionPassManager>(&M);

    // Add some optimizations.
    FPM->add(createInstructionCombiningPass());
    FPM->add(createReassociatePass());
    FPM->add(createGVNPass());
    FPM->add(createCFGSimplificationPass());
    FPM->doInitialization();

    // Run the optimizations over all functions in the module being added to
    // the JIT.
    for (auto &F : M)
      FPM->run(F);
  });

  return std::move(TSM);
}


} // end namespace orc
} // end namespace llvm
