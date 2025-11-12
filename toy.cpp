// JIT
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetMachine.h"

#include "src/KaleidoscopeJIT.h"




#include <algorithm>
#include <cstdarg>
#include <cassert>
#include <cctype>
#include <cstring>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <memory>
#include <string>
#include <iostream>
#include <numeric>
#include <utility>
#include <vector>
#include <iomanip>
#include <math.h>
#include <fenv.h>
#include <tuple>
#include <chrono>
#include <thread>
#include <random>
#include <float.h>
#include <fstream>
#include <sstream>
#include <filesystem>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>






#include "src/include.h"






using namespace llvm;
using namespace llvm::orc;








std::map<std::string, int> NotatorsMap = {
  {"bias", bias},
  {"fp32", fp32},
  {"fp16", fp16},
  {"causal", causal},
};

bool ShallCodegen = true;



LCG rng(generate_custom_seed());






  // Error Colors
// \033[0m default
// \033[31m red
// \033[33m yellow
// \033[34m blue
// \033[95m purple






// Tensor related
std::vector<std::string> return_tensor_functions, return_tensor_methods, return_tensor_fn, native_modules,
return_pinned_methods, vararg_methods, string_methods, native_methods, native_functions, native_fn,
return_string_fn, threaded_tensor_functions, require_scope_functions, notators_str;


std::map<std::string, std::string> reverse_ops;





std::vector<std::string> Sys_Arguments;



//global
std::map<std::string, std::string> floatFunctions;



//global
std::vector<std::string> Classes;

std::map<size_t, std::vector<char *>> CharPool;









//===----------------------------------------------------------------------===//
// Code Generation
//===----------------------------------------------------------------------===//

//global


std::map<std::string, std::unique_ptr<PrototypeAST>> FunctionProtos;
ExitOnError ExitOnErr;


// Vars
std::map<std::string, Value *> NamedValues;
std::map<std::string, char *> NamedStrs;
std::map<std::string, std::vector<char *>> ClassStrVecs;
std::map<std::string, DT_int_vec *> NamedIntVecs;
std::map<std::string, float> NamedClassValues;
std::map<std::string, int> NamedInts;
std::map<std::string, std::string> NamedObjects;
std::map<std::string, std::vector<std::pair<std::string, std::string>>> ScopeVarsToClean;
std::map<std::string, char *> ScopeNamesToClean;
std::map<int, std::map<std::string, std::vector<std::string>>> ThreadedScopeTensorsToClean;







// File Handling
std::vector<char *> glob_str_files;




// Handle Class self with phantom argument



































const PrototypeAST& FunctionAST::getProto() const {
  return *Proto;
}

const std::string& FunctionAST::getName() const {
  return Proto->getName();
}

Function *FunctionAST::codegen() {
  if (not ShallCodegen)
    return nullptr;
  
  // Transfer ownership of the prototype to the FunctionProtos map, but keep a
  // reference to it for use below.
  auto &P = *Proto;

    

  FunctionProtos[Proto->getName()] = std::move(Proto);
  std::string function_name = P.getName();

  Function *TheFunction = getFunction(function_name);
  if (!TheFunction)
    return nullptr;

  // If this is an operator, install it.
  if (P.isBinaryOp())
    BinopPrecedence[P.getOperatorName()] = P.getBinaryPrecedence();

  // Create a new basic block to start insertion into.
  BasicBlock *BB = BasicBlock::Create(*TheContext, "entry", TheFunction);
  Builder->SetInsertPoint(BB);

  
  // Record the function arguments in the NamedValues map.
  Value *scope_struct;
  

  

  
  scope_struct = callret("scope_struct_Create", {});
  

  current_codegen_function = function_name;




//   std::cout << "\033[32mExecuting function: " << function_name << " \033[0m\n";

  NamedValues.clear();

  LogBlue(function_name);
  
  if(function_name=="__anon_expr")
    call("scope_struct_Alloc_GC", {scope_struct});
  

  



  float val;
  int i = 0;
  for (auto &Arg : TheFunction->args()) {
    // Create an alloca for this variable.
    
    
    std::string arg_name = Arg.getName().str();
    //std::cout << "FUNCTION ARG IS: " << arg_name  << "\n";



    // Default args
    if (arg_name == "scope_struct")
    {
        scope_struct = callret("scope_struct_Overwrite", {scope_struct, &Arg});
    } else { 
        std::string type = "";

        if (data_typeVars[function_name].find(arg_name) != data_typeVars[function_name].end())
            type = UnmangleVec(data_typeVars[function_name][arg_name]);
        else
            LogError(-1, "error at argument " + arg_name + " of function " + function_name);

            
        llvm::Type *alloca_type = get_type_from_str(type);
        AllocaInst *arg_alloca = CreateEntryBlockAlloca(TheFunction, arg_name, alloca_type);
        // std::string copy_fn = type+"_CopyArg";
        // Function *F = TheModule->getFunction(copy_fn);
        // if (F)
        // {
        //     Value *copied_value = callret(copy_fn,
        //                     {scope_struct,
        //                     &Arg,
        //                     global_str("-")});
        //     Builder->CreateStore(copied_value, arg_alloca);
        // } else
        Builder->CreateStore(&Arg, arg_alloca);

        function_allocas[current_codegen_function][arg_name] = arg_alloca;
    }
  }


  
  bool expr_is_return = false;
  Value *RetVal;
  for (auto &body : Body)
  {
    expr_is_return = ends_with(typeid(*body).name(), "RetExprAST");
    RetVal = body->codegen(scope_struct);
  }

  if (RetVal) {
    // Finish off the function.
    if(!expr_is_return)
    {
        // call("scope_struct_Clear_GC_Root", {scope_struct});
        // for (const auto &pair : function_allocas[current_codegen_function]) {
        //     std::string type = typeVars[current_codegen_function][pair.first];
        //     if (!in_str(type, primary_data_tokens)) {
        //         Value *loaded_var = load_alloca(pair.first, type, current_codegen_function);
        //         call("scope_struct_Add_GC_Root", {scope_struct, loaded_var, global_str(type)});
        //     }
        // }
        call("scope_struct_Clean_Scope", {scope_struct}); 
        Builder->CreateRet(RetVal); 
    }

    // Validate the generated code, checking for consistency.
    verifyFunction(*TheFunction);

    // TheModule->print(llvm::errs(), nullptr);
    // Validate the generated code, checking for consistency.

    return TheFunction;
  }


  // Error reading body, remove function.
  TheFunction->eraseFromParent();

  if (P.isBinaryOp())
    BinopPrecedence.erase(P.getOperatorName());
  return nullptr;
}





//===----------------------------------------------------------------------===//
// Top-Level parsing and JIT Driver
//===----------------------------------------------------------------------===//



static void InitializeModule() {
  //std::cout << "\nINITIALIZING A NEW MODULE"  << "\n\n";

  // Open a new context and module.

  TheContext = std::make_unique<LLVMContext>();
  TheModule = std::make_unique<Module>("my cool jit", *TheContext);
  TheModule->setDataLayout(TheJIT->getDataLayout());

  //std::cout << "Initialize Module\n";
  

  // Create a new builder for the module.
  Builder = std::make_unique<IRBuilder<>>(*TheContext);

  floatPtrTy = Type::getFloatTy(*TheContext)->getPointerTo();
  int8PtrTy = Type::getInt8Ty(*TheContext)->getPointerTo();
  ShallCodegen = true;
  seen_var_attr = false;


  Generate_LLVM_Functions();
  Generate_Lib_Functions();
  Generate_Struct_Types();


  //===----------------------------------------------------------------------===//
  // Scalar   Operations
  //===----------------------------------------------------------------------===//

  // 
  FunctionType *fmaxTy = FunctionType::get( //TODO: automatic type detection for max and min
      Type::getFloatTy(*TheContext),
      {Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext)},
      false
  );
  TheModule->getOrInsertFunction("max", fmaxTy);


  FunctionType *mallocType = FunctionType::get(
        int8PtrTy, 
        {Type::getInt64Ty(*TheContext)}, 
        false
  );
  TheModule->getOrInsertFunction("malloc", mallocType);

  // 
  FunctionType *fminTy = FunctionType::get(
      Type::getFloatTy(*TheContext),
      {Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext)},
      false
  );
  TheModule->getOrInsertFunction("min", fminTy);


  // 
  FunctionType *flog2Ty = FunctionType::get(
      Type::getFloatTy(*TheContext),
      {Type::getFloatTy(*TheContext)},
      false
  );
  TheModule->getOrInsertFunction("logE2f", flog2Ty);


  // 
  FunctionType *roundTy = FunctionType::get(
      Type::getFloatTy(*TheContext),
      {Type::getFloatTy(*TheContext)},
      false
  );
  TheModule->getOrInsertFunction("roundE", roundTy);


  // 
  FunctionType *logical_notTy = FunctionType::get(
      Type::getFloatTy(*TheContext),
      {Type::getFloatTy(*TheContext)},
      false
  );
  TheModule->getOrInsertFunction("logical_not", logical_notTy);


  // 
  FunctionType *dir_existsTy = FunctionType::get(
      Type::getFloatTy(*TheContext),
      {int8PtrTy},
      false
  );
  TheModule->getOrInsertFunction("dir_exists", dir_existsTy);


  // 
  FunctionType *path_existsTy = FunctionType::get(
      Type::getFloatTy(*TheContext),
      {int8PtrTy},
      false
  );
  TheModule->getOrInsertFunction("path_exists", path_existsTy);


  // 
  FunctionType *floorTy = FunctionType::get(
      Type::getFloatTy(*TheContext),
      {Type::getFloatTy(*TheContext)},
      false
  );
  TheModule->getOrInsertFunction("floorE", floorTy);

  //
  FunctionType *str_DeleteTy = FunctionType::get(
      int8PtrTy,
      {int8PtrTy},
      false
  );
  TheModule->getOrInsertFunction("str_Delete", str_DeleteTy);



  //  
  FunctionType *sleepTy = FunctionType::get(
      Type::getVoidTy(*TheContext),
      {int8PtrTy, Type::getFloatTy(*TheContext)},
      false
  );
  TheModule->getOrInsertFunction("__slee_p_", sleepTy);

  
  FunctionType *silent_sleepTy = FunctionType::get(
      Type::getFloatTy(*TheContext),
      {int8PtrTy, Type::getFloatTy(*TheContext)},
      false
  );
  TheModule->getOrInsertFunction("silent_sleep", silent_sleepTy);



	FunctionType *pthread_create_auxTy= FunctionType::get(
		int8PtrTy,
		{int8PtrTy, int8PtrTy, int8PtrTy, int8PtrTy},
		false
	);
	TheModule->getOrInsertFunction("pthread_create_aux", pthread_create_auxTy);

	FunctionType *pthread_join_auxTy= FunctionType::get(
		int8PtrTy,
		{int8PtrTy},
		false
	);
	TheModule->getOrInsertFunction("pthread_join_aux", pthread_join_auxTy);









  

  FunctionType *LockTy = FunctionType::get(
    Type::getVoidTy(*TheContext),
    {int8PtrTy},
    false);
  TheModule->getOrInsertFunction("LockMutex", LockTy);

  FunctionType *UnlockMutexTy = FunctionType::get(
    Type::getVoidTy(*TheContext),
    {int8PtrTy},
    false);
  TheModule->getOrInsertFunction("UnlockMutex", UnlockMutexTy);


  //===----------------------------------------------------------------------===//
  // Str Ops
  //===----------------------------------------------------------------------===//




  //
  FunctionType *zeros_vecTy = FunctionType::get(
      int8PtrTy,
      {int8PtrTy, Type::getFloatTy(*TheContext)},
      false
  );
  TheModule->getOrInsertFunction("zeros_vec", zeros_vecTy);


  //
  FunctionType *to_stringTy = FunctionType::get(
      int8PtrTy,
      {Type::getFloatTy(*TheContext)},
      false
  );
  TheModule->getOrInsertFunction("to_string", to_stringTy);


  //
  FunctionType *cat_str_floatTy = FunctionType::get(
      Type::getFloatTy(*TheContext),
      {int8PtrTy},
      false
  );
  TheModule->getOrInsertFunction("cat_str_float", cat_str_floatTy);


  //
  FunctionType *ones_vecTy = FunctionType::get(
      int8PtrTy,
      {int8PtrTy, Type::getFloatTy(*TheContext)},
      false
  );
  TheModule->getOrInsertFunction("ones_vec", ones_vecTy);
  

  FunctionType *PrintFloatTy = FunctionType::get(
      Type::getVoidTy(*TheContext),
      {Type::getFloatTy(*TheContext)},
      false 
  );
  TheModule->getOrInsertFunction("PrintFloat", PrintFloatTy);

  FunctionType *UnbugFloatTy = FunctionType::get(
      Type::getFloatTy(*TheContext),
      {Type::getFloatTy(*TheContext)},
      false 
  );
  TheModule->getOrInsertFunction("UnbugFloat", UnbugFloatTy);

 
  
  FunctionType *SplitStringTy = FunctionType::get(
      int8PtrTy,
      {int8PtrTy, int8PtrTy, int8PtrTy},
      false 
  );
  TheModule->getOrInsertFunction("SplitString", SplitStringTy);


  //
  FunctionType *SplitStringIndexateTy = FunctionType::get(
      int8PtrTy,
      {int8PtrTy, int8PtrTy, int8PtrTy, Type::getFloatTy(*TheContext)},
      false 
  );
  TheModule->getOrInsertFunction("str_split_idx", SplitStringIndexateTy);

  
  //
  FunctionType *str_to_floatTy = FunctionType::get(
      Type::getFloatTy(*TheContext),
      {int8PtrTy, int8PtrTy},
      false 
  );
  TheModule->getOrInsertFunction("str_to_float", str_to_floatTy);


  //
  FunctionType *CopyStringTy = FunctionType::get(
      int8PtrTy,
      {int8PtrTy},
      false 
  );
  TheModule->getOrInsertFunction("CopyString", CopyStringTy);


  //
  FunctionType *appendTy = FunctionType::get(
      Type::getFloatTy(*TheContext),
      {int8PtrTy, int8PtrTy},
      false 
  );
  TheModule->getOrInsertFunction("append", appendTy);


  //
  FunctionType *objAttr_var_from_varTy = FunctionType::get(
      Type::getVoidTy(*TheContext),
      {int8PtrTy, int8PtrTy},
      false 
  );
  TheModule->getOrInsertFunction("objAttr_var_from_var", objAttr_var_from_varTy);


  //
  FunctionType *objAttr_var_from_vecTy = FunctionType::get(
      Type::getVoidTy(*TheContext),
      {int8PtrTy, int8PtrTy},
      false 
  );
  TheModule->getOrInsertFunction("objAttr_var_from_vec", objAttr_var_from_vecTy);


  //
  FunctionType *objAttr_vec_from_varTy = FunctionType::get(
      Type::getVoidTy(*TheContext),
      {int8PtrTy, int8PtrTy},
      false 
  );
  TheModule->getOrInsertFunction("objAttr_vec_from_var", objAttr_vec_from_varTy);


  //
  FunctionType *objAttr_vec_from_vecTy = FunctionType::get(
      Type::getVoidTy(*TheContext),
      {int8PtrTy, int8PtrTy},
      false 
  );
  TheModule->getOrInsertFunction("objAttr_vec_from_vec", objAttr_vec_from_vecTy);





  //
  FunctionType *PrintStrTy = FunctionType::get(
      Type::getFloatTy(*TheContext),
      {int8PtrTy}, 
      false 
  );
  TheModule->getOrInsertFunction("PrintStr", PrintStrTy);


  //
  FunctionType *PrintStrVecTy = FunctionType::get(
      Type::getFloatTy(*TheContext),
      {int8PtrTy}, 
      false 
  );
  TheModule->getOrInsertFunction("PrintStrVec", PrintStrVecTy);


  //
  FunctionType *PrintFloatVecTy = FunctionType::get(
      Type::getFloatTy(*TheContext),
      {int8PtrTy}, 
      false 
  );
  TheModule->getOrInsertFunction("PrintFloatVec", PrintFloatVecTy);



  FunctionType *str_vec_printTy = FunctionType::get(
      Type::getFloatTy(*TheContext),
      {int8PtrTy, int8PtrTy}, 
      false 
  );
  TheModule->getOrInsertFunction("str_vec_print", str_vec_printTy);


  //
  FunctionType *LoadStrVecOnDemandTy = FunctionType::get(
      int8PtrTy,
      {int8PtrTy},
      false
  );
  TheModule->getOrInsertFunction("LoadStrVecOnDemand", LoadStrVecOnDemandTy);


  
  


  //
  FunctionType *LenStrVecTy = FunctionType::get(
      Type::getFloatTy(*TheContext),
      {int8PtrTy, int8PtrTy},
      false 
  );
  TheModule->getOrInsertFunction("LenStrVec", LenStrVecTy);


  //
  FunctionType *ShuffleStrVecTy = FunctionType::get(
      int8PtrTy,
      {int8PtrTy, int8PtrTy},
      false 
  );
  TheModule->getOrInsertFunction("ShuffleStrVec", ShuffleStrVecTy);


  //
  FunctionType *IndexStrVecTy = FunctionType::get(
      int8PtrTy,
      {int8PtrTy, Type::getFloatTy(*TheContext)}, 
      false 
  );
  TheModule->getOrInsertFunction("IndexStrVec", IndexStrVecTy);


  //
  FunctionType *IndexClassStrVecTy = FunctionType::get(
      int8PtrTy,
      {int8PtrTy, int8PtrTy, Type::getFloatTy(*TheContext)}, 
      false 
  );
  TheModule->getOrInsertFunction("str_vec_Idx", IndexClassStrVecTy);

  


  FunctionType *nullptr_getTy = FunctionType::get(
      int8PtrTy,
      {}, 
      false 
  );
  TheModule->getOrInsertFunction("nullptr_get", nullptr_getTy);


  // char *
  FunctionType *shuffle_strTy = FunctionType::get(
      int8PtrTy,
      {int8PtrTy},
      false 
  );
  TheModule->getOrInsertFunction("shuffle_str", shuffle_strTy);

  //===----------------------------------------------------------------------===//
  // Other Ops
  //===----------------------------------------------------------------------===//


  // 
  FunctionType *FirstArgOnDemandTy = FunctionType::get(
      int8PtrTy,
      {int8PtrTy, int8PtrTy, int8PtrTy, int8PtrTy, Type::getInt32Ty(*TheContext), Type::getInt32Ty(*TheContext), Type::getInt32Ty(*TheContext)},
      false 
  );
  TheModule->getOrInsertFunction("FirstArgOnDemand", FirstArgOnDemandTy);
  



  

  //
  FunctionType * GetEmptyCharTy = FunctionType::get(
      int8PtrTy,
      {},
      false 
  );
  TheModule->getOrInsertFunction("GetEmptyChar", GetEmptyCharTy);
  

  //
  FunctionType *FreeCharTy = FunctionType::get(
      Type::getVoidTy(*TheContext),
      {int8PtrTy},
      false
  );
  TheModule->getOrInsertFunction("FreeChar", FreeCharTy);
  

  //
  FunctionType *FreeCharFromFuncTy = FunctionType::get(
      Type::getVoidTy(*TheContext),
      {int8PtrTy},
      false
  );
  TheModule->getOrInsertFunction("FreeCharFromFunc", FreeCharFromFuncTy);
  

  //
  FunctionType * ConcatStrTy = FunctionType::get(
      int8PtrTy,
      {int8PtrTy, int8PtrTy},
      false 
  );
  TheModule->getOrInsertFunction("ConcatStr", ConcatStrTy);
  

  //
  FunctionType * ConcatStrFreeLeftTy = FunctionType::get(
      int8PtrTy,
      {int8PtrTy, int8PtrTy},
      false 
  );
  TheModule->getOrInsertFunction("ConcatStrFreeLeft", ConcatStrFreeLeftTy);
  

  //
  FunctionType * ConcatStrFreeRightTy = FunctionType::get(
      int8PtrTy,
      {int8PtrTy, int8PtrTy},
      false 
  );
  TheModule->getOrInsertFunction("ConcatStrFreeRight", ConcatStrFreeRightTy);
  

  //
  FunctionType * ConcatStrFreeTy = FunctionType::get(
      int8PtrTy,
      {int8PtrTy, int8PtrTy},
      false 
  );
  TheModule->getOrInsertFunction("ConcatStrFree", ConcatStrFreeTy);

  
  //
  FunctionType * ConcatNumToStrTy = FunctionType::get(
      int8PtrTy,
      {int8PtrTy, Type::getInt32Ty(*TheContext)},
      false 
  );
  TheModule->getOrInsertFunction("ConcatNumToStr", ConcatNumToStrTy);

  
  //
  FunctionType * ConcatNumToStrFreeTy = FunctionType::get(
      int8PtrTy,
      {int8PtrTy, Type::getInt32Ty(*TheContext)},
      false 
  );
  TheModule->getOrInsertFunction("ConcatNumToStrFree", ConcatNumToStrFreeTy);
  

  //
  FunctionType * ConcatScopeStrTy = FunctionType::get(
      int8PtrTy,
      {int8PtrTy, int8PtrTy},
      false 
  );
  TheModule->getOrInsertFunction("ConcatScopeStr", ConcatScopeStrTy);
  
    

  //
  FunctionType * RandomStrOnDemandTy = FunctionType::get(
      int8PtrTy,
      {},
      false
  );
  TheModule->getOrInsertFunction("RandomStrOnDemand", RandomStrOnDemandTy);  

  

  
  
  FunctionType *print_codegenTy = FunctionType::get(
      int8PtrTy,
      {int8PtrTy},
      false 
  );
  TheModule->getOrInsertFunction("print_codegen", print_codegenTy);


  TheModule->getOrInsertFunction(
    "posix_memalign",
    FunctionType::get(Type::getInt32Ty(*TheContext),
                      {int8PtrTy->getPointerTo(), // void**
                       Type::getInt64Ty(*TheContext),                   // alignment
                       Type::getInt64Ty(*TheContext)},                  // size
                      false));

  //
  FunctionType *printTy = FunctionType::get(
      Type::getFloatTy(*TheContext),
      {int8PtrTy, int8PtrTy}, 
      false 
  );
  TheModule->getOrInsertFunction("print", printTy);
  

}




ThreadSafeModule irgenAndTakeOwnership(FunctionAST &FnAST,
                                       const std::string &Suffix) {
  if (auto *F = FnAST.codegen()) {
    F->setName(F->getName() + Suffix);
    auto TSM = ThreadSafeModule(std::move(TheModule), std::move(TheContext));
    // Start a new module.
    InitializeModule();
    return TSM;
  } else
    report_fatal_error("failed to JIT.");
}




  

static void HandleImport() {
    Parser_Struct parser_struct;
    parser_struct.line = LineCounter;
    ParseImport(parser_struct);
}

static void HandleClass() {
    Parser_Struct parser_struct;
    parser_struct.line = LineCounter;
    ParseClass(parser_struct);
}

static void HandleDefinition() {
  
  Parser_Struct parser_struct;
  if (auto FnAST = ParseDefinition(parser_struct)) {

    FunctionProtos[FnAST->getProto().getName()] =
      std::make_unique<PrototypeAST>(FnAST->getProto());

    ExitOnErr(TheJIT->addAST(std::move(FnAST)));
  } else {
    // Skip token for error recovery.
    getNextToken();
  }
}

static void HandleExtern() {
  Parser_Struct parser_struct;
  if (auto ProtoAST = ParseExtern(parser_struct)) {
    if (auto *FnIR = ProtoAST->codegen()) {
      fprintf(stderr, "Read extern: ");
      FnIR->print(errs());
      fprintf(stderr, "\n");
      FunctionProtos[ProtoAST->getName()] = std::move(ProtoAST);
    }
  } else {
    // Skip token for error recovery.
    getNextToken();
  }
}

std::vector<std::thread> all_threads;

static void CodegenTopLevelExpression(std::unique_ptr<FunctionAST> &FnAST) {

    auto *FnIR =  FnAST->codegen();

    /*
    fprintf(stderr, "\nRead top-level expression:");
    FnIR->print(errs());
    fprintf(stderr, "\n\n");
    */


    // TheModule->print(llvm::errs(), nullptr);

    // Create a ResourceTracker for memory managment
    // anonymous expression -- that way we can free it after executing.
    auto RT = TheJIT->getMainJITDylib().createResourceTracker();

    auto TSM = ThreadSafeModule(std::move(TheModule), std::move(TheContext));
    ExitOnErr(TheJIT->addModule(std::move(TSM), RT));
    // Add IR module


    InitializeModule();

    // Points __anon_expr
    auto Sym = ExitOnErr(TheJIT->lookup("__anon_expr"));
    //assert(Sym && "Function not found");
      
      
    // Get the symbol's address and cast it to the right type (takes no
    // arguments, returns a float) so we can call it as a native function.
    auto *FP = Sym.getAddress().toPtr<float (*)()>();
    auto fp = FP();
    
    // fprintf(stderr, "%.2f\n", fp);

    // Delete the anonymous expression module from the JIT.
    ExitOnErr(RT->remove());    
}



static void HandleTopLevelExpression() {
  // Evaluate a top-level expression into an anonymous function.
  
  Parser_Struct parser_struct;
  parser_struct.function_name = "__anon_expr";
  if (std::unique_ptr<FunctionAST> FnAST = ParseTopLevelExpr(parser_struct)) {
    CodegenTopLevelExpression(std::ref(FnAST));

  
  } else {
    // Skip token for error recovery.
    getNextToken();
  }
}


void InitializeTokenizer() {

    if (Sys_Arguments.size()>0)
    {
        tokenizer.openFile(Sys_Arguments[0]);
        getNextToken();

    } else
        getNextToken();

}

/// top ::= definition | external | expression | ';'
static void MainLoop() {


    while (true) {
        //  std::cout << "MAIN LOOP, reading token: " << ReverseToken(CurTok) << "\n";
        

        switch (CurTok) {
        case 13:
            std::cout << "FOUND CARRIAGE RETURN" << ".\n";
            break;
        case tok_eof:
            return;
        case ';': // ignore top-level semicolons.
            getNextToken();
            break;
        case tok_space:
            getNextToken();
            break;
        case tok_tab:
            getNextToken();
            break;
        case tok_def:
            HandleDefinition();
            break;
        case tok_class:
            HandleClass();
            break;
        case tok_import:
            HandleImport();
            break;
        case tok_extern:
            HandleExtern();
            break;
        default:
            
            // std::cout << "Wait top level" <<  ".\n";
            // std::cout << "reading token: " << CurTok << "/" << ReverseToken(CurTok) << "\n";

            HandleTopLevelExpression(); 
            // std::cout << "Finished top level" <<  ".\n";
            break;
        }
    }
}


//===----------------------------------------------------------------------===//
// "Library" functions that can be "extern'd" from user code.
//===----------------------------------------------------------------------===//

/// putchard - putchar that takes a float and returns 0.
extern "C" float putchard(float X) {
  fputc((char)X, stderr);
  return 0;
}

/// printd - printf that takes a float prints it as "%f\n", returning 0.
extern "C" float printd(float X) {
  fprintf(stderr, "%f\n", X);
  return 0;
}

//===----------------------------------------------------------------------===//
// Main driver code.
//===----------------------------------------------------------------------===//

__attribute__((constructor))
void early_init() {
    // std::cout << "Constructor Function Executed\n";
  InitializeNativeTarget();
  InitializeNativeTargetAsmPrinter(); // Prepare for target hardware
  InitializeNativeTargetAsmParser();
}

int main(int argc, char* argv[]) {



    for (int i = 1; i < argc; i++) {
        // std::cout << "Argument " << i << ": " << argv[i] << "\n";
        Sys_Arguments.push_back(argv[i]);
        if(i==1) {
           tokenizer.files.pop();
           tokenizer.files.push(argv[i]);
        }
    }
  
  



  // Install standard binary operators.
  // 1 is lowest precedence.
  BinopPrecedence[tok_space] = 1;
  BinopPrecedence['='] = 4;
  BinopPrecedence[tok_arrow] = 4;
  BinopPrecedence['!'] = 9;
  BinopPrecedence[tok_and] = 9;
  BinopPrecedence[tok_not] = 9;
  BinopPrecedence[tok_or] = 9;
  BinopPrecedence[tok_xor] = 9;
  BinopPrecedence['>'] = 10;
  BinopPrecedence['<'] = 10;
  BinopPrecedence[tok_equal] = 10;
  BinopPrecedence[tok_diff] = 10;
  BinopPrecedence[tok_minor_eq] = 10;
  BinopPrecedence[tok_higher_eq] = 10;
  BinopPrecedence['+'] = 20;
  BinopPrecedence['-'] = 20;
  BinopPrecedence['%'] = 35;
  BinopPrecedence['*'] = 39;
  BinopPrecedence['/'] = 40;
  BinopPrecedence[tok_int_div] = 40;
  BinopPrecedence['^'] = 50;
  BinopPrecedence['@'] = 60;


  floatFunctions["log"] = "logE";
  floatFunctions["log2"] = "logE2";
  floatFunctions["log2f"] = "logE2f";
  floatFunctions["round"] = "roundE";
  floatFunctions["floor"] = "floorE";

    gc_sizes[0] = 8;
    gc_sizes[1] = 16;
    gc_sizes[2] = 24;
    gc_sizes[3] = 48;
    gc_sizes[4] = 64;
    gc_sizes[5] = 128;
    gc_sizes[6] = 256;
    gc_sizes[7] = 384;
    gc_sizes[8] = 512;
    gc_sizes[9] = 768;
    gc_sizes[10] = 1024;
    gc_sizes[11] = 2048;
    gc_sizes[12] = 4096;
    gc_sizes[13] = 8192;
    gc_sizes[14] = GC_max_object_size;


    for (int i=0, c=0; i<=GC_N; i++) {
        int size = i * GC_ALIGN;
        while (c<GC_obj_sizes-1 && gc_sizes[c]<size)
            c++;
        GC_size_to_class[i] = gc_sizes[c];
    }



    


  set_functions_return_type();
  set_functions_args_type();
  set_user_functions();
  vararg_methods = {"tensor_view", "tensor_sum", "tensor_mean", "mean_tensor" ,"tensor_prod", "tensor_tmax", "tensor_argmax", "tensor_load_bin_idx", "zip"};


  return_tensor_functions = {"gelu", "sigmoid", "_tanh", "relu", "softmax", "log", "randu_like",
                             "RandomCrop", "RandomHorizontalFlip", "NormalizeImg", "dropout", "sigmoid_add2weights",
                             "rl_discounted_return", "self_attn", "Jitter", "mse_with_priorities",
                             "btc_mult", "btc_multT", "Linear"};

  
  

  return_tensor_fn = concat_str_vec(return_tensor_functions, return_tensor_methods);

  return_pinned_methods = {"gpu", "gpuw"};


  // Universal
  string_methods = {"split", "split_idx"};


  // tensor + string + ...
  // e.g: x.view(), str.split()
  native_methods = {"split", "split_idx", "str_vec_print"};
  native_methods = concat_str_vec(native_methods, return_tensor_methods);
  native_methods = concat_str_vec(native_methods, user_cpp_functions);

  return_string_fn = {"to_string", "cat_str_float"};


  native_functions = {"ShuffleStrVec", "gload_img", "wload_img", "silent_sleep", "__slee_p_",
                      "LenStrVec", "zeros_vec", "ones_vec", "start_timer", "end_timer",
                      "_glob_b_", "print", "cross_entropy", "backprop", "AdamW", "SGD",
                      "load_preprocess_img", "max", "min", "unbug",
                      "cpu_idx", "OneCycleLR", "CosineLR", "wload_img_resize",
                      "build_vocab", "tokenize", "wtokenize", "write_zerosw",
                      "wtokenize_pad_left", "print_randoms", "wtokenize_pad_left_batch_first",
                      "wtokenize_pad_left_idx", "print_scope", "load_bin", "wload_bin", "randint",
                      "print_tensor", "path_exists", "dir_exists", "load_bin_idx",
                      "network_ema", "mse", "priority_sample", "priority_sample_val",
                      "importance_sample_idx", "importance_sample_weight",
                      "cross_entropy_idx"};
  native_functions = concat_str_vec(native_functions, return_tensor_functions);
  native_functions = concat_str_vec(native_functions, return_string_fn);
  native_fn = concat_str_vec(native_methods, native_functions);



  reverse_ops = {{"float_tensor", "tensor_float"}};

  elements_type_return = {{"tensor_tensor", "tensor"}, {"float_float", "float"}, {"str_str", "str"}, {"str_float", "str"},
                     {"float_str", "str"}, {"int_int", "int"}, {"int_float", "float"}, {"float_int", "float"}, {"str_int", "str"}, {"int_str", "str"},
                     {"str_bool", "str"}, {"bool_str", "str"}, {"bool_bool", "bool"},
                     {"tensor_float", "tensor"}, {"pinned_tensor_pinned_tensor", "pinned_tensor"},
                     {"pinned_tensor_tensor", "pinned_tensor"}, {"pinned_tensor_float", "pinned_tensor"},
                     {"object_object", "object"}, {"str_object", "object"},
                     {"tensor_int", "tensor"}, {"int_tensor", "tensor"}, {"str_channel", "str"}, {"channel_str", "float"}, {"channel_int", "float"},
                     {"int_channel", "int"}, {"channel_float", "float"}, {"float_channel", "float"}};

  ops_type_return = {{"int_int_higher", "bool"}, {"int_int_minor", "bool"}, {"int_int_equal", "bool"}, {"int_int_different", "bool"},
                     {"int_int_higher_eq", "bool"}, {"int_int_minor_eq", "bool"},
                     {"float_float_higher", "bool"}, {"float_float_minor", "bool"}, {"float_float_equal", "bool"}, {"float_float_different", "bool"},
                     {"float_float_higher_eq", "bool"}, {"float_float_minor_eq", "bool"}};

                     

  op_map = {{'*', "mult"}, {'@', "mma"},  {'+', "add"}, {'-', "sub"}, {'/', "div"}, {'<', "minor"}, {'>', "higher"}, {tok_equal, "equal"},
            {tok_diff, "different"}, {tok_higher_eq, "higher_eq"}, {tok_minor_eq, "minor_eq"}, {'%', "mod"}, {'=', "attr"},
            {77, "error"}, {tok_arrow, "message"}, {tok_and, "and"}, {tok_not, "not"}, {tok_or, "or"}, {tok_xor, "xor"}};

  for (auto pair : op_map)
    op_map_names.push_back(pair.second);


  
  notators_str = {"bias", "fp32", "fp16", "causal"};


  // Prime the first token.


  InitializeTokenizer();

  TheJIT = ExitOnErr(KaleidoscopeJIT::Create());
  InitializeModule();


  MainLoop();

  return 0;
}

