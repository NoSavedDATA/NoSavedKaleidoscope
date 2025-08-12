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
#include <glob.h>
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
std::map<std::string, DT_float_vec *> ClassFloatVecs;
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
  Value *thread_id, *has_grad, *scope_struct;

  
  scope_struct = callret("scope_struct_Create", {});


  thread_id = ConstantInt::get(Type::getInt32Ty(*TheContext), 0);
  has_grad  = ConstantInt::get(Type::getInt32Ty(*TheContext), 1);

 
  
  call("set_scope_has_grad", {scope_struct, has_grad});
  

  current_codegen_function = function_name;




//   std::cout << "\033[32mExecuting function: " << function_name << " \033[0m\n";

  NamedValues.clear();

  
  
  call("scope_struct_Alloc_MarkSweepMap", {scope_struct}); 




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

        if (typeVars[function_name].find(arg_name) != typeVars[function_name].end())
            type = typeVars[function_name][arg_name];
        // std::cout << "------------------------------------TYPE OF " << arg_name << " IS " << type << ".\n";


        
        llvm::Type *alloca_type = get_type_from_str(type);
        AllocaInst *arg_alloca = CreateEntryBlockAlloca(TheFunction, arg_name, alloca_type);


        
        std::string copy_fn = type+"_CopyArg";
        Function *F = TheModule->getFunction(copy_fn);
        if (F)
        {
            // Value *copied_value = callret(copy_fn, // Moved to CallExpr
            //                 {scope_struct,
            //                 &Arg,
            //                 global_str(arg_name)}); 
            Builder->CreateStore(&Arg, arg_alloca);
            call("MarkToSweep_Mark", {scope_struct, &Arg, global_str(Extract_List_Suffix(type))});
            call("MarkToSweep_Unmark_Scopeful", {scope_struct, &Arg});
        } else
        {
            Builder->CreateStore(&Arg, arg_alloca);
            if(type!="float"&&type!="int")
            {
                // p2t("HELLO FROM ELSE for arg_name " + arg_name + " of type " + type);
                // call("MarkToSweep_Unmark_Scopeless", {scope_struct, &Arg});
            }
        }
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


  //===----------------------------------------------------------------------===//
  // Tensor -- Scalar   Operations
  //===----------------------------------------------------------------------===//

  //
  FunctionType *CudaScalarMultTy = FunctionType::get(
      int8PtrTy,
      {int8PtrTy, Type::getFloatTy(*TheContext), int8PtrTy}, 
      false 
  );
  TheModule->getOrInsertFunction("tensor_float_mult", CudaScalarMultTy);


  //
  FunctionType *CudaScalarDivTy = FunctionType::get(
      int8PtrTy,
      {int8PtrTy, Type::getFloatTy(*TheContext), int8PtrTy}, 
      false
  );
  TheModule->getOrInsertFunction("tensor_float_div", CudaScalarDivTy);


//   //
//   FunctionType *CudaReverseScalarDivTy = FunctionType::get(
//       int8PtrTy,
//       {int8PtrTy, Type::getFloatTy(*TheContext), Type::getInt32Ty(*TheContext)}, 
//       false
//   );
//   TheModule->getOrInsertFunction("CudaReverseScalarDiv", CudaReverseScalarDivTy);


  //
  FunctionType *CudaScalarAddTy = FunctionType::get(
      int8PtrTy,
      {int8PtrTy, Type::getFloatTy(*TheContext), int8PtrTy}, 
      false
  );
  TheModule->getOrInsertFunction("tensor_float_add", CudaScalarAddTy);


  //
  FunctionType *CudaScalarSubTy = FunctionType::get(
      int8PtrTy,
      {int8PtrTy, Type::getFloatTy(*TheContext), int8PtrTy}, 
      false
  );
  TheModule->getOrInsertFunction("tensor_float_sub", CudaScalarSubTy);


  //
  FunctionType *CudaScalarEqualTy = FunctionType::get(
      int8PtrTy,
      {int8PtrTy, Type::getFloatTy(*TheContext), int8PtrTy}, 
      false
  );
  TheModule->getOrInsertFunction("tensor_float_equal", CudaScalarEqualTy);


  //
  FunctionType *CudaScalarDiffTy = FunctionType::get(
      int8PtrTy,
      {int8PtrTy, Type::getFloatTy(*TheContext), int8PtrTy}, 
      false
  );
  TheModule->getOrInsertFunction("tensor_float_diff", CudaScalarDiffTy);


  //
  FunctionType *CudaScalarMinorTy = FunctionType::get(
      int8PtrTy,
      {int8PtrTy, Type::getFloatTy(*TheContext), int8PtrTy}, 
      false
  );
  TheModule->getOrInsertFunction("tensor_float_minor", CudaScalarMinorTy);


  //
  FunctionType *CudaScalarHigherTy = FunctionType::get(
      int8PtrTy,
      {int8PtrTy, Type::getFloatTy(*TheContext), int8PtrTy}, 
      false
  );
  TheModule->getOrInsertFunction("tensor_float_higher", CudaScalarHigherTy);

  
  //
  FunctionType *CudaScalarHigherEqTy = FunctionType::get(
      int8PtrTy,
      {int8PtrTy, Type::getFloatTy(*TheContext), int8PtrTy}, 
      false
  );
  TheModule->getOrInsertFunction("tensor_float_higher_eq", CudaScalarHigherEqTy);


  //
  FunctionType *CudaScalarMinorEqTy = FunctionType::get(
      int8PtrTy,
      {int8PtrTy, Type::getFloatTy(*TheContext), int8PtrTy}, 
      false
  );
  TheModule->getOrInsertFunction("tensor_float_minor_eq", CudaScalarMinorEqTy);


  

  //===----------------------------------------------------------------------===//
  // Tensor Tensor CUDA Ops
  //===----------------------------------------------------------------------===//

  //
  FunctionType *CudaMultTy = FunctionType::get(
      int8PtrTy,
      {int8PtrTy, int8PtrTy, int8PtrTy},
      false
  );
  TheModule->getOrInsertFunction("tensor_tensor_mma", CudaMultTy);


  //
  FunctionType *CudaAddTy = FunctionType::get(
      int8PtrTy,
      {int8PtrTy, int8PtrTy, int8PtrTy},
      false
  );
  TheModule->getOrInsertFunction("tensor_tensor_add", CudaAddTy);


  //
  FunctionType *CudaSubTy = FunctionType::get(
      int8PtrTy,
      {int8PtrTy, int8PtrTy, int8PtrTy},
      false
  );
  TheModule->getOrInsertFunction("tensor_tensor_sub", CudaSubTy);


  //
  FunctionType *CudaEqualTy = FunctionType::get(
      int8PtrTy,
      {int8PtrTy, int8PtrTy, int8PtrTy},
      false
  );
  TheModule->getOrInsertFunction("tensor_tensor_equal", CudaEqualTy);


  //
  FunctionType *CudaHadamardTy = FunctionType::get(
      int8PtrTy,
      {int8PtrTy, int8PtrTy, int8PtrTy},
      false
  );
  TheModule->getOrInsertFunction("tensor_tensor_mult", CudaHadamardTy);


  //
  FunctionType *CudaDivTy = FunctionType::get(
      int8PtrTy,
      {int8PtrTy, int8PtrTy, int8PtrTy},
      false
  );
  TheModule->getOrInsertFunction("tensor_tensor_div", CudaDivTy);

  //
  FunctionType *str_DeleteTy = FunctionType::get(
      int8PtrTy,
      {int8PtrTy},
      false
  );
  TheModule->getOrInsertFunction("str_Delete", str_DeleteTy);


  //
  FunctionType *IdxTensorTy = FunctionType::get(
      floatPtrTy,
      {int8PtrTy, int8PtrTy, Type::getInt32Ty(*TheContext), Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext)}, 
      true // vararg
  );
  TheModule->getOrInsertFunction("IdxTensor", IdxTensorTy);


  //
  FunctionType *AttrPinnedFromTensorOnIdxTy = FunctionType::get(
      floatPtrTy,
      {int8PtrTy, int8PtrTy, Type::getInt32Ty(*TheContext), Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext)}, 
      true // vararg
  );
  TheModule->getOrInsertFunction("AttrPinnedFromTensorOnIdx", AttrPinnedFromTensorOnIdxTy);


  //
  FunctionType *IdxTensorWithTensorTy = FunctionType::get(
      floatPtrTy,
      {int8PtrTy, int8PtrTy, Type::getInt32Ty(*TheContext)}, 
      false
  );
  TheModule->getOrInsertFunction("IdxTensorWithTensor", IdxTensorWithTensorTy);


  //
  FunctionType *PrintTensorFTy = FunctionType::get(
      Type::getFloatTy(*TheContext),
      {floatPtrTy,
       Type::getInt32Ty(*TheContext),
       Type::getInt32Ty(*TheContext),}, 
      false
  );
  TheModule->getOrInsertFunction("PrintTensorF", PrintTensorFTy);


  //
  FunctionType *print_tensorTy = FunctionType::get(
      Type::getFloatTy(*TheContext),
      {int8PtrTy}, 
      false
  );
  TheModule->getOrInsertFunction("print_tensor", print_tensorTy);


  //
  FunctionType *LoadDimsTy = FunctionType::get(
      int8PtrTy,
      {int8PtrTy},
      false
  );
  TheModule->getOrInsertFunction("LoadDims", LoadDimsTy);


  //
  FunctionType *PrintDimsTy = FunctionType::get(
      Type::getVoidTy(*TheContext),
      {int8PtrTy}, 
      false 
  );
  TheModule->getOrInsertFunction("PrintDims", PrintDimsTy);
  

  //
  FunctionType *clipTy = FunctionType::get(
      floatPtrTy,
      {Type::getInt32Ty(*TheContext), int8PtrTy,
       Type::getInt32Ty(*TheContext),
       Type::getInt32Ty(*TheContext)}, 
      false
  );
  TheModule->getOrInsertFunction("clip", clipTy);


  //
  FunctionType *network_emaTy = FunctionType::get(
      Type::getFloatTy(*TheContext),
      {Type::getInt32Ty(*TheContext), int8PtrTy, int8PtrTy, int8PtrTy, Type::getFloatTy(*TheContext)},
      false
  );
  TheModule->getOrInsertFunction("network_ema", network_emaTy);


  //===----------------------------------------------------------------------===//
  // Backward and Optimizers CUDA Ops
  //===----------------------------------------------------------------------===//

  //
  FunctionType *BackpropagationTy = FunctionType::get(
      Type::getFloatTy(*TheContext),
      {int8PtrTy}, 
      false
  );
  TheModule->getOrInsertFunction("backprop", BackpropagationTy);


  //
  FunctionType *clean_forwardTy = FunctionType::get(
      Type::getFloatTy(*TheContext),
      {int8PtrTy}, 
      false
  );
  TheModule->getOrInsertFunction("clean_forward", clean_forwardTy);
    
  
  //
  FunctionType *SGDTy = FunctionType::get(
      Type::getFloatTy(*TheContext),
      {int8PtrTy,
       Type::getFloatTy(*TheContext),
       Type::getFloatTy(*TheContext),
       Type::getFloatTy(*TheContext),
       Type::getFloatTy(*TheContext)}, 
      false
  );
  TheModule->getOrInsertFunction("SGD", SGDTy);
  
  
  //
  FunctionType *AdamWTy = FunctionType::get(
      Type::getFloatTy(*TheContext),
      {int8PtrTy, Type::getFloatTy(*TheContext),
       Type::getFloatTy(*TheContext),
       Type::getFloatTy(*TheContext),
       Type::getFloatTy(*TheContext),
       Type::getFloatTy(*TheContext)}, 
      false
  );
  TheModule->getOrInsertFunction("AdamW", AdamWTy);
  
  
  //
  FunctionType *OneCycleLRTy = FunctionType::get(
      Type::getFloatTy(*TheContext),
      {int8PtrTy, Type::getFloatTy(*TheContext),
       Type::getFloatTy(*TheContext),
       Type::getFloatTy(*TheContext)}, 
      false
  );
  TheModule->getOrInsertFunction("OneCycleLR", OneCycleLRTy);
  
  
  //
  FunctionType *CosineLRTy = FunctionType::get(
      Type::getFloatTy(*TheContext),
      {int8PtrTy, Type::getFloatTy(*TheContext),
       Type::getFloatTy(*TheContext),
       Type::getFloatTy(*TheContext),
       Type::getFloatTy(*TheContext)}, 
      false
  );
  TheModule->getOrInsertFunction("CosineLR", CosineLRTy);



  //===----------------------------------------------------------------------===//
  // Unary CUDA Ops
  //===----------------------------------------------------------------------===//

  //
  FunctionType *CudaLogTy = FunctionType::get(
      int8PtrTy,
      {Type::getInt32Ty(*TheContext), int8PtrTy},
      false 
  );
  TheModule->getOrInsertFunction("logE", CudaLogTy);
  

  // 
  FunctionType *log2Ty = FunctionType::get(
      int8PtrTy,
      {Type::getInt32Ty(*TheContext), int8PtrTy},
      false
  );
  TheModule->getOrInsertFunction("logE2", log2Ty);


  // 
  FunctionType *btc_multTy = FunctionType::get(
      int8PtrTy,
      {Type::getInt32Ty(*TheContext), int8PtrTy, int8PtrTy},
      false
  );
  TheModule->getOrInsertFunction("btc_mult", btc_multTy);


  // 
  FunctionType *btc_multTTy = FunctionType::get(
      int8PtrTy,
      {Type::getInt32Ty(*TheContext), int8PtrTy, int8PtrTy},
      false
  );
  TheModule->getOrInsertFunction("btc_multT", btc_multTTy);


  // 
  FunctionType *softmaxTy = FunctionType::get(
      int8PtrTy,
      {int8PtrTy, int8PtrTy},
      false
  );
  TheModule->getOrInsertFunction("softmax", softmaxTy);


  // 
  FunctionType *priority_sampleTy = FunctionType::get(
      Type::getFloatTy(*TheContext),
      {Type::getInt32Ty(*TheContext), int8PtrTy, Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext)},
      false
  );
  TheModule->getOrInsertFunction("priority_sample", priority_sampleTy);


  // 
  FunctionType *priority_sample_valTy = FunctionType::get(
      Type::getFloatTy(*TheContext),
      {Type::getInt32Ty(*TheContext), int8PtrTy, Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext)},
      false
  );
  TheModule->getOrInsertFunction("priority_sample_val", priority_sample_valTy);


  // 
  FunctionType *importance_sample_idxTy = FunctionType::get(
      Type::getFloatTy(*TheContext),
      {Type::getInt32Ty(*TheContext), int8PtrTy, Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext)},
      false
  );
  TheModule->getOrInsertFunction("importance_sample_idx", importance_sample_idxTy);


  // 
  FunctionType *importance_sample_weightTy = FunctionType::get(
      Type::getFloatTy(*TheContext),
      {Type::getInt32Ty(*TheContext), int8PtrTy, Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext)},
      false
  );
  TheModule->getOrInsertFunction("importance_sample_weight", importance_sample_weightTy);


  // 
  FunctionType *self_attnTy = FunctionType::get(
      int8PtrTy,
      {Type::getInt32Ty(*TheContext), int8PtrTy},
      false
  );
  TheModule->getOrInsertFunction("self_attn", self_attnTy);
  

  //
  FunctionType *reluTy = FunctionType::get(
      int8PtrTy,
      {int8PtrTy, int8PtrTy},
      false
  );
  TheModule->getOrInsertFunction("relu", reluTy);
  

  //
  FunctionType *gatherTy = FunctionType::get(
      int8PtrTy,
      {Type::getInt32Ty(*TheContext), int8PtrTy, int8PtrTy, Type::getFloatTy(*TheContext)},
      false
  );
  TheModule->getOrInsertFunction("gather", gatherTy);
  

  //
  FunctionType *rl_discounted_returnTy = FunctionType::get(
      int8PtrTy,
      {Type::getInt32Ty(*TheContext), int8PtrTy, int8PtrTy, Type::getFloatTy(*TheContext)},
      false
  );
  TheModule->getOrInsertFunction("rl_discounted_return", rl_discounted_returnTy);
  

  //
  FunctionType *geluTy = FunctionType::get(
      int8PtrTy,
      {int8PtrTy, int8PtrTy},
      false
  );
  TheModule->getOrInsertFunction("gelu", geluTy);
  

  //
  FunctionType *sigmoidTy = FunctionType::get(
      int8PtrTy,
      {int8PtrTy, int8PtrTy},
      false
  );
  TheModule->getOrInsertFunction("sigmoid", sigmoidTy);
  

  //
  FunctionType *sigmoid_add2weightsTy = FunctionType::get(
      int8PtrTy,
      {int8PtrTy, int8PtrTy, int8PtrTy, int8PtrTy},
      false
  );
  TheModule->getOrInsertFunction("sigmoid_add2weights", sigmoid_add2weightsTy);
  

  //
  FunctionType *tanhTy = FunctionType::get(
      int8PtrTy,
      {int8PtrTy, int8PtrTy},
      false
  );
  TheModule->getOrInsertFunction("_tanh", tanhTy);


  //
  FunctionType *BatchNorm2dTy = FunctionType::get(
      int8PtrTy,
      {int8PtrTy, int8PtrTy},
      false
  );
  TheModule->getOrInsertFunction("BatchNorm2d", BatchNorm2dTy);



  //
  FunctionType *Pool2dTy = FunctionType::get(
      int8PtrTy,
      {int8PtrTy, int8PtrTy},
      false
  );
  TheModule->getOrInsertFunction("Pool2d", Pool2dTy);



  //
  FunctionType *conv2dForwardTy = FunctionType::get(
      int8PtrTy,
      {int8PtrTy, int8PtrTy},
      false
  );
  TheModule->getOrInsertFunction("Conv2d", conv2dForwardTy);


  //
  FunctionType *LSTMForwardTy = FunctionType::get(
      int8PtrTy,
      {int8PtrTy, int8PtrTy, int8PtrTy, int8PtrTy, Type::getInt32Ty(*TheContext), int8PtrTy, Type::getInt32Ty(*TheContext)},
      false
  );
  TheModule->getOrInsertFunction("LSTMForward", LSTMForwardTy);


  //
  FunctionType *MHSAForwardTy = FunctionType::get(
      int8PtrTy,
      {int8PtrTy, int8PtrTy, Type::getInt32Ty(*TheContext), int8PtrTy, Type::getInt32Ty(*TheContext)},
      false
  );
  TheModule->getOrInsertFunction("MHSAForward", MHSAForwardTy);


  //
  FunctionType *LinearForwardTy = FunctionType::get(
      int8PtrTy,
      {int8PtrTy, int8PtrTy, Type::getInt32Ty(*TheContext), int8PtrTy, Type::getInt32Ty(*TheContext)},
      false
  );
  TheModule->getOrInsertFunction("LinearForward", LinearForwardTy);


  FunctionType *LinearTy = FunctionType::get(
      int8PtrTy,
      {int8PtrTy, int8PtrTy},
      false
  );
  TheModule->getOrInsertFunction("Linear", LinearTy);

  //
  FunctionType *EmbeddingForwardTy = FunctionType::get(
      int8PtrTy,
      {int8PtrTy, int8PtrTy, Type::getInt32Ty(*TheContext), int8PtrTy, Type::getInt32Ty(*TheContext)},
      false
  );
  TheModule->getOrInsertFunction("EmbeddingForward", EmbeddingForwardTy);


  //
  FunctionType *MaxPoolForward2dTy = FunctionType::get(
      int8PtrTy,
      {int8PtrTy, int8PtrTy, Type::getInt32Ty(*TheContext), int8PtrTy, Type::getInt32Ty(*TheContext)},
      false
  );
  TheModule->getOrInsertFunction("MaxPoolForward2d", MaxPoolForward2dTy);




  //
  FunctionType *BN2dReluForwardTy = FunctionType::get(
      int8PtrTy,
      {int8PtrTy, int8PtrTy, Type::getInt32Ty(*TheContext), int8PtrTy, Type::getInt32Ty(*TheContext)},
      false
  );
  TheModule->getOrInsertFunction("BN2dReluForward", BN2dReluForwardTy);


  //
  FunctionType *ReluForwardTy = FunctionType::get(
      int8PtrTy,
      {int8PtrTy, int8PtrTy, int8PtrTy, Type::getInt32Ty(*TheContext)},
      false
  );
  TheModule->getOrInsertFunction("ReluForward", ReluForwardTy);


  //
  FunctionType *cropTy = FunctionType::get(
      int8PtrTy,
      {int8PtrTy, Type::getFloatTy(*TheContext), Type::getInt32Ty(*TheContext)},
      false
  );
  TheModule->getOrInsertFunction("RandomCrop", cropTy);


  //
  FunctionType *RandomHorizontalFlipTy = FunctionType::get(
      int8PtrTy,
      {int8PtrTy, Type::getInt32Ty(*TheContext)},
      false
  );
  TheModule->getOrInsertFunction("RandomHorizontalFlip", RandomHorizontalFlipTy);


  //
  FunctionType *NormalizeImgTy = FunctionType::get(
      int8PtrTy,
      {int8PtrTy, int8PtrTy, int8PtrTy, Type::getInt32Ty(*TheContext)},
      false
  );
  TheModule->getOrInsertFunction("NormalizeImg", NormalizeImgTy);


  //
  FunctionType *JitterTy = FunctionType::get(
      int8PtrTy,
      {int8PtrTy, int8PtrTy, Type::getFloatTy(*TheContext)},
      false
  );
  TheModule->getOrInsertFunction("Jitter", JitterTy);


  //
  FunctionType *dropoutTy = FunctionType::get(
      int8PtrTy,
      {int8PtrTy, Type::getFloatTy(*TheContext), Type::getInt32Ty(*TheContext)},
      false
  );
  TheModule->getOrInsertFunction("dropout", dropoutTy);
  

  //
  FunctionType *onehotTy = FunctionType::get(
      int8PtrTy,
      {int8PtrTy, int8PtrTy, Type::getFloatTy(*TheContext)},
      false
  );
  TheModule->getOrInsertFunction("tensor_onehot", onehotTy);
  

  //
  FunctionType *shapeTy = FunctionType::get(
      Type::getFloatTy(*TheContext),
      {int8PtrTy, int8PtrTy},
      false
  );
  TheModule->getOrInsertFunction("tensor_shape", shapeTy);
  

  //
  FunctionType *printttTy = FunctionType::get(
      Type::getFloatTy(*TheContext),
      {Type::getInt32Ty(*TheContext), int8PtrTy},
      false
  );
  TheModule->getOrInsertFunction("printtt", printttTy);
  

  //
  FunctionType *evalTy = FunctionType::get(
      Type::getFloatTy(*TheContext),
      {int8PtrTy},
      false
  );
  TheModule->getOrInsertFunction("eval", evalTy);
  

  //
  FunctionType *trainTy = FunctionType::get(
      Type::getFloatTy(*TheContext),
      {int8PtrTy},
      false
  );
  TheModule->getOrInsertFunction("train", trainTy);

  
  //
  FunctionType *repeat_interleaveTy = FunctionType::get(
      int8PtrTy,
      {Type::getInt32Ty(*TheContext), int8PtrTy, Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext)},
      false
  );
  TheModule->getOrInsertFunction("repeat_interleave", repeat_interleaveTy);
  

  // 
  FunctionType *sumTy = FunctionType::get(
      int8PtrTy,
      {Type::getInt32Ty(*TheContext), int8PtrTy, Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext)},
      true // vararg
  );
  TheModule->getOrInsertFunction("sum", sumTy);


  // 
  FunctionType *prodTy = FunctionType::get(
      int8PtrTy,
      {Type::getInt32Ty(*TheContext), int8PtrTy, Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext)},
      true // vararg
  );
  TheModule->getOrInsertFunction("prod", prodTy);

  FunctionType *mean2Ty = FunctionType::get(
      int8PtrTy,
      {int8PtrTy, int8PtrTy, Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext)},
      true // vararg
  );
  TheModule->getOrInsertFunction("mean_tensor", mean2Ty);

  // 
  FunctionType *meanTy = FunctionType::get(
      int8PtrTy,
      {int8PtrTy, int8PtrTy, Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext)},
      true // vararg
  );
  TheModule->getOrInsertFunction("tensor_mean", meanTy);
  

  // 
  FunctionType *maxTy = FunctionType::get(
      int8PtrTy,
      {Type::getInt32Ty(*TheContext), int8PtrTy, Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext)},
      true // vararg
  );
  TheModule->getOrInsertFunction("tmax", maxTy);


  //
  FunctionType *argmaxTy = FunctionType::get(
      int8PtrTy,
      {int8PtrTy, int8PtrTy, Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext)},
      true // vararg
  );
  TheModule->getOrInsertFunction("tensor_argmax", argmaxTy);
  

  //
  FunctionType *topkTy = FunctionType::get(
      int8PtrTy,
      {Type::getInt32Ty(*TheContext), int8PtrTy, Type::getFloatTy(*TheContext)},
      true // vararg
  );
  TheModule->getOrInsertFunction("topk", topkTy);


  //
  FunctionType *print_floatTy = FunctionType::get(
      Type::getFloatTy(*TheContext),
      {int8PtrTy},
      false
  );
  TheModule->getOrInsertFunction("print_float", print_floatTy);
  

  //
  FunctionType *CalculateIdxOffsetTy = FunctionType::get(
      Type::getFloatTy(*TheContext),
      {int8PtrTy, Type::getFloatTy(*TheContext),Type::getFloatTy(*TheContext),Type::getFloatTy(*TheContext),Type::getFloatTy(*TheContext),Type::getFloatTy(*TheContext),Type::getFloatTy(*TheContext),Type::getFloatTy(*TheContext),Type::getFloatTy(*TheContext)},
      true // vararg
  );
  TheModule->getOrInsertFunction("CalculateIdxOffset", CalculateIdxOffsetTy);


  FunctionType *tensor_CalculateIdxTy = FunctionType::get(
      Type::getFloatTy(*TheContext),
      {int8PtrTy, Type::getFloatTy(*TheContext),Type::getFloatTy(*TheContext),Type::getFloatTy(*TheContext),Type::getFloatTy(*TheContext),Type::getFloatTy(*TheContext),Type::getFloatTy(*TheContext),Type::getFloatTy(*TheContext),Type::getFloatTy(*TheContext)},
      true // vararg
  );
  TheModule->getOrInsertFunction("tensor_CalculateIdx", tensor_CalculateIdxTy);

  FunctionType *pinned_tensor_CalculateIdxTy = FunctionType::get(
      Type::getFloatTy(*TheContext),
      {int8PtrTy, Type::getFloatTy(*TheContext),Type::getFloatTy(*TheContext),Type::getFloatTy(*TheContext),Type::getFloatTy(*TheContext),Type::getFloatTy(*TheContext),Type::getFloatTy(*TheContext),Type::getFloatTy(*TheContext),Type::getFloatTy(*TheContext)},
      true // vararg
  );
  TheModule->getOrInsertFunction("pinned_tensor_CalculateIdx", pinned_tensor_CalculateIdxTy);

  FunctionType *float_vec_CalculateIdxTy = FunctionType::get(
      Type::getFloatTy(*TheContext),
      {int8PtrTy, Type::getFloatTy(*TheContext),Type::getFloatTy(*TheContext),Type::getFloatTy(*TheContext),Type::getFloatTy(*TheContext),Type::getFloatTy(*TheContext),Type::getFloatTy(*TheContext),Type::getFloatTy(*TheContext),Type::getFloatTy(*TheContext)},
      true // vararg
  );
  TheModule->getOrInsertFunction("float_vec_CalculateIdx", float_vec_CalculateIdxTy);

  FunctionType *str_vec_CalculateIdxTy = FunctionType::get(
      Type::getFloatTy(*TheContext),
      {int8PtrTy, Type::getFloatTy(*TheContext),Type::getFloatTy(*TheContext),Type::getFloatTy(*TheContext),Type::getFloatTy(*TheContext),Type::getFloatTy(*TheContext),Type::getFloatTy(*TheContext),Type::getFloatTy(*TheContext),Type::getFloatTy(*TheContext)},
      true // vararg
  );
  TheModule->getOrInsertFunction("str_vec_CalculateIdx", str_vec_CalculateIdxTy);

  //
  

  //===----------------------------------------------------------------------===//
  // Loss CUDA Ops
  //===----------------------------------------------------------------------===//

  //
  FunctionType *cross_entropyTy = FunctionType::get(
      Type::getFloatTy(*TheContext),
      {int8PtrTy, int8PtrTy, int8PtrTy, Type::getFloatTy(*TheContext)}, 
      false
  );
  TheModule->getOrInsertFunction("cross_entropy", cross_entropyTy);


  //
  FunctionType *cross_entropy_idxTy = FunctionType::get(
      Type::getFloatTy(*TheContext),
      {int8PtrTy, int8PtrTy, int8PtrTy, Type::getFloatTy(*TheContext)}, 
      false
  );
  TheModule->getOrInsertFunction("cross_entropy_idx", cross_entropy_idxTy);


  //
  FunctionType *mseTy = FunctionType::get(
      Type::getFloatTy(*TheContext),
      {int8PtrTy, int8PtrTy, Type::getFloatTy(*TheContext)}, 
      false
  );
  TheModule->getOrInsertFunction("mse", mseTy);


  //
  FunctionType *mse_with_prioritiesTy = FunctionType::get(
      int8PtrTy,
      {Type::getInt32Ty(*TheContext), int8PtrTy, int8PtrTy, Type::getFloatTy(*TheContext), int8PtrTy}, 
      false
  );
  TheModule->getOrInsertFunction("mse_with_priorities", mse_with_prioritiesTy);
  

  //===----------------------------------------------------------------------===//
  // File Handling Ops
  //===----------------------------------------------------------------------===//
  
  //
  FunctionType *load_imgTy = FunctionType::get(
      floatPtrTy,
      {int8PtrTy, int8PtrTy},
      false 
  );
  TheModule->getOrInsertFunction("load_img", load_imgTy);
  

  //
  FunctionType *load_preprocess_imgTy = FunctionType::get(
      Type::getFloatTy(*TheContext),
      {int8PtrTy, int8PtrTy, int8PtrTy},
      false
  );
  TheModule->getOrInsertFunction("load_preprocess_img", load_preprocess_imgTy);
  

  //===----------------------------------------------------------------------===//
  // Pinned Tensor Ops
  //===----------------------------------------------------------------------===//

  //
  FunctionType *gload_imgTy = FunctionType::get(
      floatPtrTy,
      {int8PtrTy, int8PtrTy, int8PtrTy, Type::getFloatTy(*TheContext)},
      false 
  );
  TheModule->getOrInsertFunction("gload_img", gload_imgTy);


  //
  FunctionType *wload_imgTy = FunctionType::get(
      floatPtrTy,
      {int8PtrTy, int8PtrTy, int8PtrTy, Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext)},
      false
  );
  TheModule->getOrInsertFunction("wload_img", wload_imgTy);


  //
  FunctionType *load_binTy = FunctionType::get(
      Type::getFloatTy(*TheContext),
      {int8PtrTy, int8PtrTy},
      false
  );
  TheModule->getOrInsertFunction("load_bin", load_binTy);


  //
  FunctionType *wload_binTy = FunctionType::get(
      Type::getFloatTy(*TheContext),
      {int8PtrTy, int8PtrTy, Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext)},
      false
  );
  TheModule->getOrInsertFunction("wload_bin", wload_binTy);


  //
  FunctionType *load_bin_idxTy = FunctionType::get(
      Type::getFloatTy(*TheContext),
      {int8PtrTy, int8PtrTy, Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext)},
      true
  );
  TheModule->getOrInsertFunction("load_bin_idx", load_bin_idxTy);


  //
  FunctionType *save_as_binTy = FunctionType::get(
      Type::getFloatTy(*TheContext),
      {Type::getInt32Ty(*TheContext), int8PtrTy, int8PtrTy},
      false
  );
  TheModule->getOrInsertFunction("save_as_bin", save_as_binTy);


  //
  FunctionType *save_as_intTy = FunctionType::get(
      Type::getFloatTy(*TheContext),
      {Type::getInt32Ty(*TheContext), int8PtrTy, int8PtrTy},
      false
  );
  TheModule->getOrInsertFunction("save_as_int", save_as_intTy);


  //
  FunctionType *wload_img_resizeTy = FunctionType::get(
      floatPtrTy,
      {int8PtrTy, int8PtrTy, int8PtrTy, Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext)},
      false
  );
  TheModule->getOrInsertFunction("wload_img_resize", wload_img_resizeTy);


  //
  FunctionType *save_imgTy = FunctionType::get(
      floatPtrTy,
      {int8PtrTy, Type::getInt32Ty(*TheContext), int8PtrTy, int8PtrTy},
      false
  );
  TheModule->getOrInsertFunction("save_img", save_imgTy);
  

  //
  FunctionType *pinned_tensor_Store_IdxTy = FunctionType::get(
      Type::getVoidTy(*TheContext),
      {int8PtrTy, Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext)},
      false
  );
  TheModule->getOrInsertFunction("pinned_tensor_Store_Idx", pinned_tensor_Store_IdxTy);


  //
  FunctionType *gpuTy = FunctionType::get(
      Type::getFloatTy(*TheContext),
      {int8PtrTy, int8PtrTy, int8PtrTy},
      false 
  );
  TheModule->getOrInsertFunction("gpu", gpuTy);


  //  
  FunctionType *gpuwTy = FunctionType::get(
      Type::getFloatTy(*TheContext),
      {int8PtrTy, int8PtrTy, int8PtrTy, Type::getFloatTy(*TheContext)},
      false 
  );
  TheModule->getOrInsertFunction("tensor_gpuw", gpuwTy);


  //===----------------------------------------------------------------------===//
  // Parallel Ops
  //===----------------------------------------------------------------------===//

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


  //  
  FunctionType *start_timerTy = FunctionType::get(
      Type::getFloatTy(*TheContext),
      {int8PtrTy, Type::getFloatTy(*TheContext)},
      false
  );
  TheModule->getOrInsertFunction("start_timer", start_timerTy);


  //  
  FunctionType *end_timerTy = FunctionType::get(
      Type::getFloatTy(*TheContext),
      {int8PtrTy, Type::getFloatTy(*TheContext)},
      false
  );
  TheModule->getOrInsertFunction("end_timer", end_timerTy);
  


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


  // char *
  FunctionType *globTy = FunctionType::get(
      int8PtrTy,
      {int8PtrTy, int8PtrTy},
      false
  );
  TheModule->getOrInsertFunction("_glob_b_", globTy);


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
  FunctionType *StrToFloatTy = FunctionType::get(
      Type::getFloatTy(*TheContext),
      {int8PtrTy, int8PtrTy},
      false 
  );
  TheModule->getOrInsertFunction("StrToFloat", StrToFloatTy);


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
  FunctionType *LoadObjectScopeNameTy = FunctionType::get(
      int8PtrTy,
      {int8PtrTy},
      false 
  );
  TheModule->getOrInsertFunction("LoadObjectScopeName", LoadObjectScopeNameTy);



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
  FunctionType *float_vec_printTy = FunctionType::get(
      Type::getFloatTy(*TheContext),
      {int8PtrTy, int8PtrTy}, 
      false 
  );
  TheModule->getOrInsertFunction("float_vec_print", float_vec_printTy);

  FunctionType *float_vec_first_nonzeroTy = FunctionType::get(
      Type::getFloatTy(*TheContext),
      {int8PtrTy, int8PtrTy}, 
      false 
  );
  TheModule->getOrInsertFunction("float_vec_first_nonzero", float_vec_first_nonzeroTy);

  //
  FunctionType *LoadStrVecOnDemandTy = FunctionType::get(
      int8PtrTy,
      {int8PtrTy},
      false
  );
  TheModule->getOrInsertFunction("LoadStrVecOnDemand", LoadStrVecOnDemandTy);


  
  
  //
  FunctionType *float_vec_StoreTy = FunctionType::get(
      Type::getFloatTy(*TheContext),
      {int8PtrTy, int8PtrTy, int8PtrTy},
      false
  );
  TheModule->getOrInsertFunction("float_vec_Store", float_vec_StoreTy);

  FunctionType *float_vec_Store_IdxTy = FunctionType::get(
      Type::getFloatTy(*TheContext),
      {int8PtrTy, Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext), int8PtrTy},
      false
  );
  TheModule->getOrInsertFunction("float_vec_Store_Idx", float_vec_Store_IdxTy);


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

  
  //
  FunctionType *IndexClassFloatVecTy = FunctionType::get(
      Type::getFloatTy(*TheContext),
      {int8PtrTy, int8PtrTy, Type::getFloatTy(*TheContext)}, 
      false 
  );
  TheModule->getOrInsertFunction("float_vec_Idx", IndexClassFloatVecTy);


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

  
  //
  FunctionType *TokenizeTy = FunctionType::get(
      Type::getFloatTy(*TheContext),
      {int8PtrTy, Type::getFloatTy(*TheContext)},
      false 
  );
  TheModule->getOrInsertFunction("build_vocab", TokenizeTy);

  
  //
  FunctionType *tokenizeTy = FunctionType::get(
      Type::getFloatTy(*TheContext),
      {int8PtrTy, int8PtrTy},
      false 
  );
  TheModule->getOrInsertFunction("tokenize", tokenizeTy);

  
  //
  FunctionType *wtokenizeTy = FunctionType::get(
      Type::getFloatTy(*TheContext),
      {int8PtrTy, int8PtrTy, Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext)},
      false 
  );
  TheModule->getOrInsertFunction("wtokenize", wtokenizeTy);

  
  //
  FunctionType *wtokenize_pad_leftTy = FunctionType::get(
      Type::getFloatTy(*TheContext),
      {int8PtrTy, int8PtrTy, Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext)},
      false 
  );
  TheModule->getOrInsertFunction("wtokenize_pad_left", wtokenize_pad_leftTy);

  
  //
  FunctionType *wtokenize_pad_left_batch_firstTy = FunctionType::get(
      Type::getFloatTy(*TheContext),
      {int8PtrTy, int8PtrTy, Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext)},
      false 
  );
  TheModule->getOrInsertFunction("wtokenize_pad_left_batch_first", wtokenize_pad_left_batch_firstTy);

  
  //
  FunctionType *wtokenize_pad_left_idxTy = FunctionType::get(
      Type::getFloatTy(*TheContext),
      {int8PtrTy, int8PtrTy, Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext)},
      false 
  );
  TheModule->getOrInsertFunction("wtokenize_pad_left_idx", wtokenize_pad_left_idxTy);

  
  //
  FunctionType *write_zeroswTy = FunctionType::get(
      Type::getFloatTy(*TheContext),
      {int8PtrTy, Type::getFloatTy(*TheContext)},
      false 
  );
  TheModule->getOrInsertFunction("write_zerosw", write_zeroswTy);


  //
  FunctionType *InitObjectVecWithNullTy = FunctionType::get(
      Type::getFloatTy(*TheContext),
      {int8PtrTy, Type::getFloatTy(*TheContext)},
      false 
  );
  TheModule->getOrInsertFunction("InitObjectVecWithNull", InitObjectVecWithNullTy);


  //
  FunctionType *is_nullTy = FunctionType::get(
      Type::getFloatTy(*TheContext),
      {int8PtrTy},
      false 
  );
  TheModule->getOrInsertFunction("is_null", is_nullTy);


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
  FunctionType *objHashTy = FunctionType::get(
      int8PtrTy,
      {int8PtrTy, int8PtrTy},
      false 
  );
  TheModule->getOrInsertFunction("objHash", objHashTy);
  

  // 
  FunctionType *LoadObjectTy = FunctionType::get(
      int8PtrTy,
      {int8PtrTy},
      false 
  );
  TheModule->getOrInsertFunction("LoadObject", LoadObjectTy);
  

  //
  FunctionType *InstantiateObjectTy = FunctionType::get(
      Type::getVoidTy(*TheContext),
      {int8PtrTy, int8PtrTy},
      false 
  );
  TheModule->getOrInsertFunction("InstantiateObject", InstantiateObjectTy);
  

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


  FunctionType *Linear_Create = FunctionType::get(
      Type::getFloatTy(*TheContext),
      {int8PtrTy, int8PtrTy, Type::getFloatTy(*TheContext), int8PtrTy, int8PtrTy},
      false 
  );
  TheModule->getOrInsertFunction("Linear_Create", Linear_Create);

  FunctionType *Pool2d_Create = FunctionType::get(
      Type::getFloatTy(*TheContext),
      {int8PtrTy, int8PtrTy, Type::getFloatTy(*TheContext), int8PtrTy, int8PtrTy},
      false 
  );
  TheModule->getOrInsertFunction("Pool2d_Create", Pool2d_Create);

  FunctionType *Conv2d_Create = FunctionType::get(
      Type::getFloatTy(*TheContext),
      {int8PtrTy, int8PtrTy, Type::getFloatTy(*TheContext), int8PtrTy, int8PtrTy},
      false 
  );
  TheModule->getOrInsertFunction("Conv2d_Create", Conv2d_Create);

  FunctionType *BatchNorm2d_Create = FunctionType::get(
      Type::getFloatTy(*TheContext),
      {int8PtrTy, int8PtrTy, Type::getFloatTy(*TheContext), int8PtrTy, int8PtrTy},
      false 
  );
  TheModule->getOrInsertFunction("BatchNorm2d_Create", BatchNorm2d_Create);

  FunctionType *scope_struct_CreateTy = FunctionType::get(
      int8PtrTy,
      {},
      false 
  );
  TheModule->getOrInsertFunction("scope_struct_Create", scope_struct_CreateTy);


  FunctionType *set_scope_function_nameTy = FunctionType::get(
      int8PtrTy,
      {int8PtrTy, int8PtrTy},
      false 
  );
  TheModule->getOrInsertFunction("set_scope_function_name", set_scope_function_nameTy);


  FunctionType *set_scope_first_argTy = FunctionType::get(
      int8PtrTy,
      {int8PtrTy, int8PtrTy},
      false 
  );
  TheModule->getOrInsertFunction("set_scope_first_arg", set_scope_first_argTy);

  FunctionType *set_scope_scopeTy = FunctionType::get(
      int8PtrTy,
      {int8PtrTy, int8PtrTy},
      false 
  );
  TheModule->getOrInsertFunction("set_scope_scope", set_scope_scopeTy);
  
  FunctionType *set_scope_previous_scopeTy = FunctionType::get(
      int8PtrTy,
      {int8PtrTy, int8PtrTy},
      false 
  );
  TheModule->getOrInsertFunction("set_scope_previous_scope", set_scope_previous_scopeTy);



  FunctionType *set_scope_has_gradTy = FunctionType::get(
      int8PtrTy,
      {int8PtrTy, Type::getInt32Ty(*TheContext)},
      false 
  );
  TheModule->getOrInsertFunction("set_scope_has_grad", set_scope_has_gradTy);

  FunctionType *get_scope_first_argTy = FunctionType::get(
      int8PtrTy,
      {int8PtrTy},
      false 
  );
  TheModule->getOrInsertFunction("get_scope_first_arg", get_scope_first_argTy);

  FunctionType *get_scope_scopeTy = FunctionType::get(
      int8PtrTy,
      {int8PtrTy},
      false 
  );
  TheModule->getOrInsertFunction("get_scope_scope", get_scope_scopeTy);

  FunctionType *get_scope_previous_scopeTy = FunctionType::get(
      int8PtrTy,
      {int8PtrTy},
      false 
  );
  TheModule->getOrInsertFunction("get_scope_previous_scope", get_scope_previous_scopeTy);

  FunctionType *get_scope_thread_idTy = FunctionType::get(
      Type::getInt32Ty(*TheContext),
      {int8PtrTy},
      false 
  );
  TheModule->getOrInsertFunction("get_scope_thread_id", get_scope_thread_idTy);

  FunctionType *get_scope_has_gradTy = FunctionType::get(
      Type::getInt32Ty(*TheContext),
      {int8PtrTy},
      false 
  );
  TheModule->getOrInsertFunction("get_scope_has_grad", get_scope_has_gradTy);

  
  FunctionType *print_scopeTy = FunctionType::get(
      int8PtrTy,
      {int8PtrTy},
      false 
  );
  TheModule->getOrInsertFunction("scope_struct_Print", print_scopeTy);

  FunctionType *scope_struct_CopyTy = FunctionType::get(
      int8PtrTy,
      {int8PtrTy},
      false 
  );
  TheModule->getOrInsertFunction("scope_struct_Copy", scope_struct_CopyTy);

  FunctionType *scope_struct_OverwriteTy = FunctionType::get(
      int8PtrTy,
      {int8PtrTy, int8PtrTy},
      false 
  );
  TheModule->getOrInsertFunction("scope_struct_Overwrite", scope_struct_OverwriteTy);

  FunctionType *scope_struct_DiveTy = FunctionType::get(
      int8PtrTy,
      {int8PtrTy},
      false 
  );
  TheModule->getOrInsertFunction("scope_struct_Dive", scope_struct_DiveTy);
 
  //
  FunctionType *print_randomsTy = FunctionType::get(
      Type::getFloatTy(*TheContext),
      {Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext)},
      false 
  );
  TheModule->getOrInsertFunction("print_randoms", print_randomsTy);
  

  FunctionType *scope_struct_Save_for_AsyncTy = FunctionType::get(
      int8PtrTy,
      {int8PtrTy, int8PtrTy},
      false 
  );
  TheModule->getOrInsertFunction("scope_struct_Save_for_Async", scope_struct_Save_for_AsyncTy);


  FunctionType *scope_struct_Alloc_MarkSeepTy = FunctionType::get(
      int8PtrTy,
      {int8PtrTy},
      false 
  );
  TheModule->getOrInsertFunction("scope_struct_Alloc_MarkSweepMap", scope_struct_Alloc_MarkSeepTy);

  FunctionType *scope_struct_Copy_MarkSeepTy = FunctionType::get(
      int8PtrTy,
      {int8PtrTy, int8PtrTy},
      false 
  );
  TheModule->getOrInsertFunction("scope_struct_Copy_MarkSweepMap", scope_struct_Copy_MarkSeepTy);

  FunctionType *scope_struct_Clean_ScopeTy = FunctionType::get(
      int8PtrTy,
      {int8PtrTy},
      false 
  );
  TheModule->getOrInsertFunction("scope_struct_Clean_Scope", scope_struct_Clean_ScopeTy);

  FunctionType *scope_struct_DeleteTy = FunctionType::get(
      int8PtrTy,
      {int8PtrTy},
      false 
  );
  TheModule->getOrInsertFunction("scope_struct_Delete", scope_struct_DeleteTy);

  FunctionType *scope_struct_Load_for_AsyncTy = FunctionType::get(
      int8PtrTy,
      {int8PtrTy},
      false 
  );
  TheModule->getOrInsertFunction("scope_struct_Load_for_Async", scope_struct_Load_for_AsyncTy);

  // 
  FunctionType *randintTy = FunctionType::get(
      Type::getFloatTy(*TheContext),
      {Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext)},
      false 
  );
  TheModule->getOrInsertFunction("randint", randintTy);

  FunctionType *scope_struct_Get_Async_Scope = FunctionType::get(
    int8PtrTy,
    {int8PtrTy, Type::getInt32Ty(*TheContext), Type::getInt32Ty(*TheContext)},
    false
  );
  TheModule->getOrInsertFunction("scope_struct_Get_Async_Scope", scope_struct_Get_Async_Scope);


  //
  FunctionType *CreateLSTMOnDemandTy = FunctionType::get(
      Type::getFloatTy(*TheContext),
      {int8PtrTy,
       int8PtrTy,
       Type::getFloatTy(*TheContext),
       Type::getFloatTy(*TheContext)},
      false
  );
  TheModule->getOrInsertFunction("CreateLSTMOnDemand", CreateLSTMOnDemandTy);


  //
  FunctionType *CreateEmbeddingOnDemandTy = FunctionType::get(
      Type::getFloatTy(*TheContext),
      {int8PtrTy,
       int8PtrTy,
       Type::getFloatTy(*TheContext),
       Type::getFloatTy(*TheContext)},
      false
  );
  TheModule->getOrInsertFunction("CreateEmbeddingOnDemand", CreateEmbeddingOnDemandTy);




  //
  FunctionType *CreateMHSAOnDemandTy = FunctionType::get(
      Type::getFloatTy(*TheContext),
      {int8PtrTy,
       int8PtrTy,
       Type::getFloatTy(*TheContext),
       Type::getFloatTy(*TheContext),
       Type::getFloatTy(*TheContext),
       int8PtrTy},
      false
  );
  TheModule->getOrInsertFunction("CreateMHSAOnDemand", CreateMHSAOnDemandTy);




  //
  FunctionType *CreateBN2dReluOnDemandTy = FunctionType::get(
      Type::getFloatTy(*TheContext),
      {int8PtrTy,
       Type::getFloatTy(*TheContext)},
      false
  );
  TheModule->getOrInsertFunction("CreateBN2dReluOnDemand", CreateBN2dReluOnDemandTy);


  //
  FunctionType *CreateReluOnDemandTy = FunctionType::get(
      Type::getFloatTy(*TheContext),
      {int8PtrTy},
      false
  );
  TheModule->getOrInsertFunction("CreateReluOnDemand", CreateReluOnDemandTy);




  TheModule->getOrInsertFunction(
    "posix_memalign",
    FunctionType::get(Type::getInt32Ty(*TheContext),
                      {int8PtrTy->getPointerTo(), // void**
                       Type::getInt64Ty(*TheContext),                   // alignment
                       Type::getInt64Ty(*TheContext)},                  // size
                      false));
  






  

  //
  FunctionType *cpuTy = FunctionType::get(
      Type::getFloatTy(*TheContext),
      {int8PtrTy, int8PtrTy},
      false 
  );
  TheModule->getOrInsertFunction("cpu", cpuTy);
  

  //
  FunctionType *cpu_idxTy = FunctionType::get(
      Type::getFloatTy(*TheContext),
      {int8PtrTy, int8PtrTy, Type::getFloatTy(*TheContext)}, 
      false 
  );
  TheModule->getOrInsertFunction("cpu_idx", cpu_idxTy);


  //
  FunctionType *exitTy = FunctionType::get(
      Type::getFloatTy(*TheContext),
      {},
      false 
  );
  TheModule->getOrInsertFunction("_exit", exitTy);

  //
  FunctionType *printTTy = FunctionType::get(
      Type::getFloatTy(*TheContext),
      {int8PtrTy, int8PtrTy},
      false 
  );
  TheModule->getOrInsertFunction("PrintTensor", printTTy);
  
  
  //
  FunctionType *randu_likeTy = FunctionType::get(
      int8PtrTy,
      {int8PtrTy, int8PtrTy},
      false 
  );
  TheModule->getOrInsertFunction("randu_like", randu_likeTy);

  
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

/// top ::= definition | external | expression | ';'
static void MainLoop() {
  while (true) {
    //if (CurTok!=tok_space)
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
        // std::cout << "TOP LEVEL WITH " << CurTok << "/" << ReverseToken(CurTok) << "/" << NumVal << ".\n";
        HandleTopLevelExpression();
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

int main() {



  
  
  



  // Install standard binary operators.
  // 1 is lowest precedence.
  BinopPrecedence[tok_space] = 1;
  BinopPrecedence['='] = 4;
  BinopPrecedence['!'] = 9;
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





  set_functions_return_type();
  set_functions_args_type();
  set_user_functions();
  vararg_methods = {"tensor_view", "tensor_sum", "tensor_mean", "mean_tensor" ,"tensor_prod", "tensor_tmax", "tensor_argmax", "tensor_load_bin_idx"};


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
  native_methods = {"split", "split_idx", "float_vec_first_nonzero", "append", "float_vec_print", "str_vec_print"};
  native_methods = concat_str_vec(native_methods, return_tensor_methods);
  native_methods = concat_str_vec(native_methods, user_cpp_functions);

  return_string_fn = {"to_string", "cat_str_float"};


  native_functions = {"ShuffleStrVec", "gload_img", "wload_img", "silent_sleep", "__slee_p_",
                      "LenStrVec", "zeros_vec", "ones_vec", "start_timer", "end_timer",
                      "_glob_b_", "print", "cross_entropy", "backprop", "AdamW", "SGD",
                      "load_preprocess_img", "max", "min", "unbug", "is_null",
                      "cpu_idx", "eval", "train", "OneCycleLR", "CosineLR", "wload_img_resize",
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

  ops_type_return = {{"tensor_tensor", "tensor"}, {"float_float", "float"}, {"str_str", "str"}, {"str_float", "str"},
                     {"float_str", "str"}, {"int_int", "int"}, {"int_float", "float"}, {"float_int", "float"}, {"str_int", "str"}, {"int_str", "str"},
                     {"tensor_float", "tensor"}, {"pinned_tensor_pinned_tensor", "pinned_tensor"},
                     {"pinned_tensor_tensor", "pinned_tensor"}, {"pinned_tensor_float", "pinned_tensor"},
                     {"object_object", "object"}, {"str_object", "object"},
                     {"tensor_int", "tensor"}, {"int_tensor", "tensor"}};
                     

  op_map = {{'*', "mult"}, {'@', "mma"},  {'+', "add"}, {'-', "sub"}, {'/', "div"}, {'<', "minor"}, {'>', "higher"}, {tok_equal, "equal"},
            {tok_diff, "different"}, {tok_higher_eq, "higher_eq"}, {tok_minor_eq, "minor_eq"}, {'%', "mod"}, {'=', "attr"},
            {77, "error"}};

  for (auto pair : op_map)
    op_map_names.push_back(pair.second);


  
  notators_str = {"bias", "fp32", "fp16", "causal"};


  // Prime the first token.
  
  getNextToken();

  TheJIT = ExitOnErr(KaleidoscopeJIT::Create());
  InitializeModule();


  MainLoop();

  return 0;
}