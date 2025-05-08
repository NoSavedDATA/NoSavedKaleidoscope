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
#include <omp.h>
#include <iostream>



// Cuda
#include <cuda_runtime.h>
#include <cublasLt.h>
#include <cublas_v2.h>
#include <curand_kernel.h>
#include <cuda_fp16.h>
#include <cudnn.h>
#include <mma.h>
// #include <cute/tensor.hpp>


// #include "include/cutlass/gemm/device/gemm.h"
// #include "include/cutlass/cutlass.h"

#include "src/include.h"



float TERMINATE_VARARG = -40370000000.0f;



#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16


cudaDeviceProp deviceProp;

int WARP_SIZE;

int THREADS_PER_BLOCK = deviceProp.maxThreadsPerMultiProcessor == 1536 ? 768 : 1024;

const int TILE_SIZE = (int)floorf(sqrtf((float)THREADS_PER_BLOCK)); 
const int TILE_SIZE_SQ = TILE_SIZE*TILE_SIZE;








using namespace llvm;
using namespace llvm::orc;
using namespace nvcuda;

#if __CUDA_ARCH__ == 800 || __CUDA_ARCH__ >= 900
#define MAX_1024_THREADS_BLOCKS 2
#else
#define MAX_1024_THREADS_BLOCKS 1
#endif




int ASYNC_LOADER_THREADS = 6;
const int num_parallel_streams=32; //global of src/mma/tensor_struc.h
CudaStreams *parallel_streams[num_parallel_streams];
cudaEvent_t parallel_events[num_parallel_streams];
std::vector<cudaEvent_t> Registered_Events;
int open_streams[num_parallel_streams];
CudaStreams *main_stream, *backward_stream;
std::map<int, cudaStream_t> ThreadsStream;
std::vector<int> leaf_ops, loss_ops, gradless_ops, activation_ops, preprocessing_ops, tensor_scalar_ops, custom_ops, weightless_ops;
int nn_mode=training_mode;

std::map<std::string, int> NotatorsMap = {
  {"bias", bias},
  {"fp32", fp32},
  {"fp16", fp16},
  {"causal", causal},
};

bool ShallCodegen = true;


// Tensors
std::map<std::string, DT_tensor *> NamedTensorsT;
std::map<std::string, float *> NamedPinnedTensors;
std::map<std::string, std::vector<float>> NamedDims;
std::vector<DT_tensor> TensorsToDelete;


LCG rng(generate_custom_seed());



std::vector<std::string> rds;


pthread_mutex_t mutex, clean_scope_mutex, char_pool_mutex, vocab_mutex, random_seed_mutex, aux_mutex;

  // Error Colors
// \033[0m default
// \033[31m red
// \033[33m yellow
// \033[95m purple






// Tensor related
std::vector<std::string> return_tensor_functions, return_tensor_methods, return_tensor_fn, native_modules,
return_pinned_methods, vararg_methods, string_methods, native_methods, native_functions, native_fn, tensor_inits,
return_string_fn, threaded_tensor_functions, require_scope_functions, notators_str;


std::map<std::string, std::string> reverse_ops;








//global
std::vector<std::string> objectVars;
std::vector<std::string> globalVars;
std::map<std::string, std::string> functionVars;
std::map<std::string, std::string> floatFunctions;
std::map<std::string, std::string> stringMethods;
std::map<std::string, pthread_mutex_t *> lockVars;



//global
std::vector<std::string> Classes;
std::map<std::string, std::string> Object_toClass;
std::map<std::string, std::string> Object_toClassVec;



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
std::map<std::string, std::vector<float>> ClassFloatVecs;
std::map<std::string, float> NamedClassValues;
std::map<std::string, std::string> NamedObjects;
std::map<std::string, std::vector<std::pair<std::string, std::string>>> ScopeVarsToClean;
std::map<std::string, char *> ScopeNamesToClean;
std::map<int, std::map<std::string, std::vector<std::string>>> ThreadedScopeTensorsToClean;


// Aux to not lose pointers
std::map<std::string, std::vector<char *>> StrVecAuxHash;





// File Handling
std::vector<char *> glob_str_files;




// Handle Class self with phantom argument









extern "C" float Add(float value, float v2)
{
  return value + v2; 
}










// template<int WMMA_T, int X_WARPS, int Y_WARPS>
// __global__ void wmma_cutlass(const float *x, const float *w,
//                       float *out, const int B, const int C, const int OC) {

//   int tid = threadIdx.y * blockDim.x + threadIdx.x;
//   int laneId = tid % warpSize;
//   int mw = laneId / WMMA_T;
//   int ml = laneId % WMMA_T;

//   int warp_y = threadIdx.y;
//   int warp_x = (threadIdx.x / 32);


//   const uint32_t warpX{(blockIdx.x * blockDim.x + threadIdx.x) / warpSize};  // OC
//   const uint32_t warpY{blockIdx.y * blockDim.y + threadIdx.y};               // B

//   // warpX = (oc*X_WARPS + warp_x)


//   using ColumnMajor = cutlass::layout::ColumnMajor;
//   using RowMajor = cutlass::layout::RowMajor;

//   using Mma = cutlass::gemm::warp::DefaultMmaTensorOp<
//     cutlass::gemm::GemmShape<16,16,16>,
//     cutlass::gemm::GemmShape<16,16,16>,
//     cute::half_t, RowMajor,
//     cute::half_t, ColumnMajor,
//     float, RowMajor
//   >;


//   extern __shared__ float smem[];
//   float *out_smem = smem;
//   __half *hsmem = reinterpret_cast<__half*>(smem + Y_WARPS*WMMA_T*(X_WARPS*WMMA_T));

//   __half *x_smem     = hsmem;
//   __half *w_smem     = hsmem + Y_WARPS*WMMA_T*(WMMA_T);

//   typename Mma::IteratorA iter_A({x_smem, C}, tid);
//   typename Mma::IteratorB iter_B({w_smem, C}, tid);
//   typename Mma::IteratorC iter_C({out_smem, OC}, tid);

//   typename Mma::FragmentA x_frag;
//   typename Mma::FragmentB w_frag;
//   typename Mma::FragmentC y_frag;
  
//   Mma mma;
  
//   y_frag.clear();

// #pragma unroll
//   for (int tile=0; tile<C; tile+=WMMA_T)
//   {
//     iter_A.load(x_frag);
//     iter_B.load(w_frag);

//     ++iter_A, ++iter_B;
    
//     mma(y_frag, x_frag, w_frag, y_frag);
//   }


//   if ((warpY*WMMA_T)<B && (warpX*WMMA_T)<OC && (warp_y*WMMA_T)<B && (warp_x*WMMA_T)<OC)
//   { 

//     // float *_out = out_smem + warp_y*WMMA_T*(X_WARPS*WMMA_T) + warp_x*WMMA_T;
//     // wmma::store_matrix_sync(_out, y_frag, X_WARPS*WMMA_T, wmma::mem_row_major);
//     iter_C.store(y_frag);

    
// #pragma unroll
//     for (int tile=0; tile<std::ceil((WMMA_T*WMMA_T)/(float)warpSize); ++tile)
//     {
//       int tile_idx = tile*warpSize + laneId;

//       int row = tile_idx / WMMA_T;
//       int col = tile_idx % WMMA_T;


//       if((warpY*WMMA_T+row)<B  &&  (warpX*WMMA_T+col)<OC && row<WMMA_T)
//         out[(warpY*WMMA_T+row)*OC + warpX*WMMA_T+col] = _out[row*(X_WARPS*WMMA_T)+col];

//     }
//   }
// }








extern "C" float printtt(int thread_id, DT_tensor tensor)
{
  char* tensorName = new char[tensor.name.size() + 1]; // Allocate memory for the C-style string
  std::strcpy(tensorName, tensor.name.c_str()); // Copy the string

  std::cout << "a" << ".\n";

//   PrintTensor(thread_id, tensorName);

  delete[] tensorName;
  return 0;
}

































































extern "C" float eval(Scope_Struct *scope_struct)
{
  std::cout << "\n\n\nSETTING NN MODE TO EVAL" << "\n\n";
  
  /*
  for(int i=0; i<TensorPool.size(); i++)
  {
    for(int j=0; j<TensorPool[i].size();j++)
      cudaCheck(cudaFree(TensorPool[i][j]));
    
    TensorPool[i].clear();
  }
  TensorPool.clear();
  */
   
  for (auto& pair : NamedParamGrads)
  {
    std::cout << "Erasing gradient memory of: " << pair.first << "\n";
    cudaCheck(cudaFree(pair.second));
    NamedParamGrads.erase(pair.first);
  }

  NamedParamGrads.clear();


  //std::cout << "\n\nALL TENSORS ARE" << "\n\n";
  //for (auto& pair : NamedTensorsT)
  //  std::cout << pair.first << "\n"; 


  for (auto &pair: optimizer->NamedV)
    cudaCheck(cudaFree(pair.second));
    
  for (auto &pair: optimizer->NamedM)
    cudaCheck(cudaFree(pair.second));


  /*
  for (auto& pair : NamedTensorsT)
  {
    Tensor *tensor = pair.second;

    if (tensor->is_pinned)
    {
      std::cout << "Erasing tensor: " << pair.first << ", is pinned: " << tensor->is_pinned << "\n";
      tensor->Sync();
      cudaCheck(cudaFree(tensor->tensor_ptr));

      //delete[] tensor->cpu_tensor_ptr;
    }
  }
  */
  
  

  nn_mode = eval_mode;

  std::cout << "\n\n\n";
  return 0;
}

extern "C" float train(Scope_Struct *scope_struct)
{
  std::cout << "SETTING NN MODE TO TRAIN" << "\n\n";
  nn_mode = training_mode;
  return 0;
}




















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
  Value *first_arg, *scope_string, *previous_scope, *thread_id, *has_grad, *scope_struct, *incoming_scope_struct;

  
  scope_struct = callret("scope_struct_Create", {});


  thread_id = ConstantInt::get(Type::getInt32Ty(*TheContext), 0);
  has_grad  = ConstantInt::get(Type::getInt32Ty(*TheContext), 1);

 
  
  call("set_scope_thread_id", {scope_struct, thread_id});
  call("set_scope_has_grad", {scope_struct, has_grad});
  





  std::cout << "\033[32mExecuting function: " << function_name << " \033[0m\n";

  NamedValues.clear();

  bool has_self = false; 
  bool has_scope = false;
  bool has_previous_scope = false;

  
  
  call("scope_struct_Alloc_MarkSweepMap", {scope_struct}); 


  p2t("FunctionAST start function args.");

  float val;
  int i = 0;
  for (auto &Arg : TheFunction->args()) {
    // Create an alloca for this variable.
    
    
    std::string arg_name = Arg.getName().str();
    //std::cout << "FUNCTION ARG IS: " << arg_name  << "\n";

    std::string __print = "FunctionAST FUNCTION ALLOCA OF " + std::string(Arg.getName()) + " ";

    p2t(__print);


    p2t("FunctionAST Got arg "+arg_name+" for function "+function_name);
    // Default args
    if (arg_name == "scope_struct")
    {
        p2t("-------------------------------------------=============-----------------===========--------FunctionAST COPY SCOPE STRUCT");
        
        //   scope_struct = callret("scope_struct_Dive", {&Arg});

        scope_struct = callret("scope_struct_Overwrite", {scope_struct, &Arg});

        first_arg = callret("get_scope_first_arg", {scope_struct}); 
        scope_string = callret("get_scope_scope", {scope_struct}); 
        previous_scope = callret("get_scope_previous_scope", {scope_struct}); 
        thread_id = callret("get_scope_thread_id", {scope_struct}); 
        has_grad = callret("get_scope_has_grad", {scope_struct}); 
      
        
    } else { 
        std::string type = "";
        if (typeVars.find(arg_name) != typeVars.end())
            type = typeVars[arg_name];
        std::cout << "------------------------------------TYPE OF " << arg_name << " IS " << type << ".\n";

        // Coder args
        if (type!="tensor")
        {
            Value *var_name = global_str(arg_name);
            var_name = callret("ConcatStr", {scope_string, var_name});

            call(type+"_Store", {var_name, &Arg, scope_struct});
            call(type+"_MarkToSweep", {scope_struct, var_name, &Arg});
        } else {
            call("CopyArgTensor",
                            {&Arg,
                            global_str(arg_name),
                            previous_scope,
                            scope_string,
                            thread_id});
        }
    }
    // else if (type!="tensor")
    // {
    //   AllocaInst *Alloca = CreateEntryBlockAlloca(TheFunction, Arg.getName());
    //   Builder->CreateStore(&Arg, Alloca);
    //   //Builder->CreateStore(Builder->CreateLoad(Type::getFloatTy(*TheContext), &Arg), Alloca);

    //   NamedValues[std::string(Arg.getName())] = Alloca;

    // }
  }
  

  p2t("FunctionAST");
  // call("scope_struct_Print", {scope_struct});




  Value *RetVal;
  for (auto &body : Body)
  {
    std::string pre = "\n=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=--=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=\n" + std::string("FunctionAST Body codegen pre of: ") + typeid(*body).name();
    p2t(pre);
    RetVal = body->codegen(scope_struct);
    p2t("FunctionAST Body codegen post");
  }


  call("scope_struct_Clean_Scope", {scope_struct}); 


  
  
  

  p2t("FunctionAST return");

  if (RetVal) {
    // Finish off the function.
    
    
    p2t("FunctionAST CreateRet");
    Builder->CreateRet(RetVal);
    

    p2t("FunctionAST verify");
    // Validate the generated code, checking for consistency.
    verifyFunction(*TheFunction);


    p2t("FunctionAST verified");
    // Validate the generated code, checking for consistency.

    return TheFunction;
  }

  p2t("FunctionAST returned");

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
  FunctionType *viewTy = FunctionType::get(
      int8PtrTy,
      {int8PtrTy,int8PtrTy,Type::getFloatTy(*TheContext),Type::getFloatTy(*TheContext),Type::getFloatTy(*TheContext),Type::getFloatTy(*TheContext),Type::getFloatTy(*TheContext),Type::getFloatTy(*TheContext),Type::getFloatTy(*TheContext),Type::getFloatTy(*TheContext)},
      true // vararg
  );
  TheModule->getOrInsertFunction("tensor_view", viewTy);

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
  FunctionType *NewVecToTensorTy = FunctionType::get(
      int8PtrTy,
      {int8PtrTy, Type::getFloatTy(*TheContext),Type::getFloatTy(*TheContext),Type::getFloatTy(*TheContext),Type::getFloatTy(*TheContext),Type::getFloatTy(*TheContext),Type::getFloatTy(*TheContext),Type::getFloatTy(*TheContext),Type::getFloatTy(*TheContext)},
      true // vararg
  );
  TheModule->getOrInsertFunction("NewVecToTensor", NewVecToTensorTy);
  

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
  


  auto pthreadPtr = Type::getInt8Ty(*GlobalContext)->getPointerTo();
  auto pthreadPtrTy = pthreadPtr->getPointerTo();

  // (void *) fn (void * arg)
  FunctionType *funVoidPtrint8PtrTy = FunctionType::get(
    int8PtrTy, {int8PtrTy},
    false);
  // int pthread_create(pthread_t * thread, const pthread_attr_t * attr,
  //                  void * (*start_routine)(void *), void * arg)
  // using void * in place of pthread_attr_t *
  FunctionType *pthreadCreateTy = FunctionType::get(
                                      Type::getVoidTy(*TheContext),
                                      {pthreadPtrTy,
                                       int8PtrTy,
                                       (funVoidPtrint8PtrTy)->getPointerTo(),
                                       int8PtrTy},
                                      false
                                    );
  TheModule->getOrInsertFunction("pthread_create_aux", pthreadCreateTy);


  // int pthread_join(pthread_t thread, void **value_ptr)
  FunctionType *pthreadJoinTy = FunctionType::get(
    Type::getVoidTy(*TheContext),
    {pthreadPtr},
    false);
  TheModule->getOrInsertFunction("pthread_join_aux", pthreadJoinTy);

  

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
  FunctionType * ConcatScopeAtCallExprTy = FunctionType::get(
      int8PtrTy,
      {int8PtrTy, int8PtrTy},
      false 
  );
  TheModule->getOrInsertFunction("ConcatScopeAtCallExpr", ConcatScopeAtCallExprTy);

  
  //
  FunctionType * AddToScopeCleanListTy = FunctionType::get(
      Type::getVoidTy(*TheContext),
      {int8PtrTy, int8PtrTy},
      false 
  );
  TheModule->getOrInsertFunction("AddToScopeCleanList", AddToScopeCleanListTy);

  
  //
  FunctionType * AddFloatToScopeCleanListTy = FunctionType::get(
      Type::getVoidTy(*TheContext),
      {int8PtrTy, int8PtrTy},
      false 
  );
  TheModule->getOrInsertFunction("AddFloatToScopeCleanList", AddFloatToScopeCleanListTy);

  
  //
  FunctionType *CleanScopeVarsTy = FunctionType::get(
      Type::getVoidTy(*TheContext),
      {int8PtrTy, Type::getInt32Ty(*TheContext)},
      false 
  );
  TheModule->getOrInsertFunction("CleanScopeVars", CleanScopeVarsTy);
  

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


  FunctionType *set_scope_thread_idTy = FunctionType::get(
      int8PtrTy,
      {int8PtrTy, Type::getInt32Ty(*TheContext)},
      false 
  );
  TheModule->getOrInsertFunction("set_scope_thread_id", set_scope_thread_idTy);

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




  //
  FunctionType *CopyArgTensorTy = FunctionType::get(
      Type::getFloatTy(*TheContext),
      {int8PtrTy, int8PtrTy, int8PtrTy, int8PtrTy, Type::getInt32Ty(*TheContext)}, 
      false 
  );
  TheModule->getOrInsertFunction("CopyArgTensor", CopyArgTensorTy);

  
  //
  FunctionType *RemoveTensorScopeTy = FunctionType::get(
      Type::getFloatTy(*TheContext),
      {int8PtrTy, int8PtrTy, int8PtrTy, Type::getInt32Ty(*TheContext)},
      false 
  );
  TheModule->getOrInsertFunction("RemoveTensorScope", RemoveTensorScopeTy);


  //
  FunctionType *RemoveTensorScopeAttrOnIndexTy = FunctionType::get(
      Type::getFloatTy(*TheContext),
      {int8PtrTy,
       int8PtrTy,
       int8PtrTy,
       int8PtrTy,
       Type::getFloatTy(*TheContext), Type::getInt32Ty(*TheContext)},
      false 
  );
  TheModule->getOrInsertFunction("RemoveTensorScopeAttrOnIndex", RemoveTensorScopeAttrOnIndexTy);




  FunctionType *AttrTensorNoFreeTy = FunctionType::get(
      Type::getFloatTy(*TheContext),
      {int8PtrTy, floatPtrTy, int8PtrTy, Type::getInt32Ty(*TheContext), Type::getInt32Ty(*TheContext)},
      false 
  );
  TheModule->getOrInsertFunction("AttrTensorNoFree", AttrTensorNoFreeTy);
  

  //
  FunctionType *AttrTensorOnIdxTy = FunctionType::get(
      Type::getFloatTy(*TheContext),
      {int8PtrTy,
       int8PtrTy,
       Type::getFloatTy(*TheContext),
       Type::getInt32Ty(*TheContext)}, 
      false 
  );
  TheModule->getOrInsertFunction("AttrTensorOnIdx", AttrTensorOnIdxTy);
  

  //
  FunctionType *AttrTensorOnIdxTensorTy = FunctionType::get(
      Type::getFloatTy(*TheContext),
      {int8PtrTy, int8PtrTy, int8PtrTy, Type::getInt32Ty(*TheContext)},
      false 
  );
  TheModule->getOrInsertFunction("AttrTensorOnIdxTensor", AttrTensorOnIdxTensorTy);
  

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
    report_fatal_error("Não foi possível compilar a função JIT de forma lazy");
}




static void HandleClass() { ParseClass(); }

static void HandleDefinition() {
  
  if (auto FnAST = ParseDefinition()) {

    FunctionProtos[FnAST->getProto().getName()] =
      std::make_unique<PrototypeAST>(FnAST->getProto());

    ExitOnErr(TheJIT->addAST(std::move(FnAST)));
  } else {
    // Skip token for error recovery.
    getNextToken();
  }
}

static void HandleExtern() {
  if (auto ProtoAST = ParseExtern()) {
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
    
    fprintf(stderr, "%.2f\n", fp);

    // Delete the anonymous expression module from the JIT.
    ExitOnErr(RT->remove());    
}



static void HandleTopLevelExpression() {
  // Evaluate a top-level expression into an anonymous function.
  
  if (std::unique_ptr<FunctionAST> FnAST = ParseTopLevelExpr()) {
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
      case tok_extern:
        HandleExtern();
        break;
      default:
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
  int deviceIdx = 0;
  cudaCheck(cudaSetDevice(deviceIdx));
  cudaGetDeviceProperties(&deviceProp, deviceIdx);

  std::cout << "CuDNN Version: " << CUDNN_MAJOR << "." << CUDNN_MINOR << "." << CUDNN_PATCHLEVEL << std::endl;
  printf("Device %d: %s\n", deviceIdx, deviceProp.name);
  std::cout << "Device Max Compute Capability (SM): " << deviceProp.major << "." << deviceProp.minor << std::endl;


    
  cudaDeviceGetAttribute(&WARP_SIZE, cudaDevAttrWarpSize, 0); 
  cublasCheck(cublasCreate(&cublas_handle));
  cublasCheck(cublasLtCreate(&cublaslt_handle));


  int enable_tf32 = deviceProp.major >= 8 ? 1 : 0;
  printf("enable_tf32: %d\n", enable_tf32);
  
  cublas_compute_type = enable_tf32 ? CUBLAS_COMPUTE_32F_FAST_TF32 : CUBLAS_COMPUTE_32F;
  cublasMath_t cublas_math_mode = enable_tf32 ? CUBLAS_TF32_TENSOR_OP_MATH : CUBLAS_DEFAULT_MATH;
  cublasCheck(cublasSetMathMode(cublas_handle, cublas_math_mode));
  // setup the (global) cuBLASLt workspace
  cudaCheck(cudaMalloc(&cublaslt_workspace, cublaslt_workspace_size));
  
  cudnnCreate(&cudnn);

  std::cout << "Tile size is: " << TILE_SIZE << ".\n\n";


  // Create Global CUDA Streams
  for(int i=0; i<num_parallel_streams; i++)
  {
    CudaStreams *cuda_stream = new CudaStreams();

    cudaStreamCreate(&cuda_stream->stream);
    cuda_stream->idx = i;
    parallel_streams[i] = cuda_stream;

    open_streams[i]=1;
  }

  
  // Set the Main Stream
  main_stream = AllocateStream(0);
  cublasSetStream(cublas_handle, main_stream->stream);
  cudnnSetStream(cudnn, main_stream->stream);
  ThreadsStream[0] = main_stream->stream;



  if (pthread_mutex_init(&mutex, NULL) != 0) {
    printf("Mutex initialization failed\n");
    return 1;
  }
  if (pthread_mutex_init(&clean_scope_mutex, NULL) != 0) {
    printf("Mutex initialization failed\n");
    return 1;
  }
  if (pthread_mutex_init(&char_pool_mutex, NULL) != 0) {
    printf("Mutex initialization failed\n");
    return 1;
  }
  if (pthread_mutex_init(&vocab_mutex, NULL) != 0) {
    printf("Mutex initialization failed\n");
    return 1;
  }
  pthread_mutex_init(&random_seed_mutex, NULL);
  pthread_mutex_init(&aux_mutex, NULL);
  
  


  lockVars["mutex"] = &mutex;
  




  leaf_ops = {leaf, tensor_leaf, weight_leaf, bias_leaf};
  activation_ops = {relu_op, gelu_op, softmax_op, tanh_op, sigmoid_op, cudnn_relu_op};
  loss_ops = {cross_entropy_op, cross_entropy_idx_op, mse_op, mse_is_w_op};

  custom_ops = {sigmoid_add2weights_op, embedding_op};

  tensor_scalar_ops = {scalar_add_op, scalar_sub_op, scalar_mult_op, scalar_div_op};

  weightless_ops = {add_op, lgrad_op, dropout_op};
  weightless_ops = concat_int_vec(weightless_ops, tensor_scalar_ops);

  preprocessing_ops = {gpu_op, crop_op, random_horizontal_flip_op, normalize_img_op, jitter_op};
  gradless_ops = {randu_like_op, onehot_op, max_op, argmax_op, equal_op,
                  create_tensor_from_brackets_op, detach_op};
  gradless_ops = concat_int_vec(gradless_ops, preprocessing_ops);


  // Install standard binary operators.
  // 1 is lowest precedence.
  BinopPrecedence[tok_space] = 1;
  BinopPrecedence['='] = 4;
  BinopPrecedence[':'] = 9;
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
  BinopPrecedence['^'] = 50;
  BinopPrecedence['@'] = 60;


  floatFunctions["log"] = "logE";
  floatFunctions["log2"] = "logE2";
  floatFunctions["log2f"] = "logE2f";
  floatFunctions["round"] = "roundE";
  floatFunctions["floor"] = "floorE";

  stringMethods["split"] = "SplitString";






                        


  set_functions_return_type();
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
                     {"float_str", "str"},
                     {"tensor_float", "tensor"}, {"pinned_tensor_pinned_tensor", "pinned_tensor"},
                     {"pinned_tensor_tensor", "pinned_tensor"}, {"pinned_tensor_float", "pinned_tensor"},
                     {"object_object", "object"}, {"str_object", "object"}};
                     

  op_map = {{'*', "mult"}, {'@', "mma"},  {'+', "add"}, {'-', "sub"}, {'/', "div"}, {'<', "minor"}, {'>', "higher"}, {tok_equal, "equal"},
            {tok_diff, "different"}, {'/', "divide"}, {tok_higher_eq, "higher_eq"}, {tok_minor_eq, "minor_eq"}, {'%', "mod"}, {'=', "attr"},
            {77, "error"}};



  tensor_inits = {"binary", "arange", "int", "randu", "zeros", "ones", "xavu", "xavu_relu", "xavu_tanh", "he_normal_relu", "init_gpt", "xavn", "normal"};
  notators_str = {"bias", "fp32", "fp16", "causal"};


  // Prime the first token.
  //fprintf(stderr, "ready> ");
  getNextToken();

  TheJIT = ExitOnErr(KaleidoscopeJIT::Create());
  InitializeModule();

  // Run the main "interpreter loop" now.


  MainLoop();

  return 0;
}
