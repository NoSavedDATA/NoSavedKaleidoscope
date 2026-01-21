#include "llvm/IR/Value.h"
#include "llvm/IR/Verifier.h"

#include <string>
#include <map>
#include <unordered_map>
#include <vector>


#include <filesystem>
#include <fstream>


#include "../common/extension_functions.h"
#include "../data_types/include.h"
#include "../notators/include.h"
#include "../KaleidoscopeJIT.h"
#include "include.h"




using namespace llvm;
namespace fs = std::filesystem;


std::map<std::string, std::map<std::string, AllocaInst *>> function_allocas;
std::map<std::string, std::map<std::string, Value *>> function_values;
std::map<std::string, std::map<Value *, Value *>> function_vecs;
std::map<std::string, std::map<Value *, Value *>> function_vecs_size;
std::map<std::string, std::map<std::string, Value *>> function_pointers;
std::unordered_map<std::string, Function *> async_fn;
std::string current_codegen_function;


std::vector<Value *> thread_pointers;

PointerType *floatPtrTy, *int8PtrTy;
llvm::Type *floatTy, *intTy, *boolTy;

Value *stack_top_value;



Value * VoidPtr_toValue(void *vec)
{
  auto void_ptr_ty = Type::getInt8Ty(*TheContext)->getPointerTo();
  Value* LLVMValue = ConstantInt::get(Type::getInt64Ty(*TheContext), reinterpret_cast<uint64_t>(vec));
  return Builder->CreateIntToPtr(LLVMValue, void_ptr_ty);
}

Value* FloatPtr_toValue(float* vec)
{
    // Get the type for float*
    auto float_ptr_ty = Type::getFloatTy(*TheContext)->getPointerTo();
    
    // Convert the float* to uint64_t and create a constant integer value
    Value* LLVMValue = ConstantInt::get(Type::getInt64Ty(*TheContext), reinterpret_cast<uint64_t>(vec));
    
    // Cast the integer value to float*
    return Builder->CreateIntToPtr(LLVMValue, float_ptr_ty);
}

Value *load_alloca(std::string name, std::string type, std::string from_function) {

    llvm::Type *load_type;
    if(type=="float")
      load_type = Type::getFloatTy(*TheContext);
    else if(type=="int")
      load_type = Type::getInt32Ty(*TheContext);
    else if(type=="bool")
      load_type = Type::getInt1Ty(*TheContext);
    else
      load_type = int8PtrTy;

    AllocaInst *alloca = function_allocas[from_function][name];

    return Builder->CreateLoad(load_type, alloca, name.c_str());
}

Value *float_llvm_hash(Value *float_value) {
    LLVMContext &C = Builder->getContext();
    // Bitcast float -> i32 (exact bit pattern)
    Value *bits = Builder->CreateBitCast(float_value, intTy, "float.bits");
    // bits ^= bits >> 16
    Value *x = bits;
    x = Builder->CreateXor(
        x,
        Builder->CreateLShr(x, const_int(16))
    );
    // bits *= 0x85ebca6b
    x = Builder->CreateMul(
        x,
        const_int(0x85ebca6bU)
    );
    // bits ^= bits >> 13
    x = Builder->CreateXor(
        x,
        Builder->CreateLShr(x, const_int(13))
    );
    // bits *= 0xc2b2ae35
    x = Builder->CreateMul(
        x,
        const_int(0xc2b2ae35U)
    );
    // bits ^= bits >> 16
    x = Builder->CreateXor(
        x,
        Builder->CreateLShr(x, const_int(16))
    );
    return x; // i32
}

Value *str_llvm_hash(Value *str_value, Function *F) {
    Type *i8  = Builder->getInt8Ty();
    Type *i32 = Builder->getInt32Ty();

    Value *initHash = Builder->getInt32(2166136261u); // FNV offset basis
    Value *initPtr  = str_value;

    BasicBlock *preBB  = Builder->GetInsertBlock();
    BasicBlock *loopBB = BasicBlock::Create(*TheContext, "hash.loop", F);
    BasicBlock *exitBB = BasicBlock::Create(*TheContext, "hash.exit", F);
    BasicBlock *bodyBB = BasicBlock::Create(*TheContext, "hash.body", F);

    Builder->CreateBr(loopBB);
    Builder->SetInsertPoint(loopBB);

    PHINode *phiHash = Builder->CreatePHI(i32, 2);
    PHINode *phiPtr  = Builder->CreatePHI(int8PtrTy, 2);

    phiHash->addIncoming(initHash, preBB);
    phiPtr->addIncoming(initPtr,  preBB);

    Value *ch     = Builder->CreateLoad(i8, phiPtr);
    Value *isZero = Builder->CreateICmpEQ(ch, Builder->getInt8(0));
    Builder->CreateCondBr(isZero, exitBB, bodyBB);

    /* ---- body ---- */
    Builder->SetInsertPoint(bodyBB);

    Value *ch32 = Builder->CreateZExt(ch, i32);
    Value *h1   = Builder->CreateXor(phiHash, ch32);
    Value *h2   = Builder->CreateMul(h1, Builder->getInt32(16777619u));
    Value *nextPtr = Builder->CreateGEP(i8, phiPtr, Builder->getInt32(1));

    phiHash->addIncoming(h2, bodyBB);
    phiPtr->addIncoming(nextPtr, bodyBB);

    Builder->CreateBr(loopBB);

    /* ---- exit ---- */
    Builder->SetInsertPoint(exitBB);
    return phiHash; // i32 == unsigned int
}




Function *getFunction(std::string Name) {
  // First, see if the function has already been added to the current module.
  if (auto *F = TheModule->getFunction(Name))
    return F;

  // If not, check whether we can codegen the declaration from some existing
  // prototype.
  auto FI = FunctionProtos.find(Name);
  if (FI != FunctionProtos.end())
    return FI->second->codegen();

  // If no existing prototype exists, return null.
  return nullptr;
}



Value *ClassExprAST::codegen(Value *scope_struct) {

  return const_float(0);
}

Value *IndexExprAST::codegen(Value *scope_struct) {
  
  return const_float(0);
}


Value *Idx_Calc_Codegen(std::string type, Value *vec, std::unique_ptr<IndexExprAST> &idxs, Value *scope_struct)
{
  if (!idxs->IsSlice)
  {
    if (!TheModule->getFunction(type+"_CalculateIdx"))
        return idxs->Idxs[0]->codegen(scope_struct); 
  }
  std::vector<Value *> idxs_values;

  idxs_values.push_back(vec); // e.g, tensor uses its dims as a support to calculcate the index

  for (int i=0; i<idxs->size(); i++) {
    Value *idx = idxs->Idxs[i]->codegen(scope_struct);
    
    if (i==0 && (idxs->Idxs[i]->GetDataTree().Type=="str"))  {// dict query 
      idxs->idx_slice_or_query = "query";
      return idx;
    }
    idxs_values.push_back(idxs->Idxs[i]->codegen(scope_struct));
  }
  // Has Terminate_Vararg inserted from the parser.


  if (!idxs->IsSlice)
  {
    std::string fn = type+"_CalculateIdx";
    Function *F = TheModule->getFunction(fn);
    if (F)
      return callret(fn, idxs_values);
  } else {

    for (int i=0; i<idxs->size(); i++)
      idxs_values.push_back(idxs->Second_Idxs[i]->codegen(scope_struct));

    std::string fn = type+"_CalculateSliceIdx";
    Function *F = TheModule->getFunction(fn);
    if (F)
      return callret(fn, idxs_values);
    else
      return callret("__sliced_idx__", idxs_values);
  }
}


/// CreateEntryBlockAlloca - Create an alloca instruction in the entry block of
/// the function.  This is used for mutable variables etc.
AllocaInst *CreateEntryBlockAlloca(Function *TheFunction,
                                          StringRef VarName, Type *alloca_type) {
  IRBuilder<> TmpB(&TheFunction->getEntryBlock(),
                   TheFunction->getEntryBlock().begin());

  AllocaInst *alloca = TmpB.CreateAlloca(alloca_type, nullptr, VarName);

  return alloca;
}


bool CheckIsEquivalent(std::string LType, std::string RType) {
  if(LType==RType)
    return true;

  if (Equivalent_Types.count(RType)>0)
  for(std::string equivalent : Equivalent_Types[RType])
  {
      if (LType==equivalent)
          return true;
  }
  return false;
}


Value *NumberExprAST::codegen(Value *scope_struct) {
  if (not ShallCodegen)
    return const_float(0.0f);

  std::string msg = "NumberExpr Num has value: " + std::to_string(Val);
  p2t(msg);

  return const_float(Val);
}

Value *IntExprAST::codegen(Value *scope_struct) {
  if (not ShallCodegen)
    return const_int(0);

  std::string msg = "NumberExpr Num has value: " + std::to_string(Val);
  p2t(msg);

  return const_int(Val);
}


Value *BoolExprAST::codegen(Value *scope_struct) {
  if (not ShallCodegen)
    return const_bool(false);
  

  return const_bool(Val);
}

Value *StringExprAST::codegen(Value *scope_struct) {
  if (not ShallCodegen)
    return const_float(0.0f);
  SetName(Val);
  Value *_str = callret("CopyString", {scope_struct, global_str(Val)});
  return _str;
}

Value *NullPtrExprAST::codegen(Value *scope_struct) { 
  // return Constant::getNullValue(int8PtrTy);
  return ConstantPointerNull::get(
            cast<PointerType>(int8PtrTy)
        );
}

// Create Float Var
Value *VarExprAST::codegen(Value *scope_struct) {



  return ConstantFP::get(*TheContext, APFloat(0.0));
}


llvm::Type *get_type_from_str(std::string type)
{
  llvm::Type *llvm_type;
  if (type=="float")
    llvm_type = floatTy;
  else if (type=="int")
    llvm_type = intTy;
  else if (type=="bool")
    llvm_type = boolTy;
  else
    llvm_type = int8PtrTy;
  return llvm_type;
}


bool Check_Is_Compatible_Data_Type(Data_Tree LType, Data_Tree RType, Parser_Struct parser_struct) {

  int differences = LType.Compare(RType);

  if (differences>0)
  {
    LogError(parser_struct.line, "Tried to attribute data of different types");
    std::cout << "Left type:\n   ";
    LType.Print();
    std::cout << "\nRight type:\n   ";
    RType.Print();
    std::cout << "\n\n";
    return false;
  }

  return true;
}


bool Check_Is_Compatible_Type(std::string LType, const std::unique_ptr<ExprAST> &RHS, Parser_Struct parser_struct) {
  std::string RType = RHS->GetType();



  if (LType=="float"&&RType=="int")
    return true;

  if((RType=="nullptr")&&!in_str(LType, {"float", "int", "str", "bool"}))
    return true;
  
  
  if (LType!=RType)
  { 
    // if (dynamic_cast<NestedVectorIdxExprAST *>(RHS.get()) || dynamic_cast<VecIdxExprAST *>(RHS.get()))
    // {
    //   RType = Extract_List_Prefix(RType);
    //   RType = remove_suffix(RType, "_vec");
    // }

    if (LType!=RType)
    {

      LogError(parser_struct.line, "Tried to attribute " + RType + " to " + LType);
      return false;
    }
  }
  return true;
}

inline bool Check_ArgsV_Count(const std::string &Callee, const std::vector<Value *> &ArgsV,
                              Parser_Struct parser_struct, int backend_args=1) {  
  Function *CalleeF;
  CalleeF = getFunction(Callee);
  
  if (!CalleeF)
  {
    std::string _error = "The referenced function "+ Callee +" was not yet declared.";
    LogErrorV(parser_struct.line, _error);
    return false;
  }

  if ((CalleeF->arg_size()) != ArgsV.size() && !in_str(Callee, vararg_methods))
  {
    std::string _error = "Incorrect parameters used on function " + Callee + " call.\n\t    Expected " +  std::to_string(CalleeF->arg_size()-backend_args) + " arguments, got " + std::to_string(ArgsV.size()-backend_args);
    LogErrorV(parser_struct.line, _error);
    return false;
  }

  return true;
}

inline bool Check_Required_Args_Count(const std::string &Callee, int sent_args,
                              Parser_Struct parser_struct, int backend_args=1) {  
  if(Function_Required_Arg_Count.count(Callee)==0||in_str(Callee, vararg_methods))
    return true;

  if (sent_args<Function_Required_Arg_Count[Callee]) {
      // LogError(parser_struct.line, "Sent " + std::to_string(sent_args) + " into function " + Callee + ", but the function requires at least "+\
      //                               std::to_string(Function_Required_Arg_Count[Callee]) + " arguments.");
      return false;
  }

  return true;
}

void Cache_Array(std::string &fn, Value *var) {
    Value *vec_gep = Builder->CreateStructGEP(struct_types["vec"], var, 3);
    Value *vec = Builder->CreateLoad(int8PtrTy, vec_gep);
    function_vecs[fn][var] = vec;
}
inline Value *Load_Array(std::string &function_name, Value *var) {
    Value *vec;
    if (function_vecs[function_name].count(var)>0) {
        vec = function_vecs[function_name][var];
    } else {
        Value *vec_gep = Builder->CreateStructGEP(struct_types["vec"], var, 3);
        vec = Builder->CreateLoad(int8PtrTy, vec_gep);
    }
    return vec;
}

inline void Check_Is_Array_Inbounds(Parser_Struct &parser_struct, Value *var, Value *idx) {
    // return;
    Value *vec_gep = Builder->CreateStructGEP(struct_types["vec"], var, 0);
    Value *size = Builder->CreateLoad(intTy, vec_gep);
    // Value *size = const_int(1000000);

    Value *in_bounds = Builder->CreateICmpSLT(idx, size);

    Function *TheFunction = Builder->GetInsertBlock()->getParent();
    BasicBlock *okBB = BasicBlock::Create(*TheContext, "idx.ok", TheFunction);
    BasicBlock *bad_sizeBB = BasicBlock::Create(*TheContext, "idx.bad_size", TheFunction);

    Builder->CreateCondBr(in_bounds, okBB, bad_sizeBB);

    Builder->SetInsertPoint(bad_sizeBB);
    call("array_bad_idx", {const_int(parser_struct.line), idx, size});
    Builder->CreateUnreachable();

    Builder->SetInsertPoint(okBB);
}



inline Value *swap_scope_obj(Value *scope_struct, Value *obj) {
    StructType *st = struct_types["scope_struct"];
    
    Value *obj_gep = Builder->CreateStructGEP(st, scope_struct, 4);
    Value *previous_obj = Builder->CreateLoad(int8PtrTy, obj_gep);
    Builder->CreateStore(obj, obj_gep);

    return previous_obj;
}

inline void set_scope_obj(Value *scope_struct, Value *obj) {
    StructType *st = struct_types["scope_struct"];
    
    Value *obj_gep = Builder->CreateStructGEP(st, scope_struct, 4);
    Builder->CreateStore(obj, obj_gep);
}

inline Value *get_scope_obj(Value *scope_struct) {
    StructType *st = struct_types["scope_struct"];
    
    Value *obj_gep = Builder->CreateStructGEP(st, scope_struct, 4);
    return Builder->CreateLoad(int8PtrTy, obj_gep);
}

void check_scope_struct_sweep(Function *TheFunction, Value *scope_struct, const Parser_Struct &parser_struct) {
  // return;
  
  Value *GC_ptr = Builder->CreateStructGEP(struct_types["scope_struct"], scope_struct, 5, "GC_ptr_element");
  Value *GC = Builder->CreateLoad(struct_types["GC"]->getPointerTo(), GC_ptr, "GC");

  Value *GC_allocations_ptr = Builder->CreateStructGEP(struct_types["GC"], GC, 0, "GC_allocations_ptr");
  Value *GC_allocations = Builder->CreateLoad(intTy, GC_allocations_ptr, "GC_allocations");

  Value *GC_size_alloc_ptr = Builder->CreateStructGEP(struct_types["GC"], GC, 1, "GC_allocations_ptr");
  Value *GC_size_alloc = Builder->CreateLoad(Type::getInt64Ty(*TheContext), GC_size_alloc_ptr, "GC_allocations");

  // // Compare GC_allocations > 1000
  // Value *cmp_alloc = Builder->CreateICmpSGT(
  //     GC_allocations,
  //     ConstantInt::get(Type::getInt32Ty(*TheContext), 1000),
  //     "cmp_alloc"
  // );
  // Compare GC_size_alloc > 10000
  Value *cmp_size = Builder->CreateICmpUGT(  // unsigned since uint64_t
      GC_size_alloc,
      ConstantInt::get(Type::getInt64Ty(*TheContext), sweep_after_alloc),
      "cmp_size"
  );
  // Value *sweep_cond = Builder->CreateOr(cmp_alloc, cmp_size);
  Value *sweep_cond = cmp_size;

  BasicBlock *SweepThenBB = BasicBlock::Create(*TheContext, "sweep_then", TheFunction);
  BasicBlock *SweepContinueBB = BasicBlock::Create(*TheContext, "sweep_continue", TheFunction);

  Builder->CreateCondBr(sweep_cond, SweepThenBB, SweepContinueBB);

  Builder->SetInsertPoint(SweepThenBB);

  Set_Stack_Top(scope_struct, parser_struct.function_name);
  call("scope_struct_Sweep", {scope_struct});
  Builder->CreateBr(SweepContinueBB);

  Builder->SetInsertPoint(SweepContinueBB);
}



Value *UnkVarExprAST::codegen(Value *scope_struct) {
  if (not ShallCodegen)
    return const_float(0.0f);


  Function *TheFunction = Builder->GetInsertBlock()->getParent();

  // Register all variables and emit their initializer.
  for (unsigned i = 0, e = VarNames.size(); i != e; ++i) {
    const std::string &VarName = VarNames[i].first; 
    ExprAST *Init = VarNames[i].second.get();

    if (dynamic_cast<NullPtrExprAST*>(Init))
      continue; 

    bool is_self = GetSelf();
    bool is_attr = GetIsAttribute();

    Value *initial_value = Init->codegen(scope_struct);

      
    Type = data_typeVars[parser_struct.function_name][VarName].Type;


    if(Init->GetIsMsg()) {
      Value *void_ptr = Constant::getNullValue(int8PtrTy);
      initial_value = callret(Type+"_channel_message", {scope_struct, void_ptr, initial_value});
    }




    if(in_str(Type, primary_data_tokens)&&!(is_self||is_attr))
    { 
      function_values[parser_struct.function_name][VarName] = initial_value;
      continue;
    }


    Value *var_name, *scopeless_name;

    // --- Name Solving --- //
    var_name = callret("CopyString", {scope_struct, global_str(VarName)});
    scopeless_name = callret("CopyString", {scope_struct, var_name});
 

    


    if(is_self)
    {
      int object_ptr_offset = ClassVariables[parser_struct.class_name][VarName]; 
  
      Value *obj = get_scope_obj(scope_struct);
      call("object_ptr_Attribute_object", {obj, const_int(object_ptr_offset), initial_value});

    } else if (is_attr) {
      LogError(parser_struct.line, "Creating attribute in a data expression is not supported.");
    }
    else {
      function_values[parser_struct.function_name][VarName] = initial_value;
      Allocate_On_Pointer_Stack(scope_struct, parser_struct.function_name, VarName, initial_value); 
    }
      

    // call("str_Delete", {var_name});
    // call("str_Delete", {scopeless_name});
  }


  return const_float(0.0f);
}




Value *ListExprAST::codegen(Value *scope_struct) {
}
Value *DictExprAST::codegen(Value *scope_struct) {
}

Value *TupleExprAST::codegen(Value *scope_struct) {
  if (not ShallCodegen)
    return ConstantFP::get(*TheContext, APFloat(0.0f));
  Function *TheFunction = Builder->GetInsertBlock()->getParent();


  // Register all variables and emit their initializer.
  for (unsigned i = 0, e = VarNames.size(); i != e; ++i) {
    const std::string &VarName = VarNames[i].first; 
    ExprAST *Init = VarNames[i].second.get();
        
    
    Value *initial_value = Init->codegen(scope_struct);
    data_type.Print();
    
    function_values[parser_struct.function_name][VarName] = initial_value;
  }

  return ConstantFP::get(*TheContext, APFloat(0.0));
}

Value *Malloc_LLVM_Struct(Value *scope_struct, std::string &struct_name, std::string &type) {
    StructType *st = struct_types[struct_name];
    DataLayout DL(TheModule.get());
    uint64_t size = DL.getTypeAllocSize(st);


    // call malloc
    // Value *rawPtr = callret("malloc", {sizeV});
    Value *rawPtr = callret("allocate_void", {scope_struct, const_int(size), global_str(type)});

    // cast to DT_vec*
    Value *ptr = Builder->CreateBitCast(rawPtr, PointerType::getUnqual(st));
    return ptr;
}

void Set_Stack_Top(Value *scope_struct, std::string fn) {
    Value *stack_top_value_gep = Builder->CreateStructGEP(struct_types["scope_struct"], scope_struct, 3);
    Builder->CreateStore(function_values[fn]["QQ_stack_top"], stack_top_value_gep);
    // Builder->CreateStore(stack_top_value, stack_top_value_gep);
}
Value *Load_Stack_Top(Value *scope_struct) {
    Value *stack_top_value_gep = Builder->CreateStructGEP(struct_types["scope_struct"], scope_struct, 3);
    return Builder->CreateLoad(intTy, stack_top_value_gep);
}

void Allocate_On_Pointer_Stack(Value *scope_struct, std::string function_name, std::string var_name, Value *val) {
    Value *stack_top_value = function_values[function_name]["QQ_stack_top"];
    // pointer to [N x i8*]
    Value *stack_gep = Builder->CreateStructGEP(struct_types["scope_struct"], scope_struct, 2);

    // element pointer: &pointers_stack[i]
    // {0, i} first index the array object, then the element
    Value *void_ptr_gep = Builder->CreateGEP(ArrayType::get(int8PtrTy, ContextStackSize), stack_gep, { const_int(0), stack_top_value });
    Builder->CreateStore(val, void_ptr_gep);
    
    function_pointers[function_name][var_name] = stack_top_value;
    stack_top_value = Builder->CreateAdd(stack_top_value, const_int(1));
    function_values[function_name]["QQ_stack_top"] = stack_top_value;
    // p2t("allocate " + var_name);
}
void Allocate_On_Pointer_Stack_no_metadata(Value *scope_struct, std::string function_name, Value *val) {
    Value *stack_top_value = function_values[function_name]["QQ_stack_top"];
    // Prevents the function_pointer from being used inside another function (invalid Value *).
    
    // pointer to [N x i8*]
    Value *stack_gep = Builder->CreateStructGEP(struct_types["scope_struct"], scope_struct, 2);

    // element pointer: &pointers_stack[i]
    // {0, i} first index the array object, then the element
    Value *void_ptr_gep = Builder->CreateGEP(ArrayType::get(int8PtrTy, ContextStackSize), stack_gep, { const_int(0), stack_top_value });
    Builder->CreateStore(val, void_ptr_gep);
    
    stack_top_value = Builder->CreateAdd(stack_top_value, const_int(1));
    function_values[function_name]["QQ_stack_top"] = stack_top_value;
}

Value *Load_Pointer_Stack(Value *scope_struct, std::string function_name, std::string var_name) {
    Value *stack_gep = Builder->CreateStructGEP(struct_types["scope_struct"], scope_struct, 2);
    Value *stack_idx = function_pointers[function_name][var_name];

    Value *void_ptr_gep = Builder->CreateGEP(ArrayType::get(int8PtrTy, ContextStackSize), stack_gep, { const_int(0), stack_idx });
    return Builder->CreateLoad(int8PtrTy, void_ptr_gep);
}

void Set_Pointer_Stack(Value *scope_struct, std::string function_name, std::string var_name, Value *val) {
    // p2t("SET " + var_name);
    // call("print_void_ptr", {val});
    if (function_pointers[function_name].count(var_name)==0) {
        Allocate_On_Pointer_Stack(scope_struct, function_name, var_name, val);
        return;
    }
    Value *stack_gep = Builder->CreateStructGEP(struct_types["scope_struct"], scope_struct, 2);
    Value *stack_idx = function_pointers[function_name][var_name];

    Value *void_ptr_gep = Builder->CreateGEP(ArrayType::get(int8PtrTy, ContextStackSize), stack_gep, { const_int(0), stack_idx });
    Builder->CreateStore(val, void_ptr_gep);
}


inline std::vector<Value *> Codegen_Argument_List(Parser_Struct parser_struct, std::vector<Value *> ArgsV,
                                                  std::vector<std::unique_ptr<ExprAST>> Args, Value *scope_struct, std::string fn_name,
                                                  bool is_nsk_fn, int arg_offset=1)
{
  // -- Required Arguments -- //
  unsigned i, e;
  for (i = 0, e = Args.size(); i != e; ++i) {
    if (dynamic_cast<PositionalArgExprAST*>(Args[i].get()))
        break;
    
    
    Value *arg = Args[i]->codegen(scope_struct);
 
    Data_Tree data_type = Args[i]->GetDataTree();
    std::string type = data_type.Type;
    

    if (fn_name=="print" && i==0 && type!="str")
    {
      std::string to_string_fn = type+"_to_str";
      arg = callret(to_string_fn, {scope_struct, arg});
    }
    
    int tgt_arg = i + arg_offset;
    Data_Tree expected_data_type = Function_Arg_DataTypes[fn_name][Function_Arg_Names[fn_name][tgt_arg]];
    if (!in_str(fn_name, {"to_int", "to_float", "print"}))
    { 
      if (Function_Arg_Types.count(fn_name)>0)
      {   
        int differences = expected_data_type.Compare(data_type);
        if (differences>0) { 
          LogError(parser_struct.line, "Got an incorrect type for argument " + Function_Arg_Names[fn_name][tgt_arg] + " of function " + fn_name);
          std::cout << "Expected\n   ";
          expected_data_type.Print();
          std::cout << "\nPassed\n   ";
          data_type.Print();
          std::cout << "\n\n";
        } 
      }
    }

    
    if(type=="int"&&expected_data_type.Type=="float")
      arg = Builder->CreateSIToFP(arg, floatTy, "lfp");


    std::string copy_fn = type+"_CopyArg";
    Function *F = TheModule->getFunction(copy_fn);
    if (F&&!is_nsk_fn)
    {
        Value *copied_value = callret(copy_fn,
                        {scope_struct,
                        arg,
                        global_str("-")});   
      arg = copied_value;
    }
    ArgsV.push_back(arg);

    if (!is_nsk_fn && !in_str(type, primary_data_tokens) && \
        (F||dynamic_cast<BinaryExprAST*>(Args[i].get())\
          ||dynamic_cast<UnaryExprAST*>(Args[i].get())\
          ||dynamic_cast<NameableCall*>(Args[i].get()))) {
        // If it creates a new memory address for a high-level fn, store address on the stack.
        
        Allocate_On_Pointer_Stack_no_metadata(scope_struct, parser_struct.function_name, arg);
        Set_Stack_Top(scope_struct, parser_struct.function_name);
    }


    if (!ArgsV.back())
    {
      LogError(parser_struct.line, "Failed to codegen argument of function " + fn_name);
      return {};
    }

  }

  i = i + arg_offset-1;

  // -- Add Default Arguments -- //
  if (Function_Arg_Count.count(fn_name)>0&&!in_str(fn_name, vararg_methods)) {
      int arg_count = Function_Arg_Count[fn_name];

      int c=i+1;
      
      std::vector<std::string> fn_args_name = Function_Arg_Names[fn_name];
      for (; i<Args.size(); ++i, ++c) { // Positional Arguments
        std::cout << "try positional argument: " << i << ".\n";
          auto PosArg = dynamic_cast<PositionalArgExprAST*>(Args[i].get());
          if(!PosArg) {
            LogError(parser_struct.line, "Standard argument followed by positional argument.");
            return std::move(ArgsV);
          }

          std::string arg_name = PosArg->ArgName;
        
          auto it = std::find(fn_args_name.begin(), fn_args_name.end(), arg_name);
          int arg_idx = it-fn_args_name.begin();

          for (; c<arg_idx; ++c) {
              std::string arg_name = Function_Arg_Names[fn_name][c];
              Value *arg_default = ArgsInit[fn_name][arg_name]->codegen(scope_struct);
              ArgsV.push_back(arg_default);
          }

        
          ArgsV.push_back(Args[i]->codegen(scope_struct));
      }

      for (; i<arg_count; ++i) {
          std::string arg_name = fn_args_name[i+1];
          Value *arg_default = ArgsInit[fn_name][arg_name]->codegen(scope_struct);
          ArgsV.push_back(arg_default);
      }
  }


  if (fn_name=="list_append") {
    std::string appended_type = UnmangleVec(Args[Args.size()-1]->GetDataTree());
    ArgsV.push_back(global_str(appended_type));
  }

  return std::move(ArgsV);
}







Value *DataExprAST::codegen(Value *scope_struct) {
  if (not ShallCodegen)
    return ConstantFP::get(*TheContext, APFloat(0.0f));


  Function *TheFunction = Builder->GetInsertBlock()->getParent();

  // Register all variables and emit their initializer.
  for (unsigned i = 0, e = VarNames.size(); i != e; ++i) {
    const std::string &VarName = VarNames[i].first; 
    ExprAST *Init = VarNames[i].second.get();
    
    

    bool is_self = GetSelf();
    bool is_attr = GetIsAttribute();

    Value *initial_value = Init->codegen(scope_struct);
    Data_Tree init_dt = Init->GetDataTree();
    std::string init_type = init_dt.Type;



    Check_Is_Compatible_Data_Type(data_type, init_dt, parser_struct);    


    if (Type=="int"&&init_type=="any")
      initial_value = callret("to_int", {scope_struct, initial_value}); 
    if (Type=="float"&&init_type=="any")
      initial_value = callret("to_float", {scope_struct, initial_value});


    if(Init->GetIsMsg()) {
      Value *void_ptr = Constant::getNullValue(int8PtrTy);
      // initial_value = callret(Type+"_channel_message", {scope_struct, void_ptr, initial_value});
      initial_value = callret("void_channel_message", {scope_struct, void_ptr, initial_value});
    }


    if(in_str(Type, primary_data_tokens)&&!(is_self||is_attr))
    { 
      if (Type=="float"&&Init->GetType()=="int")
        initial_value = Builder->CreateUIToFP(initial_value, floatTy, "int_to_float");
      function_values[parser_struct.function_name][VarName] = initial_value;
      continue;
    }


    Value *var_name, *scopeless_name;

    // --- Name Solving --- //
    // var_name = callret("CopyString", {scope_struct, global_str(VarName)});
    // scopeless_name = callret("CopyString", {scope_struct, var_name});
    // Value *notes_vector;
    // if (HasNotes) {
    //   notes_vector = callret("CreateNotesVector", {});
    //   // --- Notes --- //
    //   for (int j=0; j<Notes.size(); j++)
    //   {
    //     ExprAST *note = Notes[j].get();
    //     if (NumberExprAST* numExpr = dynamic_cast<NumberExprAST*>(note)) 
    //       notes_vector = callret("Add_To_NotesVector_float", {notes_vector, note->codegen(scope_struct)});
    //     else if (IntExprAST* numExpr = dynamic_cast<IntExprAST*>(note))
    //       notes_vector = callret("Add_To_NotesVector_int", {notes_vector, note->codegen(scope_struct)});
    //     else if (StringExprAST* expr = dynamic_cast<StringExprAST*>(note)) {
    //       Value *str_val = callret("CopyString", {scope_struct, note->codegen(scope_struct)});
    //       notes_vector = callret("Add_To_NotesVector_str", {notes_vector, str_val});
    //     }
    //     else if (Nameable* expr = dynamic_cast<Nameable*>(note)) {   
    //       if(expr->Depth==1&&data_typeVars[parser_struct.function_name].count(expr->Name)==0)
    //       { 
    //         Value *_str = callret("CopyString", {scope_struct, global_str(expr->Name)});
    //         Value *str_val = callret("CopyString", {scope_struct, _str});
    //         notes_vector = callret("Add_To_NotesVector_str", {notes_vector, str_val});
    //       } else {
    //         std::string type = expr->GetDataTree().Type;
    //         notes_vector = callret("Add_To_NotesVector_"+type, {notes_vector, note->codegen(scope_struct)});
    //       }   
    //     }
    //     else {
    //       std::cout << "Could not find the data type of a note in DataExpr of " << VarName << " \n";
    //     }
    //   }
    // } else
    //   notes_vector = callret("nullptr_get", {});
        
      
    if(!IsStruct||Type=="list"||Type=="array"||Type=="map") {
      if (auto *null_stmt = dynamic_cast<NullPtrExprAST*>(VarNames[i].second.get())) {
          std::string create_fn = Type;
          create_fn = (create_fn=="tuple") ? "list" : create_fn;
          create_fn = create_fn + "_Create";
          if(Check_Required_Args_Count(create_fn, Notes.size(), parser_struct)) {

              std::vector<Value *> ArgsV = {scope_struct};
              // if (HasNotes) {
              //     for (int j=0; j<Notes.size(); j++) {
              //         ExprAST *note = Notes[j].get();
              //         ArgsV.push_back(note->codegen(scope_struct));
              //     }
              // }


              if (create_fn=="array_Create" || create_fn=="map_Create")
                  ArgsV.push_back(VoidPtr_toValue(&data_type));
              else
                  ArgsV = Codegen_Argument_List(parser_struct, std::move(ArgsV), std::move(Notes), scope_struct, create_fn, true, 1);

              initial_value = callret(create_fn, ArgsV);
          }
      }
    }

      

    if(is_self)
    {
      int object_ptr_offset = ClassVariables[parser_struct.class_name][VarName]; 
      Value *obj = get_scope_obj(scope_struct);
      call("object_ptr_Attribute_object", {obj, const_int(object_ptr_offset), initial_value});
    } else if (is_attr) {
      LogError(parser_struct.line, "Creating attribute in a data expression is not supported.");
    }
    else { 
        Allocate_On_Pointer_Stack(scope_struct, parser_struct.function_name, VarName, initial_value); 
        function_values[parser_struct.function_name][VarName] = initial_value;
        if (parser_struct.loop_depth==0&&Type=="array")
            Cache_Array(parser_struct.function_name, initial_value);
           
    }      

    // if(HasNotes)
    //   call("Dispose_NotesVector", {notes_vector, scopeless_name});
    // else
    //   call("str_Delete", {scopeless_name});
    // call("str_Delete", {var_name});
  }


  return const_float(0.0f);
}






Value *LibImportExprAST::codegen(Value *scope_struct) {
  // Library import is made before codegen
  
  return const_float(0.0f);
}



Value *GCSafePointExprAST::codegen(Value *scope_struct) {
    Function *TheFunction = Builder->GetInsertBlock()->getParent();
    Set_Stack_Top(scope_struct, parser_struct.function_name);
    check_scope_struct_sweep(TheFunction, scope_struct, parser_struct);
    return const_float(0.0f);
}


Value *IfExprAST::codegen(Value *scope_struct) {
  if (not ShallCodegen)
    return const_float(0.0f);


  Value *CondV = Cond->codegen(scope_struct);
  if (!CondV)
    return nullptr;



  Function *TheFunction = Builder->GetInsertBlock()->getParent();

  // Create blocks for the then and else cases.  Insert the 'then' block at the
  // end of the function.
  BasicBlock *ThenBB  = BasicBlock::Create(*TheContext, "if_then", TheFunction);
  BasicBlock *ElseBB  = BasicBlock::Create(*TheContext, "if_else");
  BasicBlock *MergeBB = BasicBlock::Create(*TheContext, "if_cont");
  

  Builder->CreateCondBr(CondV, ThenBB, ElseBB);
  Builder->SetInsertPoint(ThenBB);

  auto old_values = function_values[parser_struct.function_name];
 
  
  Value *ThenV;
  for (auto &then_body : Then)
    ThenV = then_body->codegen(scope_struct);
  auto then_values = function_values[parser_struct.function_name];


  bool ThenTerminated = Builder->GetInsertBlock()->getTerminator() != nullptr;
  if (!ThenV)
    return nullptr;
  if (!ThenTerminated) {
      Builder->CreateBr(MergeBB);
      // Codegen of 'Then' can change the current block, update ThenBB for the PHI.
      ThenBB = Builder->GetInsertBlock();
  }

  // Emit else block.
  TheFunction->insert(TheFunction->end(), ElseBB);
  Builder->SetInsertPoint(ElseBB);


  function_values[parser_struct.function_name] = old_values;
  Value *ElseV;
  for (auto &else_body : Else)
    ElseV = else_body->codegen(scope_struct);

  std::map<std::string, Value *> else_values;
  if(Else.size()>0)
      else_values = function_values[parser_struct.function_name];
  else {
      ElseV = const_int(0);
      else_values = old_values;
  }

  bool ElseTerminated = Builder->GetInsertBlock()->getTerminator() != nullptr;
  if (!ElseV)
    return nullptr;
  if (!ElseTerminated) {
      Builder->CreateBr(MergeBB);
      // Codegen of 'Else' can change the current block, update ElseBB for the PHI.
      ElseBB = Builder->GetInsertBlock();
  }

  if (ThenTerminated && ElseTerminated)
      return nullptr;

  // Emit merge block.
  TheFunction->insert(TheFunction->end(), MergeBB);
  Builder->SetInsertPoint(MergeBB);



  for (auto &[name, value] : old_values) {
    if (then_values[name]!=value || else_values[name]!=value) {
        PHINode *phi = Builder->CreatePHI(value->getType(), 2);
        phi->addIncoming(then_values[name], ThenBB);
        phi->addIncoming(else_values[name], ElseBB);
        function_values[parser_struct.function_name][name] = phi;
    }
  }
  return const_float(0.0f);
}


Value *IfExprAST::codegen_from_loop(Value *scope_struct, BasicBlock *LoopBody, BasicBlock *LoopAfter) {
  if (not ShallCodegen)
    return const_float(0.0f);


  Value *CondV = Cond->codegen(scope_struct);
  if (!CondV)
    return nullptr;



  Function *TheFunction = Builder->GetInsertBlock()->getParent();

  // Create blocks for the then and else cases.  Insert the 'then' block at the
  // end of the function.
  BasicBlock *ThenBB  = BasicBlock::Create(*TheContext, "if_then", TheFunction);
  BasicBlock *ElseBB  = BasicBlock::Create(*TheContext, "if_else");
  BasicBlock *MergeBB = BasicBlock::Create(*TheContext, "if_cont");
  

  Builder->CreateCondBr(CondV, ThenBB, ElseBB);
  Builder->SetInsertPoint(ThenBB);

  auto old_values = function_values[parser_struct.function_name];
 
  
  Value *ThenV;
  for (auto &then_body : Then) {
    if (auto *if_stmt = dynamic_cast<IfExprAST*>(then_body.get()))
        if_stmt->codegen_from_loop(scope_struct, ThenBB, LoopAfter);
    else if (auto *break_stmt = dynamic_cast<BreakExprAST*>(then_body.get())) {
        Builder->CreateBr(LoopAfter);
        Builder->SetInsertPoint(LoopBody);
    }
    else
        ThenV = then_body->codegen(scope_struct);
  }
  auto then_values = function_values[parser_struct.function_name];


  bool ThenTerminated = Builder->GetInsertBlock()->getTerminator() != nullptr;
  if (!ThenV)
    return nullptr;
  if (!ThenTerminated) {
      Builder->CreateBr(MergeBB);
      // Codegen of 'Then' can change the current block, update ThenBB for the PHI.
      ThenBB = Builder->GetInsertBlock();
  }

  // Emit else block.
  TheFunction->insert(TheFunction->end(), ElseBB);
  Builder->SetInsertPoint(ElseBB);


  function_values[parser_struct.function_name] = old_values;
  Value *ElseV;
  for (auto &else_body : Else)
    ElseV = else_body->codegen(scope_struct);

  std::map<std::string, Value *> else_values;
  if(Else.size()>0)
      else_values = function_values[parser_struct.function_name];
  else {
      ElseV = const_int(0);
      else_values = old_values;
  }

  bool ElseTerminated = Builder->GetInsertBlock()->getTerminator() != nullptr;
  if (!ElseV)
    return nullptr;
  if (!ElseTerminated) {
      Builder->CreateBr(MergeBB);
      // Codegen of 'Else' can change the current block, update ElseBB for the PHI.
      ElseBB = Builder->GetInsertBlock();
  }

  if (ThenTerminated && ElseTerminated)
      return nullptr;

  // Emit merge block.
  TheFunction->insert(TheFunction->end(), MergeBB);
  Builder->SetInsertPoint(MergeBB);



  for (auto &[name, value] : old_values) {
    if (then_values[name]!=value || else_values[name]!=value) {
        PHINode *phi = Builder->CreatePHI(value->getType(), 2);
        phi->addIncoming(then_values[name], ThenBB);
        phi->addIncoming(else_values[name], ElseBB);
        function_values[parser_struct.function_name][name] = phi;
    }
  }
  return const_float(0.0f);
}


  




// Output for-loop as:
//   var = alloca float
//   ...
//   start = startexpr
//   store start -> var
//   goto loop
// loop:
//   ...
//   bodyexpr
//   ...
// loopend:
//   step = stepexpr
//   endcond = endexpr
//
//   curvar = load var
//   nextvar = curvar + step
//   store nextvar -> var
//   br endcond, loop, endloop
// outloop:



Value *BreakExprAST::codegen(Value *scope_struct) { 
    p2t("break");
    return const_int(0);
}


void Codegen_Loop_Body(Value *scope_struct, std::vector<std::unique_ptr<ExprAST>> Body, BasicBlock *LoopBody, BasicBlock *AfterBB) {  
  for (auto &body : Body) {
    if (auto *if_stmt = dynamic_cast<IfExprAST*>(body.get()))
        if_stmt->codegen_from_loop(scope_struct, LoopBody, AfterBB);
    else
        body->codegen(scope_struct);
  }
}

void Get_Recursive_Assign_Statements(const std::vector<std::unique_ptr<ExprAST>> &stmt, std::vector<std::string> &assigned_vars) {

  for (auto &body : stmt) {
      if (auto *if_stmt = dynamic_cast<IfExprAST*>(body.get())) {
          Get_Recursive_Assign_Statements(if_stmt->Then, assigned_vars);
          Get_Recursive_Assign_Statements(if_stmt->Else, assigned_vars);
      }
      // if (auto *if_stmt = dynamic_cast<IfExprAST*>(body.get())) {
      //     Get_Recursive_Assign_Statements(if_stmt->Then, assigned_vars);
      //     Get_Recursive_Assign_Statements(if_stmt->Else, assigned_vars);
      // }
      if (auto *bin_stmt = dynamic_cast<BinaryExprAST*>(body.get())) {
          char op = bin_stmt->Op;
          if (op=='='||op==tok_arrow) {
            if (auto *nameable_stmt = dynamic_cast<Nameable*>(bin_stmt->LHS.get())) {
                if(typeid(*nameable_stmt)==typeid(Nameable)) {
                    if (nameable_stmt->Depth==1) {
                        assigned_vars.push_back(nameable_stmt->GetName());
                    }
                }
            }
          }
      }
  }

}


Value *ForExprAST::codegen(Value *scope_struct) {
  if (not ShallCodegen)
    return const_float(0);

  Function *TheFunction = Builder->GetInsertBlock()->getParent();
  check_scope_struct_sweep(TheFunction, scope_struct, parser_struct);

  std::string start_type = Start->GetDataTree().Type;
  llvm::Type *llvm_type = get_type_from_str(start_type);
  
 
  Value *StartVal = Start->codegen(scope_struct);
  if (!StartVal)
    return nullptr;




  // Make the new basic block for the loop header, inserting after current
  // block.
  BasicBlock *CondBB = BasicBlock::Create(*TheContext, "for expr cond", TheFunction);
  BasicBlock *LoopBB  = BasicBlock::Create(*TheContext, "for expr loop");
  BasicBlock *AfterBB  = BasicBlock::Create(*TheContext, "for expr after");
  BasicBlock *PreheaderBB = Builder->GetInsertBlock();


  Builder->CreateBr(CondBB);
  Builder->SetInsertPoint(CondBB);


  std::vector<std::string> assigned_vars, changed_vars;
  Get_Recursive_Assign_Statements(Body, assigned_vars);


  // Possible phi for each value
  auto old_function_values = function_values[parser_struct.function_name];
  std::map<std::string, PHINode*> function_phi_values;
  for (const auto &name : assigned_vars) {
      if (name!=VarName && old_function_values.count(name)>0) {
        changed_vars.push_back(name);
        Value *val = old_function_values[name];
        PHINode *phi_val = Builder->CreatePHI(val->getType(), 2, name.c_str());
        phi_val->addIncoming(val, PreheaderBB);
        function_phi_values[name] = phi_val;
        function_values[parser_struct.function_name][name] = phi_val;
      }
  }

  // Control var phi
  PHINode *LoopVar = Builder->CreatePHI(StartVal->getType(), 2, VarName.c_str());
  LoopVar->addIncoming(StartVal, PreheaderBB);
  function_values[parser_struct.function_name][VarName] = LoopVar;

 
  

  // Emit the step value.
  Value *StepVal = nullptr;
  if (Step) {
    StepVal = Step->codegen(scope_struct);
    if (!StepVal)
      return nullptr;
  } 



  // Compute the end condition.
  Value *EndCond = End->codegen(scope_struct);
  if (!EndCond)
    return nullptr;


  // conditional goto branch
  Builder->CreateCondBr(EndCond, LoopBB, AfterBB);


  // codegen body and increment
  TheFunction->insert(TheFunction->end(), LoopBB);
  Builder->SetInsertPoint(LoopBB);

  int j=0;
  Codegen_Loop_Body(scope_struct, std::move(Body), LoopBB, AfterBB);
  // for (auto &body : Body) {
  //   body->codegen(scope_struct);
  // }


  BasicBlock *CurBB = Builder->GetInsertBlock(); // Catch branching scenarios
  Value *NextVal;
  if (start_type=="int")
    NextVal = Builder->CreateAdd(LoopVar, StepVal, "nextvar"); // Increment  
  if (start_type=="float")
    NextVal = Builder->CreateFAdd(LoopVar, StepVal, "nextvar"); // Increment 
 
  Builder->CreateBr(CondBB);


  LoopVar->addIncoming(NextVal, CurBB);

  for (auto &name : changed_vars) 
    function_phi_values[name]->addIncoming(function_values[parser_struct.function_name][name], CurBB);
   


  TheFunction->insert(TheFunction->end(), AfterBB);
  Builder->SetInsertPoint(AfterBB);
 
  // verifyFunction(*TheFunction);
  // TheModule->print(llvm::errs(), nullptr);

  return Constant::getNullValue(Type::getInt32Ty(*TheContext));
}





Value *ForEachExprAST::codegen(Value *scope_struct) {
  if (not ShallCodegen)
    return ConstantFP::get(*TheContext, APFloat(0.0f));

  Function *TheFunction = Builder->GetInsertBlock()->getParent();
  check_scope_struct_sweep(TheFunction, scope_struct, parser_struct);


  Value *_zero = const_int(0);
  Value *CurIdx = const_int(0);


  function_values[parser_struct.function_name][VarName] = CurIdx;
  

  Value *vec = Vec->codegen(scope_struct);

  StructType *st = struct_types["vec"];
  Value *vec_size_gep = Builder->CreateStructGEP(st, vec, 0);
  Value *VecSize = Builder->CreateLoad(intTy, vec_size_gep);


  // VecSize = Builder->CreateFAdd(VecSize, const_float(1), "addtmp");

  // Make the new basic block for the loop header, inserting after current
  // block.
  BasicBlock *CondBB = BasicBlock::Create(*TheContext, "cond", TheFunction);
  BasicBlock *LoopBB  = BasicBlock::Create(*TheContext, "loop");
  BasicBlock *AfterBB  = BasicBlock::Create(*TheContext, "after");
  BasicBlock *PreheaderBB = Builder->GetInsertBlock();



  // Insert an explicit fall through from the current block to the LoopBB.
  Builder->CreateBr(CondBB); 
  Builder->SetInsertPoint(CondBB);


  // --- PHI Nodes --- //
  std::vector<std::string> assigned_vars, changed_vars;
  Get_Recursive_Assign_Statements(Body, assigned_vars);

  // Possible phi for each value
  auto old_function_values = function_values[parser_struct.function_name];
  std::map<std::string, PHINode*> function_phi_values;
  for (const auto &name : assigned_vars) {
      if (name!=VarName && old_function_values.count(name)>0) {
        changed_vars.push_back(name);
        Value *val = old_function_values[name];
        PHINode *phi_val = Builder->CreatePHI(val->getType(), 2, name.c_str());
        phi_val->addIncoming(val, PreheaderBB);
        function_phi_values[name] = phi_val;
        function_values[parser_struct.function_name][name] = phi_val;
      }
  }
  // Control var phi
  PHINode *LoopVar = Builder->CreatePHI(CurIdx->getType(), 2, VarName.c_str());
  LoopVar->addIncoming(CurIdx, PreheaderBB);
  function_values[parser_struct.function_name][VarName] = LoopVar;



  // Emit the body of the loop.  This, like any other expr, can change the
  // current BB.  Note that we ignore the value computed by the body, but don't
  // allow an error.
 
  Value *StepVal = const_int(1);


  // Compute the end condition.
  Value *EndCond = Builder->CreateICmpNE(
      LoopVar, VecSize, "loopcond");


  // conditional goto branch
  Builder->CreateCondBr(EndCond, LoopBB, AfterBB);


  // codegen body and increment
  TheFunction->insert(TheFunction->end(), LoopBB);
  Builder->SetInsertPoint(LoopBB);

  
  // CurIdx = Builder->CreateLoad(Type::getInt32Ty(*TheContext), idx_alloca, VarName.c_str());


  std::string vec_type = UnmangleVec(data_type);
  Value *vec_value;
  if (vec_type=="array") {
    StructType *st = struct_types["vec"];
    llvm::Type *elem_type = get_type_from_str(Type); 

    Value *elem_size_gep = Builder->CreateStructGEP(st, vec, 2); 
    Value *elem_size = Builder->CreateLoad(intTy, elem_size_gep);

    Value *vec_gep = Builder->CreateStructGEP(st, vec, 3);
    Value *array = Builder->CreateLoad(int8PtrTy, vec_gep);

    Value *element = Builder->CreateGEP(Type::getInt8Ty(*TheContext), array, Builder->CreateMul(LoopVar, elem_size));
    vec_value = Builder->CreateLoad(elem_type, element, "elem");  
  } else
      vec_value = callret(vec_type+"_Idx", {scope_struct, vec, LoopVar});
  if((vec_type=="list"||vec_type=="tuple")&&(Type=="float"||Type=="int"))
    vec_value = callret("to_"+Type, {scope_struct, vec_value});
  function_values[parser_struct.function_name][VarName] = vec_value;

  Codegen_Loop_Body(scope_struct, std::move(Body), LoopBB, AfterBB);

  BasicBlock *CurBB = Builder->GetInsertBlock(); // Catch branching scenarios
  Value *NextVal = Builder->CreateAdd(LoopVar, StepVal, "nextvar"); // Increment  

  LoopVar->addIncoming(NextVal, CurBB);
  for (auto &name : changed_vars)
    function_phi_values[name]->addIncoming(function_values[parser_struct.function_name][name], CurBB);
  

  Builder->CreateBr(CondBB);



  // when the loop body is done, return the insertion point to outside the for loop
  TheFunction->insert(TheFunction->end(), AfterBB);
  Builder->SetInsertPoint(AfterBB);


  return const_float(0.0f);
}


Value *WhileExprAST::codegen(Value *scope_struct) {
  if (not ShallCodegen)
    return const_float(0.0f);
  
  Function *TheFunction = Builder->GetInsertBlock()->getParent();
  check_scope_struct_sweep(TheFunction, scope_struct, parser_struct);

  // Create blocks for loop condition, loop body, and after loop
  BasicBlock *CondBB = BasicBlock::Create(*TheContext, "cond_while", TheFunction);
  BasicBlock *LoopBB = BasicBlock::Create(*TheContext, "loop_while", TheFunction);
  BasicBlock *AfterBB = BasicBlock::Create(*TheContext, "end_while", TheFunction);
  BasicBlock *PreheaderBB = Builder->GetInsertBlock();

  // Jump to the condition block
  Builder->CreateBr(CondBB);
  Builder->SetInsertPoint(CondBB);


  std::vector<std::string> assigned_vars, changed_vars;
  Get_Recursive_Assign_Statements(Body, assigned_vars);

  // Possible phi for each value
  auto old_function_values = function_values[parser_struct.function_name];
  std::map<std::string, PHINode*> function_phi_values;
  for (const auto &name : assigned_vars) {
      if (old_function_values.count(name)>0) {
        changed_vars.push_back(name);
        Value *val = old_function_values[name];
        PHINode *phi_val = Builder->CreatePHI(val->getType(), 2, name.c_str());
        phi_val->addIncoming(val, PreheaderBB);
        function_phi_values[name] = phi_val;
        function_values[parser_struct.function_name][name] = phi_val;
      }
  }


  // Generate the condition code
  Value* condVal = Cond->codegen(scope_struct);
  if (!condVal)
    return nullptr;


  // Create the conditional branch
  Builder->CreateCondBr(condVal, LoopBB, AfterBB);

  // Insert the loop body block
  Builder->SetInsertPoint(LoopBB);

  // Generate the loop body code
  Codegen_Loop_Body(scope_struct, std::move(Body), LoopBB, AfterBB);


  BasicBlock *CurBB = Builder->GetInsertBlock(); // handles branching
  for (auto &name : changed_vars)
    function_phi_values[name]->addIncoming(function_values[parser_struct.function_name][name], CurBB);

  // After the loop body, go back to the condition check
  Builder->CreateBr(CondBB);

  // Insert the after loop block
  Builder->SetInsertPoint(AfterBB);

  return Constant::getNullValue(Type::getFloatTy(*TheContext));
}









bool seen_var_attr = false;








bool CheckIs_CastInt_to_FloatChannel(std::string Operation, Data_Tree LTree) {
  if(Operation!="channel_int_message")
    return false;

  if(LTree.Type=="channel" && LTree.Nested_Data.size()>0) {
    if(LTree.Nested_Data[0].Type=="float")
      return true;
  }
    
  return false;
}



bool CheckIsSenderChannel(std::string Elements, Parser_Struct parser_struct, std::string LName) {

  if(begins_with(Elements, "channel")) {
    if(ChannelDirections[parser_struct.function_name].count(LName)>0) {
      if(((int)ChannelDirections[parser_struct.function_name][LName])==(int)ch_sender) {
        LogError(parser_struct.line, "1 Tried to attribute data to a sender only channel." + parser_struct.function_name + "/" + LName + ": " + std::to_string(ChannelDirections[parser_struct.function_name].count(LName)) + "; " + std::to_string(ChannelDirections[parser_struct.function_name][LName]) );
        return false;
      }
    }
    else {
      if(ChannelDirections[parser_struct.class_name].count(LName)>0) {
        if((int)ChannelDirections[parser_struct.class_name][LName]==(int)ch_sender) {
          LogError(parser_struct.line, "2 Tried to attribute data to a sender only channel. " + parser_struct.class_name + "/" + LName + ": " + std::to_string(ChannelDirections[parser_struct.class_name].count(LName)));
          return false;
        }
      }
    }
  }
  return true;
}
















std::string BinaryExprAST::GetType(bool from_assignment) {
  std::string type = Type;
  if (type=="None")
  {
    std::string LType = LHS->GetDataTree().Type, RType = RHS->GetDataTree().Type;
    if ((LType=="list"||RType=="list") && Op!='=')
      LogError(parser_struct.line, "Tuple elements type are unknown during parsing type. Please load the element into a static type variable first.");
    
    Elements = LType + "_" + RType;    
    
    std::string operation = op_map[Op];
    Operation = Elements + "_" + operation;
    

    std::string type;
    if (Operation=="int_int_div")
      type = "float";
    if (elements_type_return.count(Operation)>0)
    {
      type = elements_type_return[Operation];
      std::cout << "found " << type << " for " << Operation << ".\n";
    }
    if (elements_type_return.count(Elements)>0)
      type = elements_type_return[Elements];
    SetType(type);

  }
  return type;
}


Value *BinaryExprAST::codegen(Value *scope_struct) {
  if (not ShallCodegen)
    return ConstantFP::get(*TheContext, APFloat(0.0f));

  GetDataTree(); // gets the correct Elements and Operation.


  if (Op == '=' || (Op==tok_arrow&&!begins_with(Elements, "channel"))) {
    seen_var_attr=true;
    // Assignment requires the LHS to be an identifier.
    // This assume we're building without RTTI because LLVM builds that way by
    // default.  If you build LLVM with RTTI this can be changed to a
    // dynamic_cast for automatic error checking.


    std::string LType = LHS->GetDataTree().Type;

    // Codegen the RHS.
    Value *Val;
    bool from_channel_op=false;
    if (Op==tok_arrow)
    {
      if(ChannelDirections[parser_struct.function_name].count(RHS->GetName())==0)
      {
        LogError(parser_struct.line, "Could not find channel " + RHS->GetName());
        return const_float(0);
      }
      if(ChannelDirections[parser_struct.function_name][RHS->GetName()]==ch_receiver)
      {
        LogError(parser_struct.line, "Trying to unpack data from a receiver only channel.");
        return const_float(0);
      }
      
      bool is_high_lvl_obj = ClassSize.count(LType)>0;
      if(is_high_lvl_obj)
          Operation = "void_channel_message";
      Val = callret(Operation, {scope_struct, LHS->codegen(scope_struct), RHS->codegen(scope_struct)});

      if (!in_str(LType, primary_data_tokens) && !is_high_lvl_obj) {
          std::string copy_fn = LType + "_Copy";

          Function *F = TheModule->getFunction(copy_fn);
          if (!F) 
              return LogErrorV(parser_struct.line, "Tried to use channel operation for " + \
                                                    LType + ", but this data type has no Copy implementation.");
          Val = callret(copy_fn, {scope_struct, Val});
      }

      from_channel_op = true;
    }
    else
      Val = RHS->codegen(scope_struct);
    



    if (!Val)
    {
      seen_var_attr=false;
      return nullptr; 
    }



    if (LHS->GetIsList())
    {
      VariableListExprAST *VarList = static_cast<VariableListExprAST *>(LHS.get());

      for (int i=0; i<VarList->ExprList.size(); ++i)
      {
        Nameable *LHSE = static_cast<Nameable *>(VarList->ExprList[i].get()); 

        std::string Lname = LHSE->Name;

        std::string list_LType = LHSE->GetDataTree().Type;
        

        std::string store_trigger = list_LType + "_StoreTrigger";
        Value *Val_indexed = callret("assign_wise_list_Idx", {Val, const_int(i)});
        
        // std::string copy_fn = list_LType + "_Copy";
        // Function *F = TheModule->getFunction(copy_fn);
        // if (F)
        //   Val_indexed = callret(copy_fn, {scope_struct, Val_indexed});
        
        

        Function *F = TheModule->getFunction(store_trigger);
        if (F)
        {
          Value *old_val = function_values[parser_struct.function_name][Lname]; 
          call(store_trigger, {scope_struct, old_val, Val_indexed});
        }

        if (list_LType=="int")
          Val_indexed = callret("to_int", {scope_struct, Val_indexed});
        if (list_LType=="float")
          Val_indexed = callret("to_float", {scope_struct, Val_indexed});

        function_values[parser_struct.function_name][Lname] = Val_indexed;

        Set_Pointer_Stack(scope_struct, parser_struct.function_name, Lname, Val);
      }
      return const_float(0.0f);
    }

    std::string RType = RHS->GetDataTree().Type;
    if (LType=="int"&&RType=="any")
      Val = callret("to_int", {scope_struct, Val}); 
    if (LType=="float"&&RType=="any")
      Val = callret("to_float", {scope_struct, Val});




    bool is_alloca = (!LHS->GetSelf()&&!LHS->GetIsAttribute());
    
    Value *Lvar_name; 

    if(auto *LHSV = dynamic_cast<NameableIdx *>(LHS.get())) {
      
      Value *vec_ptr = LHSV->Inner->codegen(scope_struct);
      Data_Tree dt = LHSV->GetDataTree(true);
      std::string type = UnmangleVec(dt);

      Value *idx = Idx_Calc_Codegen(type, vec_ptr, LHSV->Idx, scope_struct); //StoreIdx

      if(type=="list"||type=="dict") {
        std::string nested_type = dt.Nested_Data[0].Type;
        if (in_str(nested_type, primary_data_tokens))
          type = nested_type + "_" + type;
      }
      
      if(type=="map")
      {
        Data_Tree map_dt = LHS->GetDataTree();
        Check_Is_Compatible_Data_Type(map_dt, RHS->GetDataTree(), parser_struct);
        std::string key_type = map_dt.Nested_Data[0].Type;
        std::string value_type = map_dt.Nested_Data[1].Type;




        Value *query = idx;

        StructType *st = struct_types["map"];
        StructType *st_node = struct_types["map_node"];

        if (query->getType()==intTy&&key_type=="float")
            query = Builder->CreateSIToFP(query, floatTy);
        else if(LHSV->Idx->GetDataTree().Type!=key_type)
            return LogErrorV(parser_struct.line, "Querying " + key_type + " map with " + LHSV->Idx->GetDataTree().Type);

        Value *nullPtr = ConstantPointerNull::get(
            cast<PointerType>(int8PtrTy)
        );

        // Create the node to be stored
        Value *new_node_ptr = callret("allocate_void", {scope_struct, const_int(24), global_str("map_node")});
        Value *new_node_key_gep = Builder->CreateStructGEP(st_node, new_node_ptr, 0);
        if (key_type=="int") {
            Value *int_ptr = callret("malloc", {const_int(4)});
            Builder->CreateStore(query, int_ptr);
            Builder->CreateStore(int_ptr, new_node_key_gep);
        } else if (key_type=="float") {
            Value *float_ptr = callret("malloc", {const_int(4)});
            Builder->CreateStore(query, float_ptr);
            Builder->CreateStore(float_ptr, new_node_key_gep);
        } else
            Builder->CreateStore(query, new_node_key_gep); 
        Value *new_node_value_gep = Builder->CreateStructGEP(st_node, new_node_ptr, 1);
        if (value_type=="int") {
            Value *int_ptr = callret("malloc", {const_int(4)});
            Builder->CreateStore(Val, int_ptr);
            Builder->CreateStore(int_ptr, new_node_value_gep);
        } else if (value_type=="float") {
            Value *float_ptr = callret("malloc", {const_int(4)});
            Builder->CreateStore(Val, float_ptr);
            Builder->CreateStore(float_ptr, new_node_value_gep);
        } else
            Builder->CreateStore(Val, new_node_value_gep);

        Value *new_node_next_gep = Builder->CreateStructGEP(st_node, new_node_ptr, 2);
        Builder->CreateStore(nullPtr, new_node_next_gep);

        // Load map attributes 
        Value *size_gep = Builder->CreateStructGEP(st, vec_ptr, 0);
        Value *map_size = Builder->CreateLoad(intTy, size_gep);

        Value *capacity_gep = Builder->CreateStructGEP(st, vec_ptr, 1);
        Value *map_capacity = Builder->CreateLoad(intTy, capacity_gep);

        Value *expand_at_gep = Builder->CreateStructGEP(st, vec_ptr, 2);
        Value *map_expand_at = Builder->CreateLoad(intTy, expand_at_gep);

        Value *nodes_gep = Builder->CreateStructGEP(st, vec_ptr, 5);
        Value *nodes = Builder->CreateLoad(int8PtrTy->getPointerTo(), nodes_gep);


        // Check for expansion
        Function *TheFunction = Builder->GetInsertBlock()->getParent();
        BasicBlock *MapInsertBB = BasicBlock::Create(*TheContext, "map.insert.bb", TheFunction);
        BasicBlock *MapExpandBB = BasicBlock::Create(*TheContext, "map.expand.bb", TheFunction);
        
        Value *new_size = Builder->CreateAdd(map_size, const_int(1));
        Builder->CreateStore(new_size, size_gep);

        Value *expandCond = Builder->CreateICmpSGE(new_size, map_expand_at);
    
        Builder->CreateCondBr(expandCond, MapExpandBB, MapInsertBB);

        
        Builder->SetInsertPoint(MapExpandBB);
        call("map_expand", {scope_struct, vec_ptr});
        Builder->CreateBr(MapInsertBB);


        Builder->SetInsertPoint(MapInsertBB);



        BasicBlock *CheckFirstKeyBB = BasicBlock::Create(*TheContext, "map.check_first_key.bb", TheFunction);
        BasicBlock *FromFirstKeyBB = BasicBlock::Create(*TheContext, "map.from_first_key.bb", TheFunction);
        BasicBlock *FromKeyBB = BasicBlock::Create(*TheContext, "map.from_first_key.bb", TheFunction);
        BasicBlock *PtrChaseBB = BasicBlock::Create(*TheContext, "map.pointer_chase.bb", TheFunction);
        BasicBlock *PtrChaseCheckKeyBB = BasicBlock::Create(*TheContext, "map.pointer_chase_check_key.bb", TheFunction);
        BasicBlock *FromPtrChaseBB = BasicBlock::Create(*TheContext, "map.new_from_pointer_chase.bb", TheFunction);
        BasicBlock *FromNullBB = BasicBlock::Create(*TheContext, "map.from_null.bb", TheFunction);
        BasicBlock *AfterBB = BasicBlock::Create(*TheContext, "map.after.bb", TheFunction);

        
        // Check if bucket is nullptr
        Value *query_hash;
        if (key_type=="str")
            query_hash = str_llvm_hash(query, TheFunction);        
        if (key_type=="float")
            query_hash = float_llvm_hash(query);
        if (key_type=="int")
            query_hash = query;
        Value *hash_pos = Builder->CreateURem(query_hash, map_capacity);

        // call("print_str", {query});
        // call("print_int", {hash_pos});

        Value *node_gep = Builder->CreateGEP(int8PtrTy->getPointerTo(), nodes, hash_pos);
        Value *node = Builder->CreateLoad(int8PtrTy, node_gep);

        Value *IsNull = Builder->CreateICmpEQ(node, nullPtr);

        Builder->CreateCondBr(IsNull, FromNullBB, CheckFirstKeyBB);

        // From nullptr
        Builder->SetInsertPoint(FromNullBB);
        Builder->CreateStore(new_node_ptr, node_gep); 
        Builder->CreateBr(AfterBB);

        // Check first key
        Builder->SetInsertPoint(CheckFirstKeyBB);
        Value *key_gep = Builder->CreateStructGEP(st_node, node, 0);
        Value *keyCond, *key;
        if (key_type=="int") {
            Value *key_void_ptr = Builder->CreateLoad(int8PtrTy, key_gep);
            Value *key_int_ptr = Builder->CreateBitCast(key_void_ptr, intTy->getPointerTo());
            key = Builder->CreateLoad(intTy, key_int_ptr);
            keyCond = Builder->CreateICmpEQ(key, query);
        } else if (key_type=="float") {
            Value *key_void_ptr = Builder->CreateLoad(int8PtrTy, key_gep);
            Value *key_float_ptr = Builder->CreateBitCast(key_void_ptr, floatTy->getPointerTo());
            key = Builder->CreateLoad(floatTy, key_float_ptr);
            keyCond = Builder->CreateFCmpUEQ(key, query);
        } else {
            key = Builder->CreateLoad(int8PtrTy, key_gep);
            keyCond = callret("strcmp", {key, query});
            keyCond = Builder->CreateICmpEQ(keyCond, const_int(0));
        }
        Builder->CreateCondBr(keyCond, FromFirstKeyBB, PtrChaseBB);


        // Key overwrite
        Builder->SetInsertPoint(FromFirstKeyBB);
        Value *next_node_of_first_gep = Builder->CreateStructGEP(st_node, node, 2);
        Value *next_node_of_first = Builder->CreateLoad(int8PtrTy, next_node_of_first_gep);
        Builder->CreateStore(next_node_of_first, new_node_next_gep);
        Builder->CreateStore(new_node_ptr, node_gep);
        Builder->CreateBr(AfterBB);


        // Pointer Chase
        Builder->SetInsertPoint(PtrChaseBB);
        PHINode *map_phi_node = Builder->CreatePHI(int8PtrTy, 2);
        map_phi_node->addIncoming(node, CheckFirstKeyBB);
        
        Value *next_node_gep = Builder->CreateStructGEP(st_node, map_phi_node, 2);
        Value *next_node = Builder->CreateLoad(int8PtrTy, next_node_gep);
        map_phi_node->addIncoming(next_node, PtrChaseCheckKeyBB);

        Value *IsNextNull = Builder->CreateICmpEQ(next_node, nullPtr);

        Builder->CreateCondBr(IsNextNull, FromPtrChaseBB, PtrChaseCheckKeyBB);

        Builder->SetInsertPoint(PtrChaseCheckKeyBB);
        key_gep = Builder->CreateStructGEP(st_node, next_node, 0);
        if (key_type=="int") {
            Value *key_void_ptr = Builder->CreateLoad(int8PtrTy, key_gep);
            Value *key_int_ptr = Builder->CreateBitCast(key_void_ptr, intTy->getPointerTo());
            key = Builder->CreateLoad(intTy, key_int_ptr);
            keyCond = Builder->CreateICmpEQ(key, query);
        } else if (key_type=="float") {
            Value *key_void_ptr = Builder->CreateLoad(int8PtrTy, key_gep);
            Value *key_float_ptr = Builder->CreateBitCast(key_void_ptr, floatTy->getPointerTo());
            key = Builder->CreateLoad(floatTy, key_float_ptr);
            keyCond = Builder->CreateFCmpUEQ(key, query);
        } else {
            key = Builder->CreateLoad(int8PtrTy, key_gep);
            keyCond = callret("strcmp", {key, query});
            keyCond = Builder->CreateICmpEQ(keyCond, const_int(0));
        }
        Builder->CreateCondBr(keyCond, FromKeyBB, PtrChaseBB);


        Builder->SetInsertPoint(FromKeyBB);
        Value *next_next_node_gep = Builder->CreateStructGEP(st_node, next_node, 2);
        Value *next_next_node = Builder->CreateLoad(int8PtrTy, next_next_node_gep);
        Builder->CreateStore(next_next_node, new_node_next_gep);
        Builder->CreateStore(new_node_ptr, next_node_gep);
        Builder->CreateBr(AfterBB);



        // New pointer from Pointer Chase
        Builder->SetInsertPoint(FromPtrChaseBB);
        Builder->CreateStore(new_node_ptr, next_node_gep); 
        Builder->CreateBr(AfterBB);


        // After
        Builder->SetInsertPoint(AfterBB);
        

        return const_float(0);
      }


















      if (type=="array") {
          Check_Is_Compatible_Data_Type(LHS->GetDataTree(), RHS->GetDataTree(), parser_struct);
          StructType *st = struct_types["vec"];
          std::string elem_type = dt.Nested_Data[0].Type;

          Check_Is_Array_Inbounds(parser_struct, vec_ptr, idx);

          if (elem_type=="float"&&RType=="int")
              Val = Builder->CreateSIToFP(Val, floatTy);

          Value *vec = Load_Array(parser_struct.function_name, vec_ptr);

          llvm::Type *idxTy;
          if (elem_type=="int")
            idxTy = intTy;
          else if (elem_type=="float") 
            idxTy = floatTy;
          else if (elem_type=="bool") 
            idxTy = boolTy;
          else 
            idxTy = int8PtrTy;
          

          Value *element = Builder->CreateGEP(idxTy, vec, idx);
          Builder->CreateStore(Val, element);
      } else
          call(type+"_Store_Idx", {vec_ptr, idx, Val, scope_struct});

      return const_float(0);
    } 

  
    
    


    std::string Lname = LHS->GetName();

    if (is_alloca)
    {
      Check_Is_Compatible_Data_Type(LHS->GetDataTree(), RHS->GetDataTree(), parser_struct);


      std::string store_trigger = LType + "_StoreTrigger";
      std::string copy_fn = LType + "_Copy";

      
      // Copy data types that support copying (i.e, function <DT>_Copy exists)
      if(auto Rvar = dynamic_cast<Nameable *>(RHS.get())) // if it is leaf
      {
        Function *F = TheModule->getFunction(copy_fn);
        if (!from_channel_op && F)
          Val = callret(copy_fn, {scope_struct, Val});
      }

      // Store trigger behavior for supported types (i.e, function <DT>_StoreTrigger exists)
      Function *F = TheModule->getFunction(store_trigger);
      if (F && function_values[parser_struct.function_name].count(Lname)>0)
      {
        Value *old_val = function_values[parser_struct.function_name][Lname];
        Val = callret(store_trigger, {scope_struct, old_val, Val});
      }
      
      // Cast int to float
      if (LType=="float" && RHS->GetType()=="int")
        Val = Builder->CreateUIToFP(Val, Type::getFloatTy(*TheContext), "floattmp");


      if (parser_struct.loop_depth==0&&LType=="array")
          Cache_Array(parser_struct.function_name, Val);


      if(!in_str(LType, primary_data_tokens)) {
          Set_Pointer_Stack(scope_struct, parser_struct.function_name, Lname, Val);
      }

      if (function_values[parser_struct.function_name].count(Lname)==0) {
          LogError(parser_struct.line, "Variable " + Lname + " not yet declared");
          return const_float(0.0f);
      }
      function_values[parser_struct.function_name][Lname] = Val;



      

    } else {
      if(auto *LHSV = dynamic_cast<Nameable *>(LHS.get())) {
        Data_Tree L_dt = LHSV->GetDataTree();

        Check_Is_Compatible_Data_Type(L_dt, RHS->GetDataTree(), parser_struct);
        LType = L_dt.Type;


        LHSV->Load_Last=false;


        Value *obj_ptr = LHSV->codegen(scope_struct);
        
        if(in_str(LType, primary_data_tokens))
          call("object_Attr_"+LType, {obj_ptr, Val});
        else
          call("tie_object_to_object", {obj_ptr, Val});
      } else {}
      

    }
    

    seen_var_attr=false;
    return const_float(0.0f);
  }


  



  
  Value *L = LHS->codegen(scope_struct);
  Value *R = RHS->codegen(scope_struct);


  // GetType(); // gets the correct Elements and Operation.




  
  if (!L || !R)
    return nullptr;

  if (cast_L_to=="int_to_float")
    L = Builder->CreateSIToFP(L, Type::getFloatTy(*TheContext), "lfp");
  if (cast_R_to=="int_to_float")
    R = Builder->CreateSIToFP(R, Type::getFloatTy(*TheContext), "lfp");
  if (Operation=="tensor_int_div") {
    Operation = "tensor_float_div";
    R = Builder->CreateSIToFP(R, Type::getFloatTy(*TheContext), "lfp");
  }

  if (CheckIs_CastInt_to_FloatChannel(Operation, LHS->GetDataTree())) {
    Operation = "channel_float_message";
    R = Builder->CreateSIToFP(R, Type::getFloatTy(*TheContext), "lfp");
  }
  if(!CheckIsSenderChannel(Elements, parser_struct, LHS->GetName()))
    return const_float(0);


  
  if (Elements=="bool_bool") {
    switch (Op) {
      case tok_equal:
        return Builder->CreateICmpEQ(L, R, "booleq");
      case tok_and:
          return Builder->CreateAnd(L, R, "booland");
      case tok_or:
          return Builder->CreateOr(L, R, "boolor");
    }
  }
  

  if (Elements=="float_float")
  {

    switch (Op) {
      case '+':
        return Builder->CreateFAdd(L, R, "addtmp");
      case ':':
        return L;
      case tok_space:
        return R;
      case '-':
        return Builder->CreateFSub(L, R, "subtmp");
      case '*':
        return Builder->CreateFMul(L, R, "multmp");
      case '/':
        return Builder->CreateFDiv(L, R, "divtmp");
      case '%':
        return Builder->CreateFRem(L, R, "remtmp");
      case tok_int_div:
        return LogErrorV(parser_struct.line, "GOTCHA");
      case '<':
        return Builder->CreateFCmpULT(L, R, "cmptmp");
      case '>':
        return Builder->CreateFCmpULT(R, L, "cmptmp");
      case tok_equal:
        return Builder->CreateFCmpUEQ(L, R, "cmptmp");
      case tok_diff:
        return Builder->CreateFCmpUNE(L, R, "cmptmp");
      case tok_minor_eq:
          return Builder->CreateFCmpULE(L, R, "cmptmp");  // less or equal
      case tok_higher_eq:
          return Builder->CreateFCmpUGE(L, R, "cmptmp");  // greater or equal
      default:
        break;
      }

  } else if (Elements=="int_int") {
    switch (Op) {
      case '+':
        return Builder->CreateAdd(L, R, "addtmp");
      case ':':
        return L;
      case tok_space:
        return R;
      case '-':
        return Builder->CreateSub(L, R, "subtmp");
      case '*':
        return Builder->CreateMul(L, R, "multmp");
      case '/':
      {
        llvm::Value* L_float = Builder->CreateSIToFP(L, Type::getFloatTy(*TheContext), "lfp");
        llvm::Value* R_float = Builder->CreateSIToFP(R, Type::getFloatTy(*TheContext), "rfp");
        return Builder->CreateFDiv(L_float, R_float, "divtmp");
        // return Builder->CreateSDiv(L, R, "divtmp");  // Signed division
      }
      case '%':
        return Builder->CreateSRem(L, R, "remtmp");  // Signed remainder
      case tok_int_div:
        return Builder->CreateSDiv(L, R, "divtmp");  // Signed division
      case '<':
        return Builder->CreateICmpSLT(L, R, "cmptmp");
      case '>':
        return Builder->CreateICmpSGT(L, R, "cmptmp");
      case tok_equal:
        return Builder->CreateICmpEQ(L, R, "cmptmp");
      case tok_diff:
        return Builder->CreateICmpNE(L, R, "cmptmp");
      case tok_minor_eq:
        return Builder->CreateICmpSLE(L, R, "cmptmp");
      case tok_higher_eq:
        return Builder->CreateICmpSGE(L, R, "cmptmp");
      default:
        break;
    }
  } else {

    
    if(LHS->GetDataTree().Type=="channel"||RHS->GetDataTree().Type=="channel")
      Check_Is_Compatible_Data_Type(LHS->GetDataTree(), RHS->GetDataTree(), parser_struct);


    Value *ret = callret(Operation, {scope_struct, L, R});

    return ret; 
  }



  

  // If it wasn't a builtin binary operator, it must be a user defined one. Emit
  // a call to it.
  Function *F = getFunction(std::string("binary") + Op);
  assert(F && "Operator not found.");

  Value *Ops[] = {L, R};
  return Builder->CreateCall(F, Ops, "binop");
}

































Value *UnaryExprAST::codegen(Value *scope_struct) {
  if (not ShallCodegen)
    return const_float(0);
  Value *OperandV = Operand->codegen(scope_struct);
  if (!OperandV)
    return nullptr;
  
  
  
  std::string operand_type = Operand->GetDataTree().Type;
  if (Opcode=='-')
  {

    
    if (Operand->GetType()=="int")
      return Builder->CreateMul(ConstantInt::get(Type::getInt32Ty(*TheContext), -1), OperandV, "multmp");
    
    if (Operand->GetType()=="tensor")
      return Builder->CreateFMul(ConstantFP::get(Type::getFloatTy(*TheContext), -1),
                              OperandV, "multmp");
  }


  
  if (Opcode==tok_not||Opcode=='!') {
    if(operand_type!="bool")
      LogError(parser_struct.line, "Cannot use not with type: " + operand_type);
    return Builder->CreateNot(OperandV, "logicalnot");
  }

  if (Opcode==';')
    return OperandV;
    // return ConstantFP::get(Type::getFloatTy(*TheContext), 0);
  

  Function *F = getFunction(std::string("unary") + std::to_string(Opcode));
  if (!F) {
    auto err = LogErrorV(parser_struct.line,"Unknown unary operator.");
    std::cout << "" << Opcode << "/" << ReverseToken(Opcode) << ".\n";
    return err;
  }

  return Builder->CreateCall(F, OperandV, "unop");
}


Value *ChannelExprAST::codegen(Value *scope_struct) {
  Function *TheFunction = Builder->GetInsertBlock()->getParent();
    
  Value *initial_value = callret("channel_Create", {scope_struct, const_int(BufferSize)});

  function_values[parser_struct.function_name][Name] = initial_value;
  Allocate_On_Pointer_Stack(scope_struct, parser_struct.function_name, Name, initial_value);

  return const_float(0);
}


Value *AsyncFnPriorExprAST::codegen(Value *scope_struct) {
  int fnIndex = 1;
  while (TheModule->getFunction("__async_" + std::to_string(fnIndex)))
    fnIndex++;
  
  
  BasicBlock *CurrentBB = Builder->GetInsertBlock();


  // Create function for this async function
  llvm::Type *int8PtrTy = Type::getInt8Ty(*TheContext)->getPointerTo();

  FunctionType *asyncFunTy = FunctionType::get(
                                            int8PtrTy,
                                            {int8PtrTy},
                                            false);
                                            
  std::string functionName = "__async_" + std::to_string(fnIndex);
  
  Function *asyncFun =
      Function::Create(asyncFunTy,
                             Function::ExternalLinkage,
                             functionName,
                             TheModule.get());

  // emit EntryBB value
  BasicBlock *BB = BasicBlock::Create(*TheContext, "async_bb", asyncFun);
  Builder->SetInsertPoint(BB);

  Value *V = const_float(0.0);

  Builder->CreateRet(Constant::getNullValue(int8PtrTy));  

  Builder->SetInsertPoint(CurrentBB);

  return const_float(0.0);
}












Function *codegenAsyncFunction(std::vector<std::unique_ptr<ExprAST>> asyncBody, Value *scope_struct, \
                               Parser_Struct parser_struct, std::string async_suffix) {

  // find existing unique function name (_async_1, _async_2, _async_3 etc)
  int fnIndex = 1;
  while (TheModule->getFunction("__async_" + std::to_string(fnIndex)))
    fnIndex++;


  // Create function for this async function
  llvm::Type *int8PtrTy = Type::getInt8Ty(*TheContext)->getPointerTo();

  FunctionType *asyncFunTy = FunctionType::get(
                                            int8PtrTy,
                                            {int8PtrTy},
                                            false);
                                            
  std::string functionName = "__async_" + std::to_string(fnIndex);
  Function *asyncFun =
      Function::Create(asyncFunTy,
                             Function::ExternalLinkage,
                             functionName,
                             TheModule.get());


  std::vector<std::string> previous_scope_value_types;
  std::vector<std::string> previous_scope_value_names;
  for (auto &pair : function_values[parser_struct.function_name]) {
    if(pair.first=="QQ_stack_top")
        continue;

    std::string type;
    if (Object_toClass[parser_struct.function_name].count(pair.first)>0)
      type = "void";
    else
    {   
      type = UnmangleVec(data_typeVars[parser_struct.function_name][pair.first]);
      if(!in_str(type, primary_data_tokens))
        type="void";
    }

    call("dive_"+type, {global_str(functionName), pair.second, global_str(pair.first)});

    previous_scope_value_types.push_back(type);
    previous_scope_value_names.push_back(pair.first);
  }

  
  //Dive scope_struct
  call("scope_struct_Save_for_Async", {scope_struct, global_str(functionName)}); 

  // emit EntryBB value
  BasicBlock *BB = BasicBlock::Create(*TheContext, "async_bb", asyncFun);
  Builder->SetInsertPoint(BB);
  
  // Recover scope_struct Value * on the new function
  Value *scope_struct_copy = callret("scope_struct_Load_for_Async", {global_str(functionName)}); 

  // define body of function
  Value *V = const_float(0.0);

  Value *scope_struct_typed = Builder->CreateBitCast(
    scope_struct_copy, 
    scope_struct->getType()  // the original struct type
  );

  std::string async_scope = parser_struct.function_name + async_suffix;
  for(int i=0; i<previous_scope_value_names.size(); ++i) {
    std::string type = previous_scope_value_types[i];
    std::string var_name = previous_scope_value_names[i];

    Value *v = callret("emerge_"+type, {global_str(functionName), global_str(var_name)});

    llvm::Type *llvm_type = get_type_from_str(type);
    function_values[async_scope][var_name] = v;
  }
  function_values[async_scope]["QQ_stack_top"] = const_int(0);

  for (auto &body : asyncBody)
    V = body->codegen(scope_struct_typed);
    

  Value *stack_top_value_gep = Builder->CreateStructGEP(struct_types["scope_struct"], scope_struct_typed, 3);
  Builder->CreateStore(const_int(0), stack_top_value_gep);
  call("scope_struct_Delete", {scope_struct_typed});


  if (V)
  { 
    Builder->CreateRet(Constant::getNullValue(int8PtrTy));  

     
    std::string functionError;
    llvm::raw_string_ostream functionErrorStream(functionError);

    if (verifyFunction(*asyncFun, &functionErrorStream)) {
      functionErrorStream.flush();
      llvm::errs() << "codegen Async Function: Function verification failed:\n" << functionError << "\n";
    } 

    verifyModule(*TheModule);
    // TheModule->print(llvm::errs(), nullptr);
    return asyncFun;
  }
  
  std::cout << "ERASING ASYNC FROM PARENT" << "\n";
  asyncFun->eraseFromParent();

  return nullptr;
}



Value *SpawnExprAST::codegen(Value *scope_struct) {
  
  BasicBlock *CurrentBB = Builder->GetInsertBlock();
  Function *asyncFun = codegenAsyncFunction(std::move(Body), scope_struct, parser_struct, "_spawn");
  Builder->SetInsertPoint(CurrentBB);

  PointerType *pthreadTy = Type::getInt8Ty(*TheContext)->getPointerTo();
  Value *pthreadPtr = Builder->CreateAlloca(pthreadTy, nullptr);
  
  
  Value *voidPtrNull = Constant::getNullValue(Type::getInt8Ty(*TheContext)->getPointerTo());

  
  call("pthread_create_aux",
    {pthreadPtr,
     voidPtrNull,
     asyncFun,
     voidPtrNull}
  );

  return const_float(0);
}












Value *AsyncExprAST::codegen(Value *scope_struct) {
  if (not ShallCodegen)
    return ConstantFP::get(*TheContext, APFloat(0.0f));
  // Create/Spawn Threads

  // Value *barrier = callret("get_barrier", {const_int(1)});

  BasicBlock *CurrentBB = Builder->GetInsertBlock();
  Function *asyncFun = codegenAsyncFunction(std::move(Body), scope_struct, parser_struct, "_async");
  Builder->SetInsertPoint(CurrentBB);
  
  Function *pthread_create = TheModule->getFunction("pthread_create_aux");


  PointerType *pthreadTy = Type::getInt8Ty(*TheContext)->getPointerTo();
  Value *pthreadPtr = Builder->CreateAlloca(pthreadTy, nullptr);
  
  

  Value *voidPtrNull = Constant::getNullValue(
      Type::getInt8Ty(*TheContext)->getPointerTo());


  
  Builder->CreateCall(pthread_create,
    {pthreadPtr,
     voidPtrNull,
     asyncFun,
     voidPtrNull}
  );
  
  p2t("AsyncExpr Created join call");


  thread_pointers.push_back(pthreadPtr);

  return pthreadPtr;
}

Value *AsyncsExprAST::codegen(Value *scope_struct) {
  if (not ShallCodegen)
    return ConstantFP::get(*TheContext, APFloat(0.0f));
  
  // Create/Spawn Threads

  call("scope_struct_Store_Asyncs_Count", {scope_struct, const_int(AsyncsCount)});
  // Value *barrier = callret("get_barrier", {const_int(AsyncsCount)});

  BasicBlock *CurrentBB = Builder->GetInsertBlock();

  Function *asyncFun = codegenAsyncFunction(std::move(Body), scope_struct, parser_struct, "_asyncs");

  Builder->SetInsertPoint(CurrentBB);

  

  for(int i=0; i<AsyncsCount; i++) 
  {
    PointerType *pthreadTy = Type::getInt8Ty(*TheContext)->getPointerTo();
    Value *pthreadPtr = Builder->CreateAlloca(pthreadTy, nullptr);
    
    Value *voidPtrNull = Constant::getNullValue(
        Type::getInt8Ty(*TheContext)->getPointerTo());
    

    call("pthread_create_aux",
      {pthreadPtr,
      voidPtrNull,
      asyncFun,
      voidPtrNull}
    );
  
    p2t("AsyncExpr Created join call");


    thread_pointers.push_back(pthreadPtr);
  }

  // return pthreadPtr;
  return ConstantFP::get(*TheContext, APFloat(0.0f));
}


Value *IncThreadIdExprAST::codegen(Value *scope_struct) {
  call("scope_struct_Increment_Thread", {scope_struct});
  return ConstantFP::get(*TheContext, APFloat(0.0f));
}




Value *SplitParallelExprAST::codegen(Value *scope_struct) {
  
  Value *inner_vec = Inner_Vec->codegen(scope_struct);
  std::string type = UnmangleVec(Inner_Vec->GetDataTree());
  SetType(type);


  std::string split_fn = type + "_Split_Parallel";

  return callret(split_fn, {scope_struct, inner_vec});
}

Value *SplitStridedParallelExprAST::codegen(Value *scope_struct) {
  
  Value *inner_vec = Inner_Vec->codegen(scope_struct);
  std::string type = UnmangleVec(Inner_Vec->GetDataTree());
  SetType(type);

  std::string split_fn = type + "_Split_Strided_Parallel";

  return callret(split_fn, {scope_struct, inner_vec});
}


Value *FinishExprAST::codegen(Value *scope_struct) {
  if (not ShallCodegen)
    return const_float(0.0f);
  
  // Function *TheFunction = Builder->GetInsertBlock()->getParent();
  // std::string functionName = TheFunction->getName().str();

  for (int i=0; i < Bodies.size(); i++)
    Bodies[i]->codegen(scope_struct);
  

  for (Value *pthreadPtr : thread_pointers)
  {
    Value *pthread = Builder->CreateLoad(int8PtrTy, pthreadPtr);
    call("pthread_join_aux", {pthread});
  }
  thread_pointers.clear();


  call("scope_struct_Reset_Threads", {scope_struct});
  
  return const_float(0);
}


Value *LockExprAST::codegen(Value *scope_struct){
  
  Builder->CreateCall(TheModule->getFunction("LockMutex"), {Builder->CreateGlobalString(Name)});

  for (auto &body : Bodies)
    body->codegen(scope_struct);

  Builder->CreateCall(TheModule->getFunction("UnlockMutex"), {Builder->CreateGlobalString(Name)});

  return ConstantFP::get(*TheContext, APFloat(0.0f));
}




Value *MainExprAST::codegen(Value *scope_struct) {
  if (not ShallCodegen)
    return const_float(0);
  
  Function *TheFunction = Builder->GetInsertBlock()->getParent();
  std::string functionName = TheFunction->getName().str();



  for (const auto &body : Bodies) {

    body->codegen(scope_struct);
    if (!body)
      return const_float(1);
  }

  
  return const_float(0);
}


Value *NoGradExprAST::codegen(Value *scope_struct){
  
  Builder->CreateCall(TheModule->getFunction("set_scope_has_grad"), {scope_struct, ConstantInt::get(Type::getInt32Ty(*TheContext), 0)});
  for (auto &body : Bodies)
    body->codegen(scope_struct);

  
  return ConstantFP::get(*TheContext, APFloat(0.0f));
}



Value *VariableListExprAST::codegen(Value *scope_struct) {
  
  std::cout << "Variable list expr" << ".\n";

  return ConstantFP::get(*TheContext, APFloat(0.0f));
}



Value *RetExprAST::codegen(Value *scope_struct) {
  seen_var_attr=true;

    
  if(Vars.size()==1)
  { 
    // p2t("Reached if");
    Value *ret = Vars[0]->codegen(scope_struct);
    std::string type = Vars[0]->GetDataTree().Type;

    seen_var_attr=false;
    Builder->CreateRet(ret);
    // Function *TheFunction = Builder->GetInsertBlock()->getParent();
    // verifyFunction(*TheFunction);
    // TheModule->print(llvm::errs(), nullptr);
    return ret;
  }
  
  std::vector<Value *> values = {scope_struct};
  for (int i=0; i<Vars.size(); i++)
  {
    Value *value = Vars[i]->codegen(scope_struct); 
    std::string type = Vars[i]->GetDataTree().Type;


    values.push_back(global_str(type));
    values.push_back(value);
  }
  values.push_back(global_str("TERMINATE_VARARG"));


  seen_var_attr=false;
  Value *ret = callret("list_New", values);
  Builder->CreateRet(ret);

  return ret;
}



Data_Tree NewTupleExprAST::GetDataTree(bool from_assignment) {
  if(!data_type.empty)
    return data_type;

  data_type.Type = "tuple";
  for (int i=0; i<Values.size(); i++)
  {
    std::string type = Values[i]->GetType();
    data_type.Nested_Data.push_back(Data_Tree(type));
  }
  data_type.empty=false;

  return data_type;
}

Value *NewTupleExprAST::codegen(Value *scope_struct) {
  if (not ShallCodegen)
    return ConstantFP::get(*TheContext, APFloat(0.0f));

  std::vector<Value *> values;

  GetDataTree();

  values.push_back(scope_struct);


  seen_var_attr = true;
  bool is_type=true;
  for (int i=0; i<Values.size(); i++)
  {
    Value *value = Values[i]->codegen(scope_struct);
    std::string type = Values[i]->GetType();

    
    if (!is_type)
    {
      if(!in_str(type, primary_data_tokens))
      {
        std::string copy_fn = type + "_Copy";
        
        Function *F = TheModule->getFunction(copy_fn);
        if (F)
          value = callret(copy_fn, {scope_struct, value});
      }
      is_type=true;
    } else
      is_type=false;
    values.push_back(value);
  }

  seen_var_attr = false;

  return callret("list_New", values);
}





Data_Tree NewVecExprAST::GetDataTree(bool from_assignment) {
  if(!data_type.empty)
    return data_type;

  data_type.Type = "tuple";
  for (int i=1; i<Values.size(); i=i+2) {
    Data_Tree type = Values[i]->GetDataTree();
    data_type.Nested_Data.push_back(type);
  }
  data_type.empty=false;

  return data_type;
}

Value *NewVecExprAST::codegen(Value *scope_struct) {
  if (not ShallCodegen)
    return ConstantFP::get(*TheContext, APFloat(0.0f));

  std::vector<Value *> values;

  values.push_back(scope_struct);

  seen_var_attr = true;
  bool is_type=true;
  for (int i=0; i<Values.size(); i++)
  {
    std::string type = Values[i]->GetType();
    Value *value = Values[i]->codegen(scope_struct);
    if (!is_type)
    {
      // std::cout << "VALUE TYPE IS: " << type << ".\n";
      if(!in_str(type, primary_data_tokens))
      {

        std::string copy_fn = type + "_Copy";
        
        Function *F = TheModule->getFunction(copy_fn);
        if (F)
          value = callret(copy_fn, {scope_struct, value});
      }
      is_type=true;
    } else
      is_type=false;
    values.push_back(value);
  }

  seen_var_attr = false;

  // std::cout << "Call list_New" << ".\n";
  return callret("list_New", values);
}


Value *NewDictExprAST::codegen(Value *scope_struct) {
  if (not ShallCodegen)
    return ConstantFP::get(*TheContext, APFloat(0.0f));

  std::vector<Value *> values;

  values.push_back(scope_struct);

  seen_var_attr = true;
  
  for (int i=0; i<Values.size()-1; i++)
  {
    std::string key_type = Keys[i]->GetType();
    std::string type = Values[i]->GetType();
     
    if (key_type!="str")
    {
      LogError(parser_struct.line, "Dictionary key must be of type string");
      return const_float(0);
    } 
    Value *key = Keys[i]->codegen(scope_struct);
    values.push_back(key);

    auto element_type = std::make_unique<StringExprAST>(type);
    values.push_back(element_type->codegen(scope_struct));

    Value *value = Values[i]->codegen(scope_struct);


    if(!in_str(type, primary_data_tokens))
    {

      std::string copy_fn = type + "_Copy";
      
      Function *F = TheModule->getFunction(copy_fn);
      if (F)
        value = callret(copy_fn, {scope_struct, value});
    }

    values.push_back(value);
  }


  Value *value = Values[Values.size()-1]->codegen(scope_struct);
  values.push_back(value);

  seen_var_attr = false;

  return callret("dict_New", values);
}







Value *ObjectExprAST::codegen(Value *scope_struct) {
  if (not ShallCodegen)
    return ConstantFP::get(*TheContext, APFloat(0.0f));


  Function *TheFunction = Builder->GetInsertBlock()->getParent();


  // Register all variables and emit their initializer.


  Value *previous_obj;
  bool has_init=false;
  for (unsigned i = 0, e = VarNames.size(); i != e; ++i)
  {
    const std::string &VarName = VarNames[i].first;
    ExprAST *Init = VarNames[i].second.get();


    Value *var_name, *obj_name;
    
    
    if(!isSelf&&!isAttribute)
    {   

      Value *ptr;
      if (Init==nullptr) // no attribution
        ptr = callret("allocate_void", {scope_struct, const_int(Size), global_str(ClassName)});
      else
        ptr = Init->codegen(scope_struct);
      Allocate_On_Pointer_Stack(scope_struct, parser_struct.function_name, VarName, ptr);
      
      function_values[parser_struct.function_name][VarName] = ptr;

      
      if (HasInit[i]) { // callee init
        if(!has_init)
            previous_obj = get_scope_obj(scope_struct);
        has_init = true;
        set_scope_obj(scope_struct, ptr);
        
        int arg_type_check_offset = 1;
        std::vector<Value *> ArgsV = {scope_struct};

        std::string Callee = ClassName + "___init__";
        ArgsV = Codegen_Argument_List(parser_struct, std::move(ArgsV), std::move(Args[i]), scope_struct, \
                                      Callee, false, arg_type_check_offset); 
        Set_Stack_Top(scope_struct, parser_struct.function_name);
        call(Callee, ArgsV);
      }
    }      
  }

  if(has_init) 
    set_scope_obj(scope_struct, previous_obj);

  return const_float(0.0f);
}





Value *NewExprAST::codegen(Value *scope_struct) {
    Value *nullPtr = ConstantPointerNull::get(
        cast<PointerType>(int8PtrTy)
    );

    std::vector<Value *> ArgsV = {scope_struct};
    
    if (Args.size()>0)
        ArgsV = Codegen_Argument_List(parser_struct, std::move(ArgsV), std::move(Args), scope_struct, \
                                      Callee, true, 1);

    if(!Check_ArgsV_Count(Callee, ArgsV, parser_struct, 1))
        return const_float(0);

    return callret(Callee, ArgsV);
}






Function *PrototypeAST::codegen() {
  if (not ShallCodegen)
    return nullptr;
  // Make the function type:  float(float,float) etc.
  
  std::vector<Type *> types;

  for (auto &type : Types)
  {
    if (type=="f"||type=="float")
      types.push_back(floatTy);
    else if(type=="i"||type=="int")
      types.push_back(intTy);
    else if(type=="b"||type=="bool")
      types.push_back(boolTy);
    else
      types.push_back(int8PtrTy);
  }
  
  FunctionType *FT;
  if (Return_Type=="float")
    FT = FunctionType::get(floatTy, types, false);
  else if (Return_Type=="int")
    FT = FunctionType::get(intTy, types, false);
  else if (Return_Type=="bool")
    FT = FunctionType::get(boolTy, types, false);
  else
    FT = FunctionType::get(int8PtrTy, types, false); 
  

  Function *F =
      Function::Create(FT, Function::ExternalLinkage, Name, TheModule.get());

  // Set names for all arguments.
  unsigned Idx = 0;
  for (auto &Arg : F->args()) {
    Arg.setName(Args[Idx++]);
    // if (!Arg.getType()->isPointerTy()) {
    //     continue;
    // }
    // // scope_struct
    // if (Arg.getName() == "scope_struct") {
    //     LogBlue("No alias for scope_struct");
    //     Arg.addAttr(Attribute::ReadOnly); 
    // }
    // Arg.addAttr(Attribute::NoAlias);
    // Arg.addAttr(Attribute::NonNull);
  }
  return F;
}





Value *NameableExprAST::codegen(Value *scope_struct) {}
Value *EmptyStrExprAST::codegen(Value *scope_struct) {}


std::string Get_Nested_Name(std::vector<std::string> expressions_string_vec, Parser_Struct parser_struct, bool to_last) {

  int last = expressions_string_vec.size();
  if (!to_last)
    last--;


  std::string _class=parser_struct.function_name;

  int i=0;
  if(expressions_string_vec[i]=="self") {
    _class = parser_struct.class_name; 
    i++;
  } else if (Object_toClass[_class].count(expressions_string_vec[i])==0) {
    _class = data_typeVars[_class][expressions_string_vec[i]].Nested_Data[0].Type;
    i++;
  }


  for(; i<last; ++i)
  {

    std::string next_class = Object_toClass[_class][expressions_string_vec[i]].Type;

    if (next_class=="")
      return _class;
    _class = next_class;
  }

  return _class;
}



int Get_Nested_Class_Size(std::vector<std::string> expressions_string_vec, Parser_Struct parser_struct) {

  int last = expressions_string_vec.size();


  std::string _class=parser_struct.function_name; 

  int i=0;
  if(expressions_string_vec[i]=="self") {
    _class = parser_struct.class_name; 
    i++;
  }

  for(; i<last; ++i)
  {
    _class = Object_toClass[_class][expressions_string_vec[i]].Type;
  }

  return ClassSize[_class];
}



int Get_Object_Offset(std::vector<std::string> expressions_string_vec, Parser_Struct parser_struct) {

  int last = expressions_string_vec.size()-1;


  std::string _class=parser_struct.function_name; 

  int i=0;
  if(expressions_string_vec[i]=="self") {
    _class = parser_struct.class_name; 
    i++;
  }


  for(; i<last; ++i)
  {
    _class = Object_toClass[_class][expressions_string_vec[i]].Type;
  }
  // p2t("Offset of " + expressions_string_vec[last] + " is " + std::to_string(ClassVariables[_class][expressions_string_vec[last]]));

  return ClassVariables[_class][expressions_string_vec[last]];
}

Value *SelfExprAST::codegen(Value *scope_struct) {
  return get_scope_obj(scope_struct);
}

Value *NestedStrExprAST::codegen(Value *scope_struct) {  

  if(skip)
    return Inner_Expr->codegen(scope_struct);
 
  
  int offset;
  if(Inner_Expr->Name=="self")
  {
    Value *obj_ptr = Inner_Expr->codegen(scope_struct);
    
    offset = ClassVariables[parser_struct.class_name][Name];

    obj_ptr = callret("offset_object_ptr", {obj_ptr, const_int(offset)});

    std::string _type = typeVars[parser_struct.class_name][Name];
    if(_type!="int"&&_type!="float" && (!IsLeaf||Load_Last))
    {
      obj_ptr = callret("object_Load_slot", {obj_ptr});
    }

    return obj_ptr;

  } else if (height>1) { 
    
    Value *obj_ptr = Inner_Expr->codegen(scope_struct);

 
    offset = Get_Object_Offset(Expr_String, parser_struct);
  
    obj_ptr = callret("offset_object_ptr", {obj_ptr, const_int(offset)});


    std::string fn_name = Get_Nested_Name(Expr_String, parser_struct, false);
    std::string _type = typeVars[fn_name][Name];

    
    if(_type!="int"&&_type!="float" && (!IsLeaf||Load_Last))
      obj_ptr = callret("object_Load_slot", {obj_ptr});
    

    return obj_ptr;

  } else if (height==1)
  {
    std::string var_type = typeVars[parser_struct.function_name][Name];
    // std::cout << "----LOADING HEIGHT==1 ALLOCA " << Name << " OF TYPE " << var_type << " AT FUNCTION " << parser_struct.function_name << ".\n"; 
    return load_alloca(Name, var_type, parser_struct.function_name);    
  }
  else {
    LogError(parser_struct.line, "uncaught");
  }
}



std::string NestedVectorIdxExprAST::GetType(bool from_assignment) {  
  if (from_assignment)
    return Type;
  return Extract_List_Prefix(Type);
}


Value *NestedVectorIdxExprAST::codegen(Value *scope_struct) {  
  if(skip)
    return Inner_Expr->codegen(scope_struct);


  Value *obj_ptr = Inner_Expr->codegen(scope_struct);
 
  Value *idx = Idx_Calc_Codegen(Type, obj_ptr, Idx, scope_struct);


  std::string homogeneous_type = Extract_List_Suffix(Type);
  std::string type = Extract_List_Prefix(Type);

  if (homogeneous_type=="dict") {
    return callret("dict_Query", {scope_struct, obj_ptr, idx});
  }

  if (!Idx->IsSlice)
    return callret(homogeneous_type+"_Idx", {scope_struct, obj_ptr, idx});
  else
  {
    Value *ret = callret(homogeneous_type+"_Slice", {scope_struct, obj_ptr, idx});
    call("Delete_Ptr", {idx});
    return ret;
  }  
}






Data_Tree NestedVariableExprAST::GetDataTree(bool from_assignment) {
  return data_type;
}

Value *NestedVariableExprAST::codegen(Value *scope_struct) {
  // std::cout << "Nested Variable Expr" << ".\n";
  // Print_Names_Str(Inner_Expr->Expr_String);

  Value *ptr = Inner_Expr->codegen(scope_struct);

  if (Load_Val&&(Type=="float"||Type=="int"))
    return callret("object_Load_"+Type, {ptr});
  
  return ptr;
}



Value *NameableLLVMIRCall::codegen(Value *scope_struct) {  
  int arg_type_check_offset=1, target_args_size=Args.size();
  bool is_nsk_fn = in_str(Callee, native_methods);

  
  // std::vector<Value*> ArgsV = {scope_struct};



  if (Callee=="pow"&&target_args_size!=2) {
    LogError(parser_struct.line, "Function pow expected 2 arguments, but got " + std::to_string(target_args_size));
    return const_float(0);
  }
  if (Callee=="sqrt"&&target_args_size!=1) {
    LogError(parser_struct.line, "Function sqrt expected 1 argument, but got " + std::to_string(target_args_size));
    return const_float(0);
  }


  if (ReturnType=="")
    GetDataTree();



  
  
  Value *ret;
  if(Callee=="pow") { // pow
    Value *x_value = Args[0]->codegen(scope_struct);
    if (Args[0]->GetDataTree().Type=="int")
      x_value = Builder->CreateSIToFP(x_value, Type::getFloatTy(*TheContext), "lfp");

    Value *exponent_value = Args[1]->codegen(scope_struct);
    if (Args[1]->GetDataTree().Type=="int")
      exponent_value = Builder->CreateSIToFP(exponent_value, Type::getFloatTy(*TheContext), "lfp");
      
    ret = Builder->CreateBinaryIntrinsic(Intrinsic::pow, x_value, exponent_value);
  } else if (Callee=="sqrt") { // sqrt
    Value *x_value = Args[0]->codegen(scope_struct);
    if (Args[0]->GetDataTree().Type=="int")
      x_value = Builder->CreateSIToFP(x_value, Type::getFloatTy(*TheContext), "lfp");

    ret = Builder->CreateUnaryIntrinsic(Intrinsic::sqrt, x_value);
  } else
    LogError(-1, "LLVM IR Function " + Callee + " not implemented");



  if(ReturnType=="void_ptr")
    LogError(-1, "return " + Callee);  

  return ret;
}



Value *NestedCallExprAST::codegen(Value *scope_struct) {
  return const_float(0);
}


Value *NameableRoot::codegen(Value *scope_struct) {

  return const_float(0);
}




Value *Nameable::codegen(Value *scope_struct) {
  Data_Tree dt = GetDataTree();
  std::string type = dt.Type;

  if(Depth==1)
  {
    if(Name=="self")
      return get_scope_obj(scope_struct); 

    if (function_values[parser_struct.function_name].count(Name)==0) {
        LogError(parser_struct.line, "Variable " + Name + " not found.");
        return const_float(0.0f);
    }
    return function_values[parser_struct.function_name][Name];
  }

  std::string scope = Inner->GetDataTree().Type;
  Value *obj_ptr = Inner->codegen(scope_struct);

  int offset = ClassVariables[scope][Name];

  
  obj_ptr = callret("offset_object_ptr", {obj_ptr, const_int(offset)});


  if(!in_str(type, primary_data_tokens) && (!IsLeaf||Load_Last))
    obj_ptr = callret("object_Load_slot", {obj_ptr});
  if (Load_Last&&in_str(type, primary_data_tokens))
    return callret("object_Load_"+type, {obj_ptr});


  return obj_ptr;
}









Value *NameableIdx::codegen(Value *scope_struct) {
  Data_Tree inner_dt = Inner->GetDataTree();


  std::string compound_type = UnmangleVec(inner_dt);
  std::string type;
  if (compound_type=="tuple") {
    if (IntExprAST *expr = dynamic_cast<IntExprAST*>(Idx->Idxs[0].get())) {
      int idx = expr->Val;
      type = inner_dt.Nested_Data[idx].Type;
    }
  }  
  else if(in_str(compound_type, compound_tokens)||ends_with(compound_type, "_vec")) {
    if (inner_dt.Nested_Data.size()==0) {
        if(compound_type=="list")
            type = "any";
        else
            return LogErrorV(parser_struct.line, "Missing " + compound_type + " nested type.");
    } else
        type = inner_dt.Nested_Data[0].Type;
  }
  else
    type = compound_type;


  // std::cout << "\nNameable idx inner type is: " << type << ".\n";


  Value *loaded_var = Inner->codegen(scope_struct);
  Value *idx = Idx_Calc_Codegen(compound_type, loaded_var, Idx, scope_struct);
  


  if (compound_type == "map") {
    Value *query = idx;

    StructType *st = struct_types["map"];
    StructType *st_node = struct_types["map_node"];

    Value *nullPtr = ConstantPointerNull::get(
        cast<PointerType>(int8PtrTy)
    );

    std::string key_type = inner_dt.Nested_Data[0].Type;
    std::string value_type = inner_dt.Nested_Data[1].Type;
    if (query->getType()==intTy&&key_type=="float")
        query = Builder->CreateSIToFP(query, floatTy);
    else if(Idx->GetDataTree().Type!=key_type)
        return LogErrorV(parser_struct.line, "Querying " + key_type + " map with " + Idx->GetDataTree().Type);


    Value *capacity_gep = Builder->CreateStructGEP(st, loaded_var, 1);
    Value *map_capacity = Builder->CreateLoad(intTy, capacity_gep);

    Value *nodes_gep = Builder->CreateStructGEP(st, loaded_var, 5);
    Value *nodes = Builder->CreateLoad(int8PtrTy->getPointerTo(), nodes_gep);

    Function *TheFunction = Builder->GetInsertBlock()->getParent();
    BasicBlock *LoadValBB = BasicBlock::Create(*TheContext, "map.get_val.bb", TheFunction);
    BasicBlock *LoopBB = BasicBlock::Create(*TheContext, "map.loop.bb", TheFunction);
    BasicBlock *NextPtrBB = BasicBlock::Create(*TheContext, "map.next_ptr.bb", TheFunction);
    BasicBlock *FromNullBB = BasicBlock::Create(*TheContext, "map.from_null.bb", TheFunction);
    BasicBlock *GetKeyBB = BasicBlock::Create(*TheContext, "map.get_key.bb", TheFunction);


    Value *query_hash;
    if (key_type=="str")
        query_hash = str_llvm_hash(query, TheFunction);        
    if (key_type=="float")
        query_hash = float_llvm_hash(query);
    if (key_type=="int")
        query_hash = query;
    Value *hash_pos = Builder->CreateURem(query_hash, map_capacity);


    Value *node_gep = Builder->CreateGEP(int8PtrTy, nodes, hash_pos);
    Value *node = Builder->CreateLoad(int8PtrTy, node_gep);
    BasicBlock *curBB = Builder->GetInsertBlock();


    // Verify node loop
    Builder->CreateBr(LoopBB);

    Builder->SetInsertPoint(LoopBB);
    PHINode *cur_node = Builder->CreatePHI(int8PtrTy, 1);
    cur_node->addIncoming(node, curBB);

    Value *IsNull = Builder->CreateICmpEQ(cur_node, nullPtr);
    Builder->CreateCondBr(IsNull, FromNullBB, GetKeyBB);


    // Get Key
    Builder->SetInsertPoint(GetKeyBB);
    Value *key_gep = Builder->CreateStructGEP(st_node, cur_node, 0);
    Value *keyCond, *key;
    if (key_type=="int") {
        Value *key_void_ptr = Builder->CreateLoad(int8PtrTy, key_gep);
        Value *key_int_ptr = Builder->CreateBitCast(key_void_ptr, intTy->getPointerTo());
        key = Builder->CreateLoad(intTy, key_int_ptr);
        keyCond = Builder->CreateICmpEQ(key, query);
    } else if (key_type=="float") {
        Value *key_void_ptr = Builder->CreateLoad(int8PtrTy, key_gep);
        Value *key_float_ptr = Builder->CreateBitCast(key_void_ptr, floatTy->getPointerTo());
        key = Builder->CreateLoad(floatTy, key_float_ptr);
        keyCond = Builder->CreateFCmpUEQ(key, query);
    } else {
        key = Builder->CreateLoad(int8PtrTy, key_gep);
        keyCond = callret("strcmp", {key, query});
        keyCond = Builder->CreateICmpEQ(keyCond, const_int(0));
    }
    Builder->CreateCondBr(keyCond, LoadValBB, NextPtrBB);
 
    // Get next node
    Builder->SetInsertPoint(NextPtrBB);
    Value *next_node_gep = Builder->CreateStructGEP(st_node, cur_node, 2);
    Value *next_node = Builder->CreateLoad(int8PtrTy, next_node_gep);
    cur_node->addIncoming(next_node, NextPtrBB);
    Builder->CreateBr(LoopBB);

    // From Null
    Builder->SetInsertPoint(FromNullBB);
    call("map_bad_key_"+key_type, {scope_struct, query});
    Builder->CreateUnreachable();

    // Get node value
    Builder->SetInsertPoint(LoadValBB);

    Value *value_gep = Builder->CreateStructGEP(st_node, cur_node, 1);
    Value *value;
    if (value_type=="int") {
        Value *value_void_ptr = Builder->CreateLoad(int8PtrTy, value_gep);
        Value *value_int_ptr = Builder->CreateBitCast(value_void_ptr, intTy->getPointerTo());
        value = Builder->CreateLoad(intTy, value_int_ptr);
    } else if (value_type=="float") {
        Value *value_void_ptr = Builder->CreateLoad(int8PtrTy, value_gep);
        Value *value_float_ptr = Builder->CreateBitCast(value_void_ptr, floatTy->getPointerTo());
        value = Builder->CreateLoad(floatTy, value_float_ptr);
    } else
        value = Builder->CreateLoad(int8PtrTy, value_gep);

    return value;
  }



  if(Idx->idx_slice_or_query=="query") {
    Value *ret_val = callret(compound_type+"_Query", {scope_struct, loaded_var, idx});
    return ret_val;
  }
  
  if (!Idx->IsSlice) {
    std::string idx_fn = compound_type + "_Idx";
  
    
    Value *ret_val;
    if (TheModule->getFunction(idx_fn))
      ret_val = callret(idx_fn, {scope_struct, loaded_var, idx});
    else {
        llvm::Type *elemTy;
        std::string elem_type;
        if(compound_type=="array")  {
          elem_type = Inner->GetDataTree().Nested_Data[0].Type;
          elemTy = get_type_from_str(elem_type); 
        }
        else {
            elem_type = remove_suffix(compound_type, "_vec");
            elemTy = get_type_from_str(elem_type); 
        }

        Check_Is_Array_Inbounds(parser_struct, loaded_var, idx);

        Value *vec = Load_Array(parser_struct.function_name, loaded_var);

        llvm::Type *idxTy;
        if (elem_type=="int")
            idxTy = intTy;
        else if (elem_type=="float")
            idxTy = floatTy;
        else if (elem_type=="bool")
            idxTy = boolTy;
        else
            idxTy = int8PtrTy; 

        Value *element = Builder->CreateGEP(idxTy, vec, idx);
        ret_val = Builder->CreateLoad(elemTy, element, "elem"); 
    }

    if(!(ends_with(compound_type,"_vec"))&&(type=="float"||type=="int"||type=="bool") && compound_type!="array")
      ret_val = callret("to_"+type, {scope_struct, ret_val});
    
    return ret_val;
  } else {
    std::string slice_fn = compound_type + "_Slice";    
    Value *ret =  callret(slice_fn, {scope_struct, loaded_var, idx});
    call("Delete_Ptr", {idx});
    return ret;
  }
}


Value *PositionalArgExprAST::codegen(Value *scope_struct) {
    return Inner->codegen(scope_struct);
}


inline bool Check_Args_Count(const std::string &Callee, std::vector<std::unique_ptr<ExprAST>> &Args,
                             int target_args_size, Parser_Struct parser_struct) {
  Function *CalleeF;
  CalleeF = getFunction(Callee);
  
  if (Callee=="list_append")
    target_args_size++;
  if (!CalleeF)
  {
    std::string _error = "The referenced function "+ Callee +" was not yet declared.";
    LogErrorV(parser_struct.line, _error);
    return false;
  }
  
  // If argument mismatch error.
  if ((CalleeF->arg_size()) != target_args_size && !in_str(Callee, vararg_methods))
  {
    // std::cout << "CalleeF->arg_size() " << std::to_string(CalleeF->arg_size()) << " target_args_size " << std::to_string(target_args_size) << "\n";
    std::string _error = "Incorrect parameters used on function " + Callee + " call.\n\t    Expected " +  std::to_string(CalleeF->arg_size()-1) + " arguments, got " + std::to_string(target_args_size-1);
    LogErrorV(parser_struct.line, _error);
    return false;
  }
  return true;
}

Value *NameableAppend::codegen(Value *scope_struct) {  

    Value *loaded_var = Inner->codegen(scope_struct);

    Value *appended_val = Args[0]->codegen(scope_struct);

    std::string elem_type = inner_dt.Nested_Data[0].Type;


    if (inner_dt.Type=="array")
    {
        StructType *st = struct_types["scope_struct"];
        llvm::Type *elemTy;
        Value *vec = Load_Array(parser_struct.function_name, loaded_var);

        elemTy = get_type_from_str(elem_type); 

        Value *vsize_gep = Builder->CreateStructGEP(st, loaded_var, 0);
        Value *vsize = Builder->CreateLoad(intTy, vsize_gep);

        Value *size_gep = Builder->CreateStructGEP(st, loaded_var, 1);
        Value *size = Builder->CreateLoad(intTy, size_gep);
         

        Function *TheFunction = Builder->GetInsertBlock()->getParent();
        BasicBlock *good_sizeBB = BasicBlock::Create(*TheContext, "array.append.ok_size", TheFunction);
        BasicBlock *bad_sizeBB = BasicBlock::Create(*TheContext, "array.append.bad_size", TheFunction);
        BasicBlock *postBB = BasicBlock::Create(*TheContext, "array.append.post", TheFunction);

        Value *Cond = Builder->CreateICmpSLT(vsize, size);
        Builder->CreateCondBr(Cond, good_sizeBB, bad_sizeBB);


        //good size
        Builder->SetInsertPoint(good_sizeBB);

        
        Value *elem_gep = Builder->CreateGEP(elemTy, vec, vsize); 
        Builder->CreateStore(appended_val, elem_gep);
        Value *next_vsize = Builder->CreateAdd(vsize, const_int(1));
        Builder->CreateStore(next_vsize, vsize_gep);
 
        Builder->CreateBr(postBB);


        //bad size (thus double array size)
        Builder->SetInsertPoint(bad_sizeBB);

        Value *new_size = Builder->CreateMul(size, const_int(2));
        call("array_double_size", {loaded_var, new_size}); //does vsize++
        // if(parser_struct.loop_depth==0)
        //     Cache_Array(parser_struct.function_name, loaded_var);
        
        elem_gep = Builder->CreateGEP(elemTy, vec, vsize); 
        Builder->CreateStore(appended_val, elem_gep);
        
        Builder->CreateBr(postBB);


        //post
        Builder->SetInsertPoint(postBB); 
    }


    return const_float(0.0f);
}

Value *NameableCall::codegen(Value *scope_struct) {  

  Function *TheFunction = Builder->GetInsertBlock()->getParent();

  int arg_type_check_offset=1, target_args_size=Args.size()+1;
  bool is_nsk_fn = in_str(Callee, native_methods);

  Value *previous_obj, *previous_stack_top;
  
  if (!is_nsk_fn||Callee=="scope_struct_Sweep") {
      // Prevents the case in which it allocates a slot for an argument
      previous_stack_top = function_values[parser_struct.function_name]["QQ_stack_top"];
      Set_Stack_Top(scope_struct, parser_struct.function_name);
  }



  std::vector<Value*> ArgsV = {scope_struct};


  bool has_obj_overwrite = (Depth>1&&!FromLib&&!is_nsk_fn);
  if(Depth>1&&!FromLib) {
    if (ends_with(Callee, "__init__")&&isSelf)
    {
      Inner->IsLeaf=true;
      Inner->Load_Last=false; // inhibits Load_slot
    }

    Value *obj_ptr = Inner->codegen(scope_struct);
   
    if(!is_nsk_fn)
    {
        if(ends_with(Callee, "__init__")&&isSelf) // mallocs an object inside another
        {
            std::string obj_class = Inner->GetDataTree().Type;

            int size = ClassSize[obj_class];


            Value *new_ptr = callret("allocate_void", {scope_struct, const_int(size), global_str(obj_class)});
            // LogBlue("Add " + obj_class + " as root.");
            call("tie_object_to_object", {obj_ptr, new_ptr});

            obj_ptr = new_ptr;

        }
        previous_obj = swap_scope_obj(scope_struct, obj_ptr); 
    }
    else{
        ArgsV.push_back(obj_ptr);
        arg_type_check_offset++;
        target_args_size++;

        BasicBlock *GotNullBB = BasicBlock::Create(*TheContext, "nested_call.bad.bb", TheFunction);
        BasicBlock *AfterBB = BasicBlock::Create(*TheContext, "nested_call.ok.bb", TheFunction);

        Value *nullPtr = ConstantPointerNull::get(
            cast<PointerType>(int8PtrTy)
        );

        Builder->CreateCondBr(Builder->CreateICmpEQ(obj_ptr, nullPtr), GotNullBB, AfterBB);  
        
        Builder->SetInsertPoint(GotNullBB);
        call("LogErrorCall", {const_int(parser_struct.line), global_str("Could not call " + Callee + ", got a nullptr as object")});
        Builder->CreateUnreachable();

        Builder->SetInsertPoint(AfterBB);
    }
  }
  

  // if(!Check_Args_Count(Callee, Args, target_args_size, parser_struct))
  //   return const_float(0);


  if (ReturnType=="")
    GetDataTree();

  ArgsV = Codegen_Argument_List(parser_struct, std::move(ArgsV), std::move(Args), scope_struct,\
                                Callee, is_nsk_fn, arg_type_check_offset);

  
  Value *ret = callret(Callee, ArgsV);

  if (has_obj_overwrite) // Retrieve previous object
    set_scope_obj(scope_struct, previous_obj);
  if (!is_nsk_fn&&function_values["QQ_stack_top"]["QQ_stack_top"]!=previous_stack_top) {
      // stack_top_value = previous_stack_top;
      function_values[parser_struct.function_name]["QQ_stack_top"] = previous_stack_top;
      Set_Stack_Top(scope_struct, parser_struct.function_name);
  }

  // if(ReturnType=="")
  //   when it is void

  if(ReturnType=="void_ptr")
    LogError(-1, "return " + Callee);


  return ret;
}









Value *CallExprAST::codegen(Value *scope_struct) {
  return const_float(0);
}



