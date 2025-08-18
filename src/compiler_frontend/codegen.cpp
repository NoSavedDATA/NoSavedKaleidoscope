#include "llvm/IR/Value.h"
#include "llvm/IR/Verifier.h"

#include <string>
#include <map>
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
std::string current_codegen_function;


std::vector<Value *> thread_pointers;

PointerType *floatPtrTy, *int8PtrTy;

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
    else
      load_type = int8PtrTy;

    // std::cout << "ALLOCA LOAD OF " << from_function << "/" << name << " type " << type << ".\n";
    AllocaInst *alloca = function_allocas[from_function][name];

    return Builder->CreateLoad(load_type, alloca, name.c_str());
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

Value *IndexExprAST::codegen(Value *scope_struct) {
  
  return const_float(0);
}


inline Value *Idx_Calc_Codegen(std::string type, Value *vec, const std::unique_ptr<IndexExprAST> &idxs, Value *scope_struct)
{
  std::vector<Value *> idxs_values;

  idxs_values.push_back(vec); // e.g, tensor uses its dims as a support to calculcate the index

  for (int i=0; i<idxs->size(); i++)
  {
    Value *idx = idxs->Idxs[i]->codegen(scope_struct);

    if (i==0 && (idxs->Idxs[i]->GetType()=="str")) // dict query
      return idx;
    
    idxs_values.push_back(idxs->Idxs[i]->codegen(scope_struct));
  }


  if (!idxs->IsSlice)
  {
    std::string fn = type+"_CalculateIdx";

    Function *F = TheModule->getFunction(fn);
    if (F)
      return callret(fn, idxs_values);
    else
      return callret("__idx__", idxs_values);
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

  // if (alloca_type==int8PtrTy)
  //   Builder->CreateStore(callret("nullptr_get", {}), alloca);

  return alloca;
}

Value *Get_Object_Value(NameSolverAST *name_solver, Parser_Struct parser_struct)
{

  std::string object_name = std::get<0>(name_solver->Names[0]);
  AllocaInst *object_alloca = function_allocas[parser_struct.function_name][object_name];
  Value *obj = Builder->CreateLoad(int8PtrTy, object_alloca);
  return obj;
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

Value *StringExprAST::codegen(Value *scope_struct) {
  if (not ShallCodegen)
    return ConstantFP::get(*TheContext, APFloat(0.0f));
  SetName(Val);
  Value *_str = callret("CopyString", {global_str(Val)});
  call("MarkToSweep_Mark", {scope_struct, _str, global_str("str")});
  return _str;
}

Value *NullPtrExprAST::codegen(Value *scope_struct) {
 
  return callret("nullptr_get", {});
}

// Create Float Var
Value *VarExprAST::codegen(Value *scope_struct) {
  if (not ShallCodegen)
    return ConstantFP::get(*TheContext, APFloat(0.0f));

  return ConstantFP::get(*TheContext, APFloat(0.0));
}





llvm::Type *get_type_from_str(std::string type)
{
  llvm::Type *llvm_type;
  if (type=="float")
    llvm_type = Type::getFloatTy(*TheContext);
  else if (type=="int")
    llvm_type = Type::getInt32Ty(*TheContext);
  else
    llvm_type = int8PtrTy;
  return llvm_type;
}




bool Check_Is_Compatible_Type(std::string LType, const std::unique_ptr<ExprAST> &RHS, Parser_Struct parser_struct) {
  std::string RType = RHS->GetType();


  // LogBlue("Attributing " + RType + " to " + LType);
  
  // RType==null -> !primary data
  // if(RType=="nullptr")

  if (LType=="float"&&RType=="int")
    return true;

  if((RType=="nullptr")&&!in_str(LType, {"float", "int", "str", "bool"}))
    return true;
  
  if(begins_with(LType, "unknown_")||begins_with(RType,"unknown_"))
    return true;
  
  if (LType!=RType)
  { 
    if (dynamic_cast<NestedVectorIdxExprAST *>(RHS.get()) || dynamic_cast<VecIdxExprAST *>(RHS.get()))
    {
      RType = Extract_List_Prefix(RType);
      RType = remove_suffix(RType, "_vec");
    }

    if (LType!=RType)
    {

      LogError(parser_struct.line, "Tried to attribute " + RType + " to " + LType);
      return false;
    }
  }
  return true;
}


Value *DataExprAST::codegen(Value *scope_struct) {
  if (not ShallCodegen)
    return ConstantFP::get(*TheContext, APFloat(0.0f));


  Function *TheFunction = Builder->GetInsertBlock()->getParent();

  // Register all variables and emit their initializer.
  for (unsigned i = 0, e = VarNames.size(); i != e; ++i) {
    const std::string &VarName = VarNames[i].first; 
    ExprAST *Init = VarNames[i].second.get();
    
    
    Value *init;

    bool is_self = GetSelf();
    bool is_attr = GetIsAttribute();

    Value *initial_value = Init->codegen(scope_struct);


    if(!Check_Is_Compatible_Type(Type, VarNames[i].second, parser_struct))
      return const_float(0);

    if((Type=="float"||Type=="int")&&!(is_self||is_attr))
    { 
      if (Type=="float"&&Init->GetType()=="int")
        initial_value = Builder->CreateUIToFP(initial_value, Type::getFloatTy(*TheContext), "floattmp");
      llvm::Type *alloca_type = get_type_from_str(Type);
      AllocaInst *alloca = CreateEntryBlockAlloca(TheFunction, Name, alloca_type);
      Builder->CreateStore(initial_value, alloca);
      function_allocas[parser_struct.function_name][VarName] = alloca;
      continue;
    }


    Value *var_name, *scopeless_name;

    // --- Name Solving --- //
    var_name = callret("CopyString", {global_str(VarName)});
    scopeless_name = callret("CopyString", {var_name});


    Value *notes_vector = callret("CreateNotesVector", {});


    // --- Notes --- //
    for (int j=0; j<Notes.size(); j++)
    {
      ExprAST *note = Notes[j].get();
      
      if (NumberExprAST* numExpr = dynamic_cast<NumberExprAST*>(note)) {
        
        notes_vector = callret("Add_To_NotesVector_float", {notes_vector, note->codegen(scope_struct)});
      }
      else if (IntExprAST* numExpr = dynamic_cast<IntExprAST*>(note)) {
        notes_vector = callret("Add_To_NotesVector_int", {notes_vector, note->codegen(scope_struct)});
      }
      else if (StringExprAST* expr = dynamic_cast<StringExprAST*>(note)) {
        Value *str_val = callret("CopyString", {note->codegen(scope_struct)});
        notes_vector = callret("Add_To_NotesVector_str", {notes_vector, str_val});
      }
      else if (VariableExprAST* expr = dynamic_cast<VariableExprAST*>(note)) {
        std::string type = expr->GetType();
        notes_vector = callret("Add_To_NotesVector_"+type, {notes_vector, note->codegen(scope_struct)});
      }
      else {
        std::cout << "Could not find the data type of a note in DataExpr of " << VarName << " \n";
      }
    }
    

 


    std::string create_fn = ((ends_with(Type,"_list")) ? "list" : Type);
    create_fn = ((ends_with(create_fn,"_dict")) ? "dict" : create_fn);
    create_fn = create_fn + "_Create";



    initial_value = callret(create_fn, {scope_struct, var_name, scopeless_name, initial_value, notes_vector});



    if(Type!="float"&&Type!="int")
    {
      call("MarkToSweep_Mark", {scope_struct, initial_value, global_str(Extract_List_Suffix(Type))});
      if(is_self||is_attr)
        call("MarkToSweep_Unmark_Scopeless", {scope_struct, initial_value});
      else
        call("MarkToSweep_Unmark_Scopeful", {scope_struct, initial_value});
    }


      

    if(is_self)
    {
      int object_ptr_offset = ClassVariables[parser_struct.class_name][VarName]; 
      // p2t(VarName+" offset is "+std::to_string(object_ptr_offset));
  
      Value *obj = callret("get_scope_object", {scope_struct});
      call("object_ptr_Attribute_object", {obj, const_int(object_ptr_offset), initial_value});

    } else if (is_attr) {
      LogError(parser_struct.line, "Creating attribute in a data expression is not supported.");
    }
    else {
      llvm::Type *alloca_type = get_type_from_str(Type);
      AllocaInst *alloca = CreateEntryBlockAlloca(TheFunction, Name, alloca_type);
      Builder->CreateStore(initial_value, alloca);

      function_allocas[parser_struct.function_name][VarName] = alloca;
    }
      

    call("Dispose_NotesVector", {notes_vector, scopeless_name});
    call("str_Delete", {var_name});
  }


  return ConstantFP::get(*TheContext, APFloat(0.0));
}




Value *LibImportExprAST::codegen(Value *scope_struct) {

  // Library import is made before codegen

  
  return const_float(0.0f);
}





Value *IfExprAST::codegen(Value *scope_struct) {
  if (not ShallCodegen)
    return ConstantFP::get(*TheContext, APFloat(0.0f));


  Value *CondV = Cond->codegen(scope_struct);
  if (!CondV)
    return nullptr;


  // Convert condition to a bool by comparing equal to 0.0.
  
  if(Cond->GetType()=="int")
    CondV = Builder->CreateICmpNE(
        CondV, const_int(0), "ifcond");
  if(Cond->GetType()=="float")
    CondV = Builder->CreateFCmpONE(
        CondV, const_float(0), "ifcond");

  Function *TheFunction = Builder->GetInsertBlock()->getParent();

  // Create blocks for the then and else cases.  Insert the 'then' block at the
  // end of the function.
  BasicBlock *ThenBB  = BasicBlock::Create(*TheContext, "then", TheFunction);
  BasicBlock *ElseBB  = BasicBlock::Create(*TheContext, "else");
  BasicBlock *MergeBB = BasicBlock::Create(*TheContext, "ifcont");

  Builder->CreateCondBr(CondV, ThenBB, ElseBB);

  // Emit then value.
  Builder->SetInsertPoint(ThenBB);

  
  
  Value *ThenV;
  for (auto &then_body : Then)
    ThenV = then_body->codegen(scope_struct);
  

  if (!ThenV)
    return nullptr;


  Builder->CreateBr(MergeBB);
  // Codegen of 'Then' can change the current block, update ThenBB for the PHI.
  ThenBB = Builder->GetInsertBlock();

  // Emit else block.
  TheFunction->insert(TheFunction->end(), ElseBB);
  Builder->SetInsertPoint(ElseBB);


  Value *ElseV;
  for (auto &else_body : Else)
    ElseV = else_body->codegen(scope_struct);

  if (!ElseV)
    return nullptr;

    

  Builder->CreateBr(MergeBB);
  // Codegen of 'Else' can change the current block, update ElseBB for the PHI.
  ElseBB = Builder->GetInsertBlock();

  // Emit merge block.
  TheFunction->insert(TheFunction->end(), MergeBB);
  Builder->SetInsertPoint(MergeBB);

  // if (ThenV->getType() != Type::getFloatTy(*TheContext))
    ThenV = Builder->CreateFPCast(ThenV, Type::getFloatTy(*TheContext));
  // if (ElseV->getType() != Type::getFloatTy(*TheContext))
    ElseV = Builder->CreateFPCast(ElseV, Type::getFloatTy(*TheContext));

  PHINode *PN = Builder->CreatePHI(Type::getFloatTy(*TheContext), 2, "iftmp");
  PN->addIncoming(ThenV, ThenBB);
  PN->addIncoming(ElseV, ElseBB);
  
  return PN;
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

Value *ForExprAST::codegen(Value *scope_struct) {
  if (not ShallCodegen)
    return const_float(0);
  Function *TheFunction = Builder->GetInsertBlock()->getParent();

  std::string cond_type = End->GetType();

  llvm::Type *llvm_type = get_type_from_str(cond_type);
  

  // Create an alloca for the variable in the entry block.
  AllocaInst *control_var_alloca = CreateEntryBlockAlloca(TheFunction, VarName, llvm_type);

  function_allocas[parser_struct.function_name][VarName] = control_var_alloca;

  // Emit the start code first, without 'variable' in scope.
  Value *StartVal = Start->codegen(scope_struct);
  if (!StartVal)
    return nullptr;

  Value *_zero;
  
  if (cond_type=="int")
    _zero = const_int(0);
  if (cond_type=="float")
    _zero = const_float(0.0f);

  if(Start->GetType()=="int"&&cond_type=="float")
    StartVal = Builder->CreateSIToFP(StartVal, Type::getFloatTy(*TheContext), "lfp");

  // std::cout << "CURRENT FUNCTION ON CODEGEN " << parser_struct.function_name << ".\n";




  Builder->CreateStore(StartVal, control_var_alloca);




  // Make the new basic block for the loop header, inserting after current
  // block.
  BasicBlock *CondBB = BasicBlock::Create(*TheContext, "for expr cond", TheFunction);
  BasicBlock *LoopBB  = BasicBlock::Create(*TheContext, "for expr loop");
  BasicBlock *AfterBB  = BasicBlock::Create(*TheContext, "for expr after");



  // Insert an explicit fall through from the current block to the LoopBB.
  Builder->CreateBr(CondBB);

  
  Builder->SetInsertPoint(CondBB);



  // Within the loop, the variable is defined equal to the PHI node.  If it
  // shadows an existing variable, we have to restore it outside this scope



  // Emit the body of the loop.  This, like any other expr, can change the
  // current BB.  Note that we ignore the value computed by the body, but don't
  // allow an error.
  
  

  // Emit the step value.
  Value *StepVal = nullptr;
  if (Step) {
    StepVal = Step->codegen(scope_struct);
    if (!StepVal)
      return nullptr;
  } 

  if(Step->GetType()=="int"&&cond_type=="float")
    StepVal = Builder->CreateSIToFP(StepVal, Type::getFloatTy(*TheContext), "lfp");


  // Compute the end condition.
  Value *EndCond = End->codegen(scope_struct);
  if (!EndCond)
    return nullptr;

  // Convert condition to a bool by comparing equal to 0.0.

  if(cond_type=="int")
    EndCond = Builder->CreateICmpNE(
        EndCond, _zero, "for expr loopcond"); 
  else if(cond_type=="float")
    EndCond = Builder->CreateFCmpONE(
        EndCond, _zero, "for expr loopcond");
  else
      return LogErrorV(parser_struct.line, "Unsupported type " + cond_type + " on the for expression");







  // conditional goto branch
  Builder->CreateCondBr(EndCond, LoopBB, AfterBB);




  // codegen body and increment
  TheFunction->insert(TheFunction->end(), LoopBB);
  Builder->SetInsertPoint(LoopBB);

  int j=0;
  for (auto &body : Body)
    body->codegen(scope_struct);

  call("scope_struct_Sweep", {scope_struct});


  // Reload, increment, and restore the alloca.  This handles the case where
  // the body of the loop mutates the variable.
  Value *CurVal = Builder->CreateLoad(llvm_type, control_var_alloca, VarName.c_str());
  Value *NextVal;
  if (cond_type=="int")
    NextVal = Builder->CreateAdd(CurVal, StepVal, "nextvar"); // Increment  
  if (cond_type=="float")
    NextVal = Builder->CreateFAdd(CurVal, StepVal, "nextvar"); // Increment 
  Builder->CreateStore(NextVal, control_var_alloca);

  
  
  Builder->CreateBr(CondBB);




  // when the loop body is done, return the insertion point to outside the for loop
  TheFunction->insert(TheFunction->end(), AfterBB);
  Builder->SetInsertPoint(AfterBB);


  // for expr always returns 0.0.
  return Constant::getNullValue(Type::getInt32Ty(*TheContext));
}







Value *ForEachExprAST::codegen(Value *scope_struct) {
  if (not ShallCodegen)
    return ConstantFP::get(*TheContext, APFloat(0.0f));
  Function *TheFunction = Builder->GetInsertBlock()->getParent();

  // // Create an alloca for the variable in the entry block.
  

  llvm::Type *alloca_type = get_type_from_str(Type);
  AllocaInst *control_var_alloca = CreateEntryBlockAlloca(TheFunction, VarName, alloca_type);
  AllocaInst *idx_alloca = CreateEntryBlockAlloca(TheFunction, VarName, Type::getInt32Ty(*TheContext));
  function_allocas[parser_struct.function_name][VarName] = control_var_alloca;


  Value *_zero = const_int(0);
  Value *CurIdx = const_int(0);

  

  Value *vec = Vec->codegen(scope_struct);

  Value *VecSize = callret("nsk_vec_size", {scope_struct, vec}); 
  Builder->CreateStore(CurIdx, idx_alloca);


  // VecSize = Builder->CreateFAdd(VecSize, const_float(1), "addtmp");  


  // Make the new basic block for the loop header, inserting after current
  // block.
  BasicBlock *CondBB = BasicBlock::Create(*TheContext, "cond", TheFunction);
  BasicBlock *LoopBB  = BasicBlock::Create(*TheContext, "loop");
  BasicBlock *AfterBB  = BasicBlock::Create(*TheContext, "after");



  // Insert an explicit fall through from the current block to the LoopBB.
  Builder->CreateBr(CondBB);

  
  Builder->SetInsertPoint(CondBB);


  // Emit the body of the loop.  This, like any other expr, can change the
  // current BB.  Note that we ignore the value computed by the body, but don't
  // allow an error.
 
  Value *StepVal = const_int(1);


  // Compute the end condition.
  Value *EndCond=Builder->CreateLoad(Type::getInt32Ty(*TheContext), idx_alloca, VarName.c_str());
  // Convert condition to a bool by comparing equal to 0.0.
  EndCond = Builder->CreateICmpNE(
      EndCond, VecSize, "loopcond");




  // conditional goto branch
  Builder->CreateCondBr(EndCond, LoopBB, AfterBB);




  // codegen body and increment
  TheFunction->insert(TheFunction->end(), LoopBB);
  Builder->SetInsertPoint(LoopBB);

  
  CurIdx = Builder->CreateLoad(Type::getInt32Ty(*TheContext), idx_alloca, VarName.c_str());


  std::string vec_type = Extract_List_Suffix(VecType);
  Value *vec_value = callret(vec_type+"_Idx", {scope_struct, vec, CurIdx});
  if(vec_type=="list"&&(Type=="float"||Type=="int"))
    vec_value = callret("to_"+Type, {scope_struct, vec_value});

  Builder->CreateStore(vec_value, control_var_alloca);

  Value *NextIdx = Builder->CreateAdd(CurIdx, StepVal, "nextvar"); // Increment  

  int j=0;
  for (auto &body : Body)
    body->codegen(scope_struct);
  call("scope_struct_Sweep", {scope_struct});


  // Reload, increment, and restore the alloca.  This handles the case where
  // the body of the loop mutates the variable.
  Builder->CreateStore(NextIdx, idx_alloca);

 
  
  Builder->CreateBr(CondBB);




  // when the loop body is done, return the insertion point to outside the for loop
  TheFunction->insert(TheFunction->end(), AfterBB);
  Builder->SetInsertPoint(AfterBB);


  // for expr always returns 0.0.
  // return Constant::getNullValue(Type::getFloatTy(*TheContext));
  return const_float(0);
}






Value *WhileExprAST::codegen(Value *scope_struct) {
  if (not ShallCodegen)
    return ConstantFP::get(*TheContext, APFloat(0.0f));
  
  Function* TheFunction = Builder->GetInsertBlock()->getParent();

  // Create blocks for loop condition, loop body, and after loop
  BasicBlock *CondBB = BasicBlock::Create(*TheContext, "cond_while", TheFunction);
  BasicBlock *LoopBB = BasicBlock::Create(*TheContext, "loop_while", TheFunction);
  BasicBlock *AfterBB = BasicBlock::Create(*TheContext, "end_while", TheFunction);

  // Jump to the condition block
  Builder->CreateBr(CondBB);

  // Insert the condition check block
  Builder->SetInsertPoint(CondBB);

  // Generate the condition code
  Value* condVal = Cond->codegen(scope_struct);
  if (!condVal)
    return nullptr;
  // Value *_zero = ConstantFP::get(*TheContext, APFloat(0.0));
  // condVal = Builder->CreateFCmpONE(condVal, _zero, "loopcond");
  Value *_zero = const_int(0);
  condVal = Builder->CreateICmpNE(condVal, _zero, "whilecond");


  // Create the conditional branch
  Builder->CreateCondBr(condVal, LoopBB, AfterBB);

  // Insert the loop body block
  Builder->SetInsertPoint(LoopBB);

  // Generate the loop body code
  for (auto &body : Body)
    body->codegen(scope_struct);
  call("scope_struct_Sweep", {scope_struct});

  // After the loop body, go back to the condition check
  Builder->CreateBr(CondBB);

  // Insert the after loop block
  Builder->SetInsertPoint(AfterBB);

  return Constant::getNullValue(Type::getFloatTy(*TheContext));
}







bool seen_var_attr = false;
Value *VariableExprAST::codegen(Value *scope_struct) {
  if (not ShallCodegen)
    return ConstantFP::get(*TheContext, APFloat(0.0f));
  // Look this variable up in the function.

  Function *TheFunction = Builder->GetInsertBlock()->getParent();
  //std::string functionName = TheFunction->getName().str();
  

  // p2t("Codegen for variableexprast");

  //std::cout << "Create value V" << "\n";
  Value *V;

  std::string type = GetType();
  std::string pre_dot = GetPreDot();
  bool is_self = GetSelf();
  bool is_attr = GetIsAttribute();
    

  // if (type=="str") // todo: NamedTensorsT break
  // {
  //   for (const auto &entry : NamedTensorsT)
  //   {
  //     std::cout << "Returning None because a tensor with name " << Name << " was found on strings map " << "\n";
  //     if (ends_with(entry.first, Name))
  //       return ConstantFP::get(*TheContext, APFloat(0.0f));
  //   } 
  // }





  if (!(is_self||is_attr))
    return load_alloca(Name, type, parser_struct.function_name); 

  

  if (is_self) {
    int object_ptr_offset = ClassVariables[parser_struct.class_name][Name];
    if (type=="float"||type=="int")
      return callret("object_Load_on_Offset_"+type, {scope_struct, const_int(object_ptr_offset)});
    else
    {
      // p2t("object_Load_on_Offset");
      return callret("object_Load_on_Offset", {scope_struct, const_int(object_ptr_offset)});
    }
  }

  Value *var_name;

  var_name = NameSolver->codegen(scope_struct);
  NameSolverAST *name_solver = static_cast<NameSolverAST *>(NameSolver.get());
  std::string Name = std::get<0>(name_solver->Names[name_solver->Names.size()-1]);
  


 
  std::string msg = "VariableExpr Variable " + Name + " load for type: " + type;
  p2t(msg);



  if (type=="object")
    return var_name;


  // if (type=="tensor" && !seen_var_attr)
  //   call("PrintTensor", {scope_struct, loaded_tensor});
  

  
  // p2t("Variable load");
  std::string load_fn = type + "_Load";
  // std::cout << "Load: " << load_fn << "-------------------------------------------.\n";
  V = callret(load_fn, {scope_struct, var_name});
  // call("str_Delete", {var_name});

  return V;
}









Value *VecIdxExprAST::codegen(Value *scope_struct) {
  if (not ShallCodegen)
    return ConstantFP::get(*TheContext, APFloat(0.0f));
  // Look this variable up in the function.

  std::string msg = "VecIdxExpr Now Loading Vec indexation for type: " + Type;
  p2t(msg);

  Function *TheFunction = Builder->GetInsertBlock()->getParent();
  std::string functionName = TheFunction->getName().str();


  Value *V, *var_name;


  Value *loaded_var = Loaded_Var->codegen(scope_struct);

  Value *idx = Idx_Calc_Codegen(Type, loaded_var, Idx, scope_struct);
  


  if (Type=="tensor")
  {
    std::cout << "vec idx of tensor, idx type: " << Idx->Idxs[0]->GetType() << "\n";

    if (Idx->Idxs[0]->GetType()!="tensor")
    {
      std::vector<Value *> idx_calc_args;
      idx_calc_args.push_back(var_name);
      idx_calc_args.push_back(Builder->CreateCall(TheModule->getFunction("get_scope_scope"), {scope_struct}));
      idx_calc_args.push_back(Builder->CreateCall(TheModule->getFunction("get_scope_thread_id"), {scope_struct}));
      for (int i=0; i<Idx->size(); i++)
        idx_calc_args.push_back(Idx->Idxs[i]->codegen(scope_struct));

      return Builder->CreateCall(TheModule->getFunction("IdxTensor"), idx_calc_args);
    } else {
      VariableExprAST *idx = static_cast<VariableExprAST *>(Idx->Idxs[0].get());
      Value *idx_tensor_name = idx->NameSolver->codegen(scope_struct);
      
      return Builder->CreateCall(TheModule->getFunction("IdxTensorWithTensor"), {var_name, idx_tensor_name, Builder->CreateCall(TheModule->getFunction("get_scope_thread_id"), {scope_struct})});      
    } 
  }


  std::string homogeneous_type = Extract_List_Suffix(Type);
  std::string type = Extract_List_Prefix(Type);


  if (homogeneous_type == "dict") {
    Value *ret_val = callret("dict_Query", {scope_struct, loaded_var, idx});
    if(type=="float"||type=="int")
      ret_val = callret("to_"+type, {scope_struct, ret_val});
    return ret_val;
  }
  
  if (!Idx->IsSlice) {
    std::string idx_fn = homogeneous_type + "_Idx";
    Value *ret_val = callret(idx_fn, {scope_struct, loaded_var, idx});
    if(type=="float"||type=="int")
      ret_val = callret("to_"+type, {scope_struct, ret_val});
    return ret_val;
  } else {
    std::string slice_fn = homogeneous_type + "_Slice";    
    Value *ret =  callret(slice_fn, {scope_struct, loaded_var, idx});
    call("Delete_Ptr", {idx});
    return ret;
  }
}





Value *ObjectVecIdxExprAST::codegen(Value *scope_struct) {
  return ConstantFP::get(*TheContext, APFloat(0.0f));
}
























Value *BinaryExprAST::codegen(Value *scope_struct) {
  if (not ShallCodegen)
    return ConstantFP::get(*TheContext, APFloat(0.0f));

  if (Op == '=') {
    seen_var_attr=true;
    // Assignment requires the LHS to be an identifier.
    // This assume we're building without RTTI because LLVM builds that way by
    // default.  If you build LLVM with RTTI this can be changed to a
    // dynamic_cast for automatic error checking.


    // Codegen the RHS.
    Value *Val = RHS->codegen(scope_struct);
    if (!Val)
    {
      seen_var_attr=false;
      return nullptr;
      
    }



    if (LHS->GetIsList())
    {
      // std::cout << "LIST ATTRIBUTION" << ".\n";
      
      VariableListExprAST *VarList = static_cast<VariableListExprAST *>(LHS.get());

      for (int i=0; i<VarList->ExprList.size(); ++i)
      {
        // std::cout << "Attributing: " << i << ".\n";
        VariableExprAST *LHSE = static_cast<VariableExprAST *>(VarList->ExprList[i].get());
        Value *Lvar_name = LHSE->NameSolver->codegen(scope_struct);

        NameSolverAST *name_solver = static_cast<NameSolverAST *>(LHSE->NameSolver.get());
        std::string Lname = std::get<0>(name_solver->Names[0]);
        std::string LType = LHSE->GetType();

        // std::cout << "ATTRIBUTION: " << LType << " for " << i << ".\n";
        
        std::string store_trigger = LType + "_StoreTrigger";
        Value *Val_indexed = callret("assign_wise_list_Idx", {Val, const_int(i)});

        AllocaInst *alloca = function_allocas[parser_struct.function_name][Lname];
        



        Function *F = TheModule->getFunction(store_trigger);
        if (F)
        {
          Value *old_val = Builder->CreateLoad(int8PtrTy, alloca);
          Lvar_name = LHSE->NameSolver->codegen(scope_struct);
          call(store_trigger, {Lvar_name, old_val, Val_indexed, scope_struct});
        }

        if (LType=="int")
          Val_indexed = callret("to_int", {scope_struct, Val_indexed});
        if (LType=="float")
          Val_indexed = callret("to_float", {scope_struct, Val_indexed});
        //   Val_indexed = Builder->CreatePtrToInt(Val_indexed, Type::getInt32Ty(*TheContext));

        Builder->CreateStore(Val_indexed, alloca);

        if (LType!="float"&&LType!="int")
        {
          call("MarkToSweep_Mark", {scope_struct, Val_indexed, global_str(Extract_List_Suffix(LType))});
        }
        
      }
      return ConstantFP::get(*TheContext, APFloat(0.0f));
    }




    std::string LType = LHS->GetType();
    std::string store_op = LType + "_Store";



    if(LHS->GetIsVec())
    {
      VecIdxExprAST *LHSV = static_cast<VecIdxExprAST *>(LHS.get());
      Value *vec = LHSV->Loaded_Var->codegen(scope_struct);

      Value *idx = Idx_Calc_Codegen(LHS->GetType(), vec, LHSV->Idx, scope_struct);

      store_op = Extract_List_Suffix(store_op);

      if(ends_with(LHSV->GetType(), "dict"))
      {
        
        store_op = store_op + "_Key";
        

        std::string RType = RHS->GetType();
        if (RType=="int" || RType=="float")
        {
          store_op = store_op + "_" + RType;
          call(store_op, {scope_struct, vec, idx, Val}); 

          return const_float(0);
        }
        
        call(store_op, {scope_struct, vec, idx, Val, global_str(RType)});

        return const_float(0);
      }

      store_op = store_op + "_Idx";
      
      call(store_op, {vec, idx, Val, scope_struct});

      return const_float(0);
    }







    // bool is_alloca = ((LType=="float"||LType=="str"||LType=="int"||LType=="tensor")&&!LHS->GetSelf()&&!LHS->GetIsAttribute());
    bool is_alloca = (!LHS->GetSelf()&&!LHS->GetIsAttribute());
    

    Value *Lvar_name;
      
    
    
    

    

    

    std::string Lname = LHS->GetName();

    if (is_alloca)
    {
      Check_Is_Compatible_Type(LType, RHS, parser_struct);

      std::string store_trigger = LType + "_StoreTrigger";
      std::string copy_fn = LType + "_Copy";


      AllocaInst *alloca = function_allocas[parser_struct.function_name][Lname];

      if(auto Rvar = dynamic_cast<VariableExprAST *>(RHS.get())) // if it is leaf
      {
        Function *F = TheModule->getFunction(copy_fn);
        if (F)
        {
          Val = callret(copy_fn, {scope_struct, Val});
          call("MarkToSweep_Mark", {scope_struct, Val, global_str(Extract_List_Suffix(LType))});
        }
      }



      if (LType!="float"&&LType!="int")
      {
        Value *old_val = Builder->CreateLoad(int8PtrTy, alloca);
        call("MarkToSweep_Mark_Scopeful", {scope_struct, old_val, global_str(Extract_List_Suffix(LType))});
      }



      Function *F = TheModule->getFunction(store_trigger);
      if (F)
      {
        Value *old_val = Builder->CreateLoad(int8PtrTy, alloca);
        Value *Lvar_name = callret("CopyString", {global_str(Lname)});
        Val = callret(store_trigger, {Lvar_name, old_val, Val, scope_struct});
      }

      if (LType!="float"&&LType!="int")
        call("MarkToSweep_Unmark_Scopeful", {scope_struct, Val, global_str(LType)});
      if (LType=="float"&&RHS->GetType()=="int")
        Val = Builder->CreateUIToFP(Val, Type::getFloatTy(*TheContext), "floattmp");

      // p2t("Store " + LType + " at " + parser_struct.function_name + "/"+ Lname);



      Builder->CreateStore(Val, alloca);
    } else
    {


      if(auto *LHSV = dynamic_cast<NestedVectorIdxExprAST *>(LHS.get())) {

        LHSV->skip=true;
        Value *obj_ptr = LHSV->codegen(scope_struct);
        std::string type = LHSV->GetType();

        Value *idx = Idx_Calc_Codegen(type, obj_ptr, LHSV->Idx, scope_struct);

        call(type+"_Store_Idx", {obj_ptr, idx, Val, scope_struct});
      }

      if(auto *LHSV = dynamic_cast<NestedVariableExprAST *>(LHS.get())) {
        Check_Is_Compatible_Type(LType, RHS, parser_struct);

        LHSV->Load_Val = false;
        LHSV->Inner_Expr->Load_Last=false;


        Value *obj_ptr = LHSV->codegen(scope_struct);
        
        if(LType=="float"||LType=="int")
          call("object_Attr_"+LType, {obj_ptr, Val});
        else
          call("tie_object_to_object", {obj_ptr, Val});
      }
      

      if (LType!="float"&&LType!="int")
        call("MarkToSweep_Unmark_Scopeless", {scope_struct, Val, global_str(LType)});
      
    }
    



    

    seen_var_attr=false;
    return ConstantFP::get(*TheContext, APFloat(0.0f));
  }


  
  
  Value *L = LHS->codegen(scope_struct);
  Value *R = RHS->codegen(scope_struct);
  
  if (!L || !R)
  return nullptr;

  if (Elements=="int_float") {
    Elements = "float_float"; 
    L = Builder->CreateSIToFP(L, Type::getFloatTy(*TheContext), "lfp");
  }
  if (Elements=="float_int") {
    Elements = "float_float"; 
    R = Builder->CreateSIToFP(R, Type::getFloatTy(*TheContext), "lfp");
  }
  if (Operation=="tensor_int_div")
  {
    Operation = "tensor_float_div";
    R = Builder->CreateSIToFP(R, Type::getFloatTy(*TheContext), "lfp");
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
        L = Builder->CreateFCmpULT(L, R, "cmptmp");
        // Convert bool 0/1 to float 0.0 or 1.0
        return Builder->CreateUIToFP(L, Type::getFloatTy(*TheContext), "booltmp");
      case '>':
        L = Builder->CreateFCmpULT(R, L, "cmptmp");
        // Convert bool 0/1 to float 0.0 or 1.0
        return Builder->CreateUIToFP(L, Type::getFloatTy(*TheContext), "booltmp");
      case tok_equal:
        L = Builder->CreateFCmpUEQ(L, R, "cmptmp");
        // Convert bool 0/1 to float 0.0 or 1.0
        return Builder->CreateUIToFP(L, Type::getFloatTy(*TheContext), "booltmp");
      case tok_diff:
        L = Builder->CreateFCmpUNE(L, R, "cmptmp");
        // Convert bool 0/1 to float 0.0 or 1.0
        return Builder->CreateUIToFP(L, Type::getFloatTy(*TheContext), "booltmp");
      case tok_minor_eq:
          L = Builder->CreateFCmpULE(L, R, "cmptmp");  // less or equal
          return Builder->CreateUIToFP(L, Type::getFloatTy(*TheContext), "booltmp");
      case tok_higher_eq:
          L = Builder->CreateFCmpUGE(L, R, "cmptmp");  // greater or equal
          return Builder->CreateUIToFP(L, Type::getFloatTy(*TheContext), "booltmp");
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
        L = Builder->CreateICmpSLT(L, R, "cmptmp");
        return Builder->CreateZExt(L, Type::getInt32Ty(*TheContext), "booltmp");
      case '>':
        L = Builder->CreateICmpSGT(L, R, "cmptmp");
        return Builder->CreateZExt(L, Type::getInt32Ty(*TheContext), "booltmp");
      case tok_equal:
        L = Builder->CreateICmpEQ(L, R, "cmptmp");
        return Builder->CreateZExt(L, Type::getInt32Ty(*TheContext), "booltmp");
      case tok_diff:
        L = Builder->CreateICmpNE(L, R, "cmptmp");
        return Builder->CreateZExt(L, Type::getInt32Ty(*TheContext), "booltmp");
      case tok_minor_eq:
        L = Builder->CreateICmpSLE(L, R, "cmptmp");
        return Builder->CreateZExt(L, Type::getInt32Ty(*TheContext), "booltmp");
      case tok_higher_eq:
        L = Builder->CreateICmpSGE(L, R, "cmptmp");
        return Builder->CreateZExt(L, Type::getInt32Ty(*TheContext), "booltmp");
      default:
        break;
    }
  } else {
    // std::string msg = "Codegen for operation: " + Operation;
    // p2t(msg);


    Value *ret = callret(Operation, {scope_struct, L, R});

    if(ops_type_return.count(Elements)>0)
    {
      std::string return_type = ops_type_return[Elements];
      // std::cout << "Operation of " << Elements << " has a return of " << return_type << ".\n";

      if (return_type!="float"&&return_type!="int")
        call("MarkToSweep_Mark", {scope_struct, ret, global_str(Extract_List_Suffix(return_type))});
    }


    return ret; 
  }



  

  // If it wasn't a builtin binary operator, it must be a user defined one. Emit
  // a call to it.
  Function *F = getFunction(std::string("binary") + Op);
  assert(F && "Operator not found.");

  Value *Ops[] = {L, R};
  return Builder->CreateCall(F, Ops, "binop");
}















Value *BinaryObjExprAST::codegen(Value *scope_struct) {
  if (not ShallCodegen)
    return ConstantFP::get(*TheContext, APFloat(0.0f));


  Value *LName = Builder->CreateGlobalString(LHS->GetName());
  Value *RName = Builder->CreateGlobalString(RHS->GetName());
  Value *object_name;

  


  // if is attribution
  if (Op == '=') {
  
    seen_var_attr=true;


    // if (!LHS->GetIsVec())
    // {
    //   std::cout << "\n\n3 3 attr\n";
    //   VariableExprAST *LHSE = static_cast<VariableExprAST *>(LHS.get());
    //   if (!LHSE)
    //     return LogErrorV("'=' object attribution destiny must be an object variable.");
    //   LName = LHSE->NameSolver->codegen(scope_struct);
      
    //   if (RHS->GetIsVec())
    //   {
    //     std::cout << "3 3 other INDEXED of RHS->GetIsVec() && RHS->GetType()==object" << "\n";
    //     VecIdxExprAST *RHSE = static_cast<VecIdxExprAST *>(RHS.get());
    //     RName = RHSE->NameSolver->codegen(scope_struct);
        
    //     Builder->CreateCall(TheModule->getFunction("objAttr_var_from_vec"),
    //                                                     {LName, RName});
    //   } else {
    //     VariableExprAST *RHSE = static_cast<VariableExprAST *>(RHS.get());
    //     RName = RHSE->NameSolver->codegen(scope_struct);
        
    //     Builder->CreateCall(TheModule->getFunction("objAttr_var_from_var"),
    //                                                     {LName, RName});

    //   }
    
    // } else {
    //   std::cout << "\n\n3 3 other INDEXED attr\n";
    //   VecIdxExprAST *LHSE = static_cast<VecIdxExprAST *>(LHS.get());
    //   if (!LHSE)
    //     return LogErrorV("'=' object attribution destiny must be an object variable.");
    //   LName = LHSE->NameSolver->codegen(scope_struct);


    //   std::cout << "ok" << "\n";
      
    //   if (RHS->GetIsVec())
    //   {
    //     std::cout << "3 3 other INDEXED of RHS->GetIsVec() && RHS->GetType()==object" << "\n";
    //     VecIdxExprAST *RHSE = static_cast<VecIdxExprAST *>(RHS.get());
    //     RName = RHSE->NameSolver->codegen(scope_struct);
        
    //     Builder->CreateCall(TheModule->getFunction("objAttr_vec_from_vec"),
    //                                                     {LName, RName});
    //   } else {
    //     std::cout << "3 3 VEC FROM VAR" << "\n";
    //     VariableExprAST *RHSE = static_cast<VariableExprAST *>(RHS.get());
    //     RName = RHSE->NameSolver->codegen(scope_struct);
        
    //     Builder->CreateCall(TheModule->getFunction("objAttr_vec_from_var"),
    //                                                     {LName, RName});

    //   }


    // }
    seen_var_attr=false;
    return ConstantFP::get(*TheContext, APFloat(0.0f));
  }
  
}


















Value *UnaryExprAST::codegen(Value *scope_struct) {
  if (not ShallCodegen)
    return ConstantFP::get(*TheContext, APFloat(0.0f));
  Value *OperandV = Operand->codegen(scope_struct);
  if (!OperandV)
    return nullptr;
  
  
  
  //std::cout << "Operand type: " << Operand->GetType();
  if (Opcode=='-')
  {
    //std::cout << "\n\n\n\n\n\nIT'S A MINUS " << Operand->GetType() << "\n\n\n\n\n\n\n";
    if (Operand->GetType()=="tensor")
    {
      Value *tensor_name = global_str(Operand->GetName());

      std::string pre_dot = Operand->GetPreDot();
      bool is_self = Operand->GetSelf();
      bool is_attr = Operand->GetIsAttribute();

      if (is_attr) { // Gets from pre_dot if it is a class attribute
        Value * object_name = global_str(pre_dot);

        tensor_name = callret("ConcatStr", {object_name, tensor_name});
      }
      if (is_self)
        tensor_name = callret("ConcatStr", {callret("get_scope_first_arg", {scope_struct}), tensor_name});
      if (!(is_self||is_attr))
        tensor_name = callret("ConcatStr", {callret("get_scope_scope", {scope_struct}), tensor_name});
        

      Value *tensorPtr = callret("tensor_Load", {scope_struct, tensor_name});
      Value *R = ConstantFP::get(Type::getFloatTy(*TheContext), -1);

      return callret("CudaScalarMult", {tensorPtr, R, callret("get_scope_thread_id", {scope_struct})});
    }
    
    if (Operand->GetType()=="int")
      return Builder->CreateMul(ConstantInt::get(Type::getInt32Ty(*TheContext), -1), OperandV, "multmp");
    
    if (Operand->GetType()=="tensor")
      return Builder->CreateFMul(ConstantFP::get(Type::getFloatTy(*TheContext), -1),
                              OperandV, "multmp");
  }

  //std::cout << "Opcode: " << Opcode << "\n";


  if (Opcode='!')
  {
    return Builder->CreateCall(TheModule->getFunction("logical_not"), {OperandV});
  }
  if (Opcode=';')
    return OperandV;
    // return ConstantFP::get(Type::getFloatTy(*TheContext), 0);
  

  Function *F = getFunction(std::string("unary") + Opcode);
  if (!F)
    return LogErrorV(parser_struct.line,"Unknown unary operator.");

  return Builder->CreateCall(F, OperandV, "unop");
}






























Function *codegenAsyncFunction(std::vector<std::unique_ptr<ExprAST>> &asyncBody, Value *scope_struct, Parser_Struct parser_struct, Value *barrier) {
  


  // find existing unique function name (_async_1, _async_2, _async_3 etc)
  int fnIndex = 1;
  while (TheModule->getFunction("__async_" + std::to_string(fnIndex)))
    fnIndex++;
  
  // todo: solve thread stream
  // cudaStream_t thread_stream = createCudaStream();
  // ThreadsStream[fnIndex] = thread_stream;

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

  // function_allocas[parser_struct.function_name][VarName] = alloca;

  // std::vector<Value *> previous_scope_values;
  std::vector<std::string> previous_scope_value_types;
  std::vector<std::string> previous_scope_value_names;


  for (auto &pair : function_allocas["__anon_expr"]) {

    std::string type;
    if (Object_toClass[parser_struct.function_name].count(pair.first)>0)
      type = "void";
    else
    {   
      type = typeVars[parser_struct.function_name][pair.first];
      if(type!="float"&&type!="int")
        type="void";
      // Value *v = load_alloca(pair.first, type, "__anon_expr");
      // std::cout << "Got type " << type << ".\n";
      // call("earth_cable", {v});
      // continue;
    }

    Value *local = load_alloca(pair.first, type, "__anon_expr");
    call("dive_"+type, {global_str(functionName), local, global_str(pair.first)});


    previous_scope_value_types.push_back(type);
    previous_scope_value_names.push_back(pair.first);

  }
  
  //Dive scope_struct
  Builder->CreateCall(TheModule->getFunction("scope_struct_Save_for_Async"), {scope_struct, Builder->CreateGlobalString(functionName), barrier}); 



  // emit EntryBB value
  BasicBlock *BB = BasicBlock::Create(*TheContext, "async_bb", asyncFun);
  Builder->SetInsertPoint(BB);
  



  

  // Recover scope_struct Value * on the new function
  Value *scope_struct_copy = Builder->CreateCall(TheModule->getFunction("scope_struct_Load_for_Async"), {Builder->CreateGlobalString(functionName)}); 

  // define body of function
  Value *V;

  for(int i=0; i<previous_scope_value_names.size(); ++i) {
    std::string type = previous_scope_value_types[i];
    std::string var_name = previous_scope_value_names[i];

    Value *v = callret("emerge_"+type, {global_str(functionName), global_str(var_name)});

    llvm::Type *llvm_type = get_type_from_str(type);
    AllocaInst * alloca = CreateEntryBlockAlloca(asyncFun, var_name, llvm_type);
    Builder->CreateStore(v, alloca);
    function_allocas["asyncs"][var_name] = alloca;
  }

  
  for (auto &body : asyncBody)
    V = body->codegen(scope_struct_copy);



  if (V)
  { 
    p2t("codegenAsyncFunction create return");
    Builder->CreateRet(Constant::getNullValue(int8PtrTy));
    
     
    std::string functionError;
    llvm::raw_string_ostream functionErrorStream(functionError);

    if (verifyFunction(*asyncFun, &functionErrorStream)) {
      functionErrorStream.flush();
      llvm::errs() << "codegenAsyncFunction: Function verification failed:\n" << functionError << "\n";
    } 

    verifyModule(*TheModule);
    return asyncFun;
  }
  
  std::cout << "ERASING ASYNC FROM PARENT" << "\n";
  asyncFun->eraseFromParent();

  return nullptr;
}




Value *AsyncExprAST::codegen(Value *scope_struct) {
  if (not ShallCodegen)
    return ConstantFP::get(*TheContext, APFloat(0.0f));

  
  // Create/Spawn Threads

  

  Value *barrier = callret("get_barrier", {const_int(1)});

  BasicBlock *CurrentBB = Builder->GetInsertBlock();


  Function *asyncFun = codegenAsyncFunction(Body, scope_struct, parser_struct, barrier);


  Builder->SetInsertPoint(CurrentBB);

  
  Function *pthread_create = TheModule->getFunction("pthread_create_aux");


  PointerType *pthreadTy = Type::getInt8Ty(*GlobalContext)->getPointerTo();
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
  Value *barrier = callret("get_barrier", {const_int(AsyncsCount)});

  BasicBlock *CurrentBB = Builder->GetInsertBlock();

  Function *asyncFun = codegenAsyncFunction(Body, scope_struct, parser_struct, barrier);


  Builder->SetInsertPoint(CurrentBB);

  

  for(int i=0; i<AsyncsCount; i++) 
  {
    PointerType *pthreadTy = Type::getInt8Ty(*GlobalContext)->getPointerTo();
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
  // call("scope_struct_Increment_Thread", {scope_struct});
  
  Value *inner_vec = Inner_Vec->codegen(scope_struct);
  SetType(Inner_Vec->GetType());


  std::string split_fn = Inner_Vec->GetType() + "_Split_Parallel";

  return callret(split_fn, {scope_struct, inner_vec});
}

Value *SplitStridedParallelExprAST::codegen(Value *scope_struct) {
  
  Value *inner_vec = Inner_Vec->codegen(scope_struct);
  SetType(Inner_Vec->GetType());

  std::string split_fn = Inner_Vec->GetType() + "_Split_Strided_Parallel";

  return callret(split_fn, {scope_struct, inner_vec});
}


Value *FinishExprAST::codegen(Value *scope_struct) {
  if (not ShallCodegen)
    return ConstantFP::get(*TheContext, APFloat(0.0f));
  
  Function *TheFunction = Builder->GetInsertBlock()->getParent();
  std::string functionName = TheFunction->getName().str();


  for (int i=0; i < Bodies.size(); i++)
    Bodies[i]->codegen(scope_struct);
  

  PointerType *pthreadTy = Type::getInt8Ty(*GlobalContext)->getPointerTo();

  Function *pthread_join = TheModule->getFunction("pthread_join_aux");


  //std::cout << "\n\n\n\nFINISH HAS " << thread_pointers.size() << " ASYNC EXPRESSIONS "  << "\n\n\n\n\n";


  for (Value *pthreadPtr : thread_pointers)
  {
    Value *pthread = Builder->CreateLoad(pthreadTy, pthreadPtr);

    Builder->CreateCall(pthread_join,
                        {pthread});
    
  }
  thread_pointers.clear();

  call("scope_struct_Reset_Threads", {scope_struct});
  
  return ConstantFP::get(*TheContext, APFloat(0.0f));
}


Value *LockExprAST::codegen(Value *scope_struct){
  
  Builder->CreateCall(TheModule->getFunction("LockMutex"), {Builder->CreateGlobalString(Name)});

  for (auto &body : Bodies)
    body->codegen(scope_struct);

  Builder->CreateCall(TheModule->getFunction("UnlockMutex"), {Builder->CreateGlobalString(Name)});

  return ConstantFP::get(*TheContext, APFloat(0.0f));
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
    std::string type = Vars[0]->GetType();
    if(type!="float"&&type!="int")
      call("MarkToSweep_Unmark_Scopeless", {scope_struct, ret});

    seen_var_attr=false;
    call("scope_struct_Clean_Scope", {scope_struct}); 
    Builder->CreateRet(ret);
    return const_float(0);
  }
  
  std::vector<Value *> values = {scope_struct};
  for (int i=0; i<Vars.size(); i++)
  {
    Value *value = Vars[i]->codegen(scope_struct);
    if(Vars[i]->GetType()!="float"&&Vars[i]->GetType()!="int")
      call("MarkToSweep_Unmark_Scopeless", {scope_struct, value});
    std::string type = Vars[i]->GetType();
    values.push_back(global_str(type));
    values.push_back(value);
  }
  values.push_back(global_str("TERMINATE_VARARG"));

  seen_var_attr=false;
  Value *ret = callret("list_New", values);
  call("scope_struct_Clean_Scope", {scope_struct}); 
  Builder->CreateRet(ret);
  return ret;

  // for (int i=0; i<Values.size(); i++)
  // {
  //   std::string type = Values[i]->GetType();
  //   Value *value = Values[i]->codegen(scope_struct);
  //   if (!is_type)
  //   {
  //     if (type!="float")
  //     {
  //       std::string copy_fn = type + "_" + "Copy";
  //       value = callret(copy_fn, {scope_struct, value});
  //     }
  //     is_type=true;
  //   } else
  //     is_type=false;
  //   values.push_back(value);
 
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
      if (type!="float"&&type!="int")
      {

        std::string copy_fn = type + "_Copy";
        
        Function *F = TheModule->getFunction(copy_fn);
        if (F)
        {
          value = callret(copy_fn, {scope_struct, value});
          call("MarkToSweep_Mark", {scope_struct, value, global_str(Extract_List_Suffix(type))});
        }
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


    if (type!="float"&&type!="int")
    {

      std::string copy_fn = type + "_Copy";
      
      Function *F = TheModule->getFunction(copy_fn);
      if (F)
      {
        value = callret(copy_fn, {scope_struct, value});
        call("MarkToSweep_Mark", {scope_struct, value, global_str(Extract_List_Suffix(type))});
      }
    }

    values.push_back(value);
  }


  Value *value = Values[Values.size()-1]->codegen(scope_struct);
  values.push_back(value);

  seen_var_attr = false;

  return callret("dict_New", values);
}




Value* callMalloc(Value *size) {
    // Get or declare malloc: `i8* @malloc(i64)`
    // Call malloc
    return callret("malloc", {size});
}

Value *ObjectExprAST::codegen(Value *scope_struct) {
  if (not ShallCodegen)
    return ConstantFP::get(*TheContext, APFloat(0.0f));


  Function *TheFunction = Builder->GetInsertBlock()->getParent();

  Value *init;
  if (Init)
    init = Init->codegen(scope_struct);

  // Register all variables and emit their initializer.

  for (unsigned i = 0, e = VarNames.size(); i != e; ++i)
  {
    const std::string &VarName = VarNames[i].first;

    Value *var_name, *obj_name;// = Builder->CreateCall(TheModule->getFunction("GetEmptyChar"), {});
    
    std::string pre_dot = GetPreDot();
    bool is_self = GetSelf();
    bool is_attr = GetIsAttribute();
    
    if (!GetIsVec())
    {
      if(!is_self&&!is_attr)
      {   
        AllocaInst *alloca = CreateEntryBlockAlloca(TheFunction, VarName, int8PtrTy);
        Value *ptr = callret("malloc", {const_int64(Size)});
        Builder->CreateStore(ptr, alloca);
        // Value *ptr = callret("posix_memalign", {alloca, const_int64(8), const_int64(Size)});
        // std::cout << "ADDING OBJECT " << VarName << " TO FUNCTION " << parser_struct.function_name << ".\n";
        function_allocas[parser_struct.function_name][VarName] = alloca;


        continue;
      }
    }
    else if (Init) // init of vec[size]
    {
      var_name = global_str(VarName);

      if (is_self||is_attr) 
        var_name = callret("ConcatStr", {callret("get_scope_first_arg", {scope_struct}), var_name});
      if (!(is_self||is_attr))
        var_name = callret("ConcatStr", {callret("get_scope_scope", {scope_struct}), var_name});


      call("InitObjectVecWithNull", {var_name, init});
    } else
    {}
  }


  return ConstantFP::get(*TheContext, APFloat(0.0));
}









Function *PrototypeAST::codegen() {
  if (not ShallCodegen)
    return nullptr;
  // Make the function type:  float(float,float) etc.


  std::vector<Type *> types;


  for (auto &type : Types)
  {
    if (type=="f"||type=="float")
      types.push_back(Type::getFloatTy(*TheContext));
    else if(type=="i"||type=="int")
      types.push_back(Type::getInt32Ty(*TheContext));
    else
      types.push_back(int8PtrTy);
  }
  
  FunctionType *FT;
  if (Return_Type=="float")
    FT = FunctionType::get(Type::getFloatTy(*TheContext), types, false);
  else if (Return_Type=="int")
    FT = FunctionType::get(Type::getInt32Ty(*TheContext), types, false);
  else
    FT = FunctionType::get(int8PtrTy, types, false); 
  

  Function *F =
      Function::Create(FT, Function::ExternalLinkage, Name, TheModule.get());

  // Set names for all arguments.
  unsigned Idx = 0;
  for (auto &Arg : F->args())
    Arg.setName(Args[Idx++]);




  return F;
}




inline std::vector<Value *> codegen_Argument_List(Parser_Struct parser_struct, std::vector<Value *> ArgsV, std::vector<std::unique_ptr<ExprAST>> Args, Value *scope_struct, std::string fn_name, bool is_nsk_fn, int arg_offset=1)
{

  // Get Arguments
  for (unsigned i = 0, e = Args.size(); i != e; ++i) {
    Value *arg; 

    arg = Args[i]->codegen(scope_struct);
    std::string type = Args[i]->GetType();
      

    if (!in_str(fn_name, {"to_int", "to_float"}))
    { 
      if (Function_Arg_Types.count(fn_name)>0)
      {
        int tgt_arg = i + arg_offset;

        std::string expected_type = Function_Arg_Types[fn_name][Function_Arg_Names[fn_name][tgt_arg]];
        if (type!=expected_type)
        {
          bool is_equivalent = false;
          if (Equivalent_Types.count(type)>0)
            for(std::string equivalent : Equivalent_Types[type])
              if (equivalent==expected_type)
                is_equivalent=true;

          if(!is_equivalent)
            LogError(parser_struct.line, "Passed type " + type + " for the argument " + Function_Arg_Names[fn_name][tgt_arg] + " of function " + fn_name + ", but expected " + expected_type + ".");
        }
      }  
    }


    std::string copy_fn = type+"_CopyArg";
    Function *F = TheModule->getFunction(copy_fn);
    if (F&&!is_nsk_fn)
    {
        Value *copied_value = callret(copy_fn,
                        {scope_struct,
                        arg,
                        global_str("-")});
                        
        
        // call("MarkToSweep_Mark", {scope_struct, copied_value, global_str(type)}); // Mark at FunctionAST
        // call("MarkToSweep_Unmark_Scopeful", {scope_struct, copied_value});
      ArgsV.push_back(copied_value);
    } else
      ArgsV.push_back(arg);



    if (!ArgsV.back())
    {
      LogError(parser_struct.line, "Failed to codegen argument of function " + fn_name);
      return {};
    }
  }

  return std::move(ArgsV);
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
  }

  for(; i<last; ++i)
  {
    std::string next_class = Object_toClass[_class][expressions_string_vec[i]];
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
    _class = Object_toClass[_class][expressions_string_vec[i]];
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
    _class = Object_toClass[_class][expressions_string_vec[i]];
  }
  // p2t("Offset of " + expressions_string_vec[last] + " is " + std::to_string(ClassVariables[_class][expressions_string_vec[last]]));

  return ClassVariables[_class][expressions_string_vec[last]];
}

Value *SelfExprAST::codegen(Value *scope_struct) {
  return callret("get_scope_object", {scope_struct});
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







Value *NestedVariableExprAST::codegen(Value *scope_struct) {
  // std::cout << "Nested Variable Expr" << ".\n";
  // Print_Names_Str(Inner_Expr->Expr_String);

  Value *ptr = Inner_Expr->codegen(scope_struct);

  if (Load_Val&&(Type=="float"||Type=="int"))
    return callret("object_Load_"+Type, {ptr});
  
  return ptr;
}



Value *NestedCallExprAST::codegen(Value *scope_struct) {
  // std::cout << "--NestedCall Calling " << Callee << ".\n";
  // p2t("Calling: " + Callee);
  
  bool is_nsk_fn = in_str(Callee, native_fn);

  
  Value *obj_ptr;
  Value *scope_struct_copy = callret("scope_struct_Copy", {scope_struct});

  call("set_scope_function_name", {scope_struct_copy, global_str(Callee)});


  // p2t("Calling: " + Callee);


  if(ends_with(Callee, "__init__")&&Inner_Expr->From_Self) // mallocs an object inside another
  {
    // p2t("RUN INIT FOR CALLEE "+ Callee);

    Inner_Expr->Load_Last=false; // inhibits Load_slot
    obj_ptr = Inner_Expr->codegen(scope_struct);


    int size = Get_Nested_Class_Size(Inner_Expr->Expr_String, parser_struct);
    Value *new_ptr = callret("malloc", {const_int64(size)});

    // p2t("Malloc size of " + Inner_Expr->Name + " is " + std::to_string(size));

    call("tie_object_to_object", {obj_ptr, new_ptr});
    
    obj_ptr = new_ptr;  
  } else
    obj_ptr = Inner_Expr->codegen(scope_struct);
  
  
  if(!is_nsk_fn)
    call("set_scope_object", {scope_struct_copy, obj_ptr});


  



  int target_args_size = Args.size()+1; // +1 for scope_struct



  std::vector<Value *> ArgsV = {scope_struct_copy};
  int arg_type_check_offset = 1; // 1 for scope_struct
  if (is_nsk_fn) // load of x at x.shape()
  {
    target_args_size++;
    ArgsV.push_back(obj_ptr);
    arg_type_check_offset++;
  }
  ArgsV = codegen_Argument_List(parser_struct, std::move(ArgsV), std::move(Args), scope_struct, Callee, is_nsk_fn, arg_type_check_offset);



  Function *CalleeF;
  CalleeF = getFunction(Callee);
  if (!CalleeF)
  {
    std::string _error = "The referenced function "+ Callee +" was not yet declared.";
    return LogErrorV(parser_struct.line, _error);
  }
  std::string tgt_function_name = CalleeF->getName().str();
  // If argument mismatch error.
  if ((CalleeF->arg_size()) != target_args_size && !in_str(tgt_function_name, vararg_methods))
  {
    // std::cout << "CalleeF->arg_size() " << std::to_string(CalleeF->arg_size()) << " target_args_size " << std::to_string(target_args_size) << "\n";
    std::string _error = "Incorrect parameters used on function " + Callee + " call.\n\t    Expected " +  std::to_string(CalleeF->arg_size()-1) + " arguments, got " + std::to_string(target_args_size-1);
    return LogErrorV(parser_struct.line, _error);
  }


  // Call function
  Value *ret = callret(Callee, ArgsV);

  // Clean-up
  call("scope_struct_Delete", {scope_struct_copy});
  
  if(!in_str(Type, {"float", "int", "", "None"}))
    call("MarkToSweep_Mark", {scope_struct, ret, global_str(Extract_List_Suffix(Type))});
  
  return ret;
}







Value *CallExprAST::codegen(Value *scope_struct) {
  if (not ShallCodegen)
    return const_float(0);
  // Look up the name in the global module table.

  

  std::string tgt_function = (CalleeOverride!="none") ? CalleeOverride : Callee;
  

  Function *TheFunction = Builder->GetInsertBlock()->getParent();
  std::string functionName = TheFunction->getName().str();
  std::string tgt_function_name;
  std::string msg;

  // std::cout << "\n\nFunction: " << tgt_function << "\n";


  int nested_function;
  if (functionName=="__anon_expr" || starts_with(functionName.c_str(), "__async_"))
    nested_function=0;
  else
    nested_function=1;


  bool has_scope = false;
  bool changed_first_arg = false;
  bool has_first_arg_copy = false;
  bool must_free_arg0 = false;

  Value *name;
  


  msg = "---\nCallExpr " + Callee;
  p2t(msg);



  Value *scope_struct_copy = callret("scope_struct_Copy", {scope_struct});
  Value *first_arg, *scope_string;


  




  msg = "\n\n\nCallExpr Function name: " + functionName;
  p2t(msg);
  msg = msg + "\n\n\n\n\n\n";
  p2t(msg);





  
  
  Value *_pre_dot_str = global_str(_pre_dot);
  Value *first_arg_copy;


  int target_args_size = Args.size();
  std::vector<Value *> ArgsV; 
  

  

  msg = "CallExpr Call name mangle";
  p2t(msg);

  

  bool is_self_of_nested_function = (nested_function==1 && isSelf);
  bool is_user_cpp_function = in_str(tgt_function, user_cpp_functions);
  bool is_nsk_fn = in_str(tgt_function, native_methods);
  

  
  
  call("set_scope_function_name", {scope_struct_copy, global_str(tgt_function)});



  p2t("CallExpr Finish mangle, get scope info.\n---Function Name: " + tgt_function);
  


  if (!(CalleeOverride!="none" || in_str(Callee, native_fn)) || Callee=="print_scope" || is_user_cpp_function) // user defined functions
  {
    has_scope = true;
    if(Callee!="print_scope" && !is_user_cpp_function)
    {
      scope_string = global_str(Scope_Random_Str);
      scope_string = callret("str_int_add", {scope_struct, scope_string, callret("get_scope_thread_id", {scope_struct})});
      call("set_scope_scope", {scope_struct_copy, scope_string});
    }
  }

  target_args_size+=1; //always add scope_struct






 

  p2t("CallExpr " + tgt_function + " check for args");

  // Detect function errors
  Function *CalleeF;
  // std::cout << "TGT_FUNCTION " <<  tgt_function << ".\n";
  // p2t("TGT FUNCTION " + tgt_function);

  CalleeF = getFunction(tgt_function);
  if (!CalleeF)
  {
    std::string _error = "The referenced function "+ tgt_function +" was not yet declared.";
    return LogErrorV(parser_struct.line, _error);
  }

  tgt_function_name = CalleeF->getName().str();

  // If argument mismatch error.
  if ((CalleeF->arg_size()) != target_args_size && !in_str(tgt_function_name, vararg_methods))
  {
    // std::cout << "CalleeF->arg_size() " << std::to_string(CalleeF->arg_size()) << " target_args_size " << std::to_string(target_args_size) << "\n";
    std::string _error = "Incorrect parameters used on function " + tgt_function + " call.\n\t    Expected " +  std::to_string(CalleeF->arg_size()-1) + " arguments, got " + std::to_string(target_args_size-1);
    return LogErrorV(parser_struct.line, _error);
  }
  // std::cout << "\n\n\nCalling function: " << tgt_function <<"\n";
  msg = "CallExpr Calling function: " + tgt_function;
  p2t(msg);






 




  // Sends the non-changed scope_struct to load/codegen the arguments from the argument list
  ArgsV = codegen_Argument_List(parser_struct, std::move(ArgsV), std::move(Args), scope_struct, tgt_function, is_nsk_fn);

  // Always include scope on the beggining
  ArgsV.insert(ArgsV.begin(), scope_struct_copy);



 
  

  
  Value *ret;
  if (CalleeOverride=="none")
  {
    ret = Builder->CreateCall(CalleeF, ArgsV, "calltmp");
    call("scope_struct_Delete", {scope_struct_copy});

    if (Type!="float"&&Type!="int"&&Type!="")
    {
      call("MarkToSweep_Mark", {scope_struct, ret, global_str(Extract_List_Suffix(Type))});
    }

    // std::cout << "Returning" << ".\n";
    return ret;
  }
  else
  {
    
    if (in_str(CalleeOverride, native_modules))
    {
      CalleeF = getFunction(CalleeOverride);
      
      if (CalleeF->arg_size() != ArgsV.size())
      {
        std::string _error = "Incorrect parameters used on function " + tgt_function + " call.";
        return LogErrorV(parser_struct.line, _error);
      }
      ret = Builder->CreateCall(CalleeF, ArgsV, "calltmp");
    }
    else
    {
      // p2t("**CALLEE " + CalleeOverride); 
      // call("print", {scope_struct_copy, callret("get_scope_first_arg", {scope_struct_copy})});
      ret = Builder->CreateCall(getFunction(CalleeOverride), ArgsV, "calltmp");
    }
    call("scope_struct_Delete", {scope_struct_copy});

    

    if (Type!="float"&&Type!=""&&Type!="int")
    {
      call("MarkToSweep_Mark", {scope_struct, ret, global_str(Extract_List_Suffix(Type))});
    }
    
    return ret;
  }  
}




Value *ChainCallExprAST::codegen(Value *scope_struct) {
  Value *inner_return = Inner_Call->codegen(scope_struct);

  std::vector<Value *> ArgsV; 

  ArgsV.push_back(inner_return);
  

  ArgsV = codegen_Argument_List(parser_struct, std::move(ArgsV), std::move(Args), scope_struct, Call_Of, true);


  ArgsV.insert(ArgsV.begin(), scope_struct);
  std::string call_fn = Call_Of;
  Value *ret = callret(call_fn, ArgsV);

  if (Type!="float"&&Type!="int"&&Type!="")
  {

    p2t("RETURN OF CHAIN CALL " + call_fn + " is " + Type);
    call("MarkToSweep_Mark", {scope_struct, ret, global_str(Extract_List_Suffix(Type))});
  }

  return ret;
}
