#pragma once


#include "llvm/IR/Value.h"

#include <string>
#include <map>
#include <vector>

#include "../data_types/include.h"
#include "../notators/include.h"
#include "../tensor/include.h"
#include "../KaleidoscopeJIT.h"
#include "include.h"



using namespace llvm;


PointerType *floatPtrTy, *int8PtrTy;


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



/// CreateEntryBlockAlloca - Create an alloca instruction in the entry block of
/// the function.  This is used for mutable variables etc.
AllocaInst *CreateEntryBlockAlloca(Function *TheFunction,
                                          StringRef VarName) {
  IRBuilder<> TmpB(&TheFunction->getEntryBlock(),
                   TheFunction->getEntryBlock().begin());
  return TmpB.CreateAlloca(Type::getFloatTy(*TheContext), nullptr, VarName);
}



Value *NumberExprAST::codegen(Value *scope_struct) {
  if (not ShallCodegen)
    return ConstantFP::get(*TheContext, APFloat(0.0f));

  std::string msg = "NumberExpr Num has value: " + std::to_string(Val);
  p2t(msg);

  return ConstantFP::get(*TheContext, APFloat(Val));
}

Value *StringExprAST::codegen(Value *scope_struct) {
  if (not ShallCodegen)
    return ConstantFP::get(*TheContext, APFloat(0.0f));
  SetName(Val);
  return Builder->CreateGlobalString(Val);
}


// Create Float Var
Value *VarExprAST::codegen(Value *scope_struct) {
  if (not ShallCodegen)
    return ConstantFP::get(*TheContext, APFloat(0.0f));

  return ConstantFP::get(*TheContext, APFloat(0.0));
}












Value *DataExprAST::codegen(Value *scope_struct) {
  p2t("DataExpr");
  if (not ShallCodegen)
    return ConstantFP::get(*TheContext, APFloat(0.0f));


  Function *TheFunction = Builder->GetInsertBlock()->getParent();

  // Register all variables and emit their initializer.
  for (unsigned i = 0, e = VarNames.size(); i != e; ++i) {
    const std::string &VarName = VarNames[i].first; 
    ExprAST *Init = VarNames[i].second.get();
    
    Value *var_name, *scopeless_name, *init;
    



    p2t("DataExpr get var name");
    var_name = Builder->CreateCall(TheModule->getFunction("CopyString"),
                                            {Builder->CreateGlobalString(VarName)});

    bool is_self = GetSelf();
    bool is_attr = GetIsAttribute();


    p2t("DataExpr name mangle");

    if (is_self||is_attr)
      var_name = Builder->CreateCall(TheModule->getFunction("ConcatStrFreeRight"),
                                            {Builder->CreateCall(TheModule->getFunction("get_scope_first_arg"), {scope_struct}), var_name});
    scopeless_name = Builder->CreateCall(TheModule->getFunction("CopyString"),
                                            {var_name});
    if (!(is_self||is_attr))
      var_name = Builder->CreateCall(TheModule->getFunction("ConcatStrFreeRight"),
                                            {Builder->CreateCall(TheModule->getFunction("get_scope_scope"), {scope_struct}), var_name});



    p2t("DataExpr Create nodes vector");

    Value *notes_vector = Builder->CreateCall(TheModule->getFunction("CreateNotesVector"),
                                            {});


    for (int j=0; j<Notes.size(); j++)
    {
      ExprAST *note = Notes[j].get();
      if (NumberExprAST* numExpr = dynamic_cast<NumberExprAST*>(note)) {
        
        notes_vector = Builder->CreateCall(TheModule->getFunction("Add_Float_To_NotesVector"),
                                                {notes_vector, note->codegen(scope_struct)});
                                                // {notes_vector});
      }
      else if (StringExprAST* expr = dynamic_cast<StringExprAST*>(note)) {
        Value *str_val = Builder->CreateCall(TheModule->getFunction("CopyString"),
                                            {note->codegen(scope_struct)});
        notes_vector = Builder->CreateCall(TheModule->getFunction("Add_String_To_NotesVector"),
                                                {notes_vector, str_val});
      }
      else if (VariableExprAST* expr = dynamic_cast<VariableExprAST*>(note)) {
        notes_vector = Builder->CreateCall(TheModule->getFunction("Add_Float_To_NotesVector"),
                                                {notes_vector, note->codegen(scope_struct)});
      }
      else {
        std::cout << "Could not find the data type\n";
      }

    }
    



    // if (Type=="tensor")
    //   call("scope_struct_Print", {scope_struct});

    
    p2t("DataExpr Call create");

    std::string create_fn = Type + "_Create";


    Builder->CreateCall(TheModule->getFunction(create_fn),
                                              {var_name, scopeless_name, Init->codegen(scope_struct), notes_vector,
                                               scope_struct});
    
    p2t("DataExpr Dispose notes vector");

    Builder->CreateCall(TheModule->getFunction("Dispose_NotesVector"), {notes_vector});

    
  }


  return ConstantFP::get(*TheContext, APFloat(0.0));
}






Value *IfExprAST::codegen(Value *scope_struct) {
  if (not ShallCodegen)
    return ConstantFP::get(*TheContext, APFloat(0.0f));


  Value *CondV = Cond->codegen(scope_struct);
  if (!CondV)
    return nullptr;

  // Convert condition to a bool by comparing equal to 0.0.
  CondV = Builder->CreateFCmpONE(
      CondV, ConstantFP::get(*TheContext, APFloat(0.0)), "ifcond");

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
    return ConstantFP::get(*TheContext, APFloat(0.0f));
  Function *TheFunction = Builder->GetInsertBlock()->getParent();

  // Create an alloca for the variable in the entry block.
  AllocaInst *Alloca = CreateEntryBlockAlloca(TheFunction, VarName);

  // Emit the start code first, without 'variable' in scope.
  Value *StartVal = Start->codegen(scope_struct);
  if (!StartVal)
    return nullptr;

  Value *_zero = ConstantFP::get(*TheContext, APFloat(0.0));



  Value *var_name = Builder->CreateGlobalString(VarName);
  var_name = Builder->CreateCall(TheModule->getFunction("ConcatStr"),
                                    {Builder->CreateCall(TheModule->getFunction("get_scope_scope"), {scope_struct}), var_name});

  Builder->CreateCall(TheModule->getFunction("StoreOnDemandNoFree"),
                                                  {var_name, StartVal});

  // Store the value into the alloca.
  //Builder->CreateStore(StartVal, Alloca);

  // Make the new basic block for the loop header, inserting after current
  // block.
  BasicBlock *CondBB = BasicBlock::Create(*TheContext, "cond", TheFunction);
  BasicBlock *LoopBB  = BasicBlock::Create(*TheContext, "loop");
  BasicBlock *AfterBB  = BasicBlock::Create(*TheContext, "after");



  // Insert an explicit fall through from the current block to the LoopBB.
  Builder->CreateBr(CondBB);

  
  Builder->SetInsertPoint(CondBB);

  // Within the loop, the variable is defined equal to the PHI node.  If it
  // shadows an existing variable, we have to restore it outside this scope
  //Value *OldVal = NamedValues[VarName];
  //NamedValues[VarName] = Alloca;



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


  // Compute the end condition.
  Value *EndCond = End->codegen(scope_struct);
  if (!EndCond)
    return nullptr;

  // Convert condition to a bool by comparing equal to 0.0.
  EndCond = Builder->CreateFCmpONE(
      EndCond, _zero, "loopcond");




  // conditional goto branch
  Builder->CreateCondBr(EndCond, LoopBB, AfterBB);




  // codegen body and increment
  TheFunction->insert(TheFunction->end(), LoopBB);
  Builder->SetInsertPoint(LoopBB);

  int j=0;
  for (auto &body : Body)
    body->codegen(scope_struct);

  // Reload, increment, and restore the alloca.  This handles the case where
  // the body of the loop mutates the variable.
  Value *CurVar = Builder->CreateCall(TheModule->getFunction("LoadOnDemandNoFree"), {var_name});
  Value *NextVar = Builder->CreateFAdd(CurVar, StepVal, "nextvar"); // Increment
  Builder->CreateCall(TheModule->getFunction("StoreOnDemandNoFree"),
                                                  {var_name, NextVar});

  
  
  Builder->CreateBr(CondBB);




  // when the loop body is done, return the insertion point to outside the for loop
  TheFunction->insert(TheFunction->end(), AfterBB);
  Builder->SetInsertPoint(AfterBB);

  Builder->CreateCall(TheModule->getFunction("FreeChar"), {var_name});
  // Restore the unshadowed variable.
  //if (OldVal)
  //  NamedValues[VarName] = OldVal;
  //else
  //  NamedValues.erase(VarName);

  // for expr always returns 0.0.
  return Constant::getNullValue(Type::getFloatTy(*TheContext));
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

  Value *_zero = ConstantFP::get(*TheContext, APFloat(0.0));
  condVal = Builder->CreateFCmpONE(condVal, _zero, "loopcond");

  // Create the conditional branch
  Builder->CreateCondBr(condVal, LoopBB, AfterBB);

  // Insert the loop body block
  Builder->SetInsertPoint(LoopBB);

  // Generate the loop body code
  for (auto &body : Body)
    body->codegen(scope_struct);

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
  
  //std::cout << "Create value V" << "\n";
  Value * ret = ConstantFP::get(*TheContext, APFloat(0.0f));
  Value *V, *var_name;

  std::string type = GetType();
  std::string pre_dot = GetPreDot();
  bool is_self = GetSelf();
  bool is_attr = GetIsAttribute();
  
  //std::string __print = "\n\nLOAD OF " + std::string(Name) + " ";
  //Builder->CreateCall(TheModule->getFunction("print"),
  //    {Builder->CreateGlobalString(__print), ConstantFP::get(*TheContext, APFloat(0.0f))});

  var_name = NameSolver->codegen(scope_struct);
  NameSolverAST *name_solver = static_cast<NameSolverAST *>(NameSolver.get());
  std::string Name = std::get<0>(name_solver->Names[name_solver->Names.size()-1]);
  
  std::string msg = "VariableExpr Variable " + Name + " load for type: " + type;
  p2t(msg);



  if (type=="str")
  {
    for (const auto &entry : NamedTensorsT)
    {
      std::cout << "Returning None because a tensor with name " << Name << " was found on strings map " << "\n";
      if (ends_with(entry.first, Name))
      return ret;
    } 
  }
  if (type=="object")
    return var_name;
  if (type=="tensor" && !seen_var_attr)
  {
    Builder->CreateCall(TheModule->getFunction("PrintTensor"), {Builder->CreateCall(TheModule->getFunction("get_scope_thread_id"), {scope_struct}), var_name});
    return ConstantFP::get(*TheContext, APFloat(0.0f));
  }
  

  std::string load_fn = type + "_Load";
  // std::cout << "Load fn: " << load_fn << ".\n";
  V = Builder->CreateCall(TheModule->getFunction(load_fn),
                                                  {var_name, scope_struct});
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
  
  
  Value * ret = ConstantFP::get(*TheContext, APFloat(0.0f));
  Value *V, *idx;

  if (Type!="object_vec")
    idx = Idx[0]->codegen(scope_struct);


  Value *var_name, *object_name, *object_var_name;
  var_name = NameSolver->codegen(scope_struct);
  
  NameSolverAST *name_solver = static_cast<NameSolverAST *>(NameSolver.get());
  std::string Name = std::get<0>(name_solver->Names[0]);
  




  std::string pre_dot = GetPreDot();
  bool is_self = GetSelf();
  bool is_attr = GetIsAttribute();
  std::cout << "is self: " << is_self << ", is_attr: " << is_attr << "\n";

  if (is_self||is_attr)
  {
    
    if (Type=="str_vec"){
      
      V = Builder->CreateCall(TheModule->getFunction("IndexClassStrVec"), {var_name, idx});
      
      return V;
    }

    if (Type=="float_vec"){
      V = Builder->CreateCall(TheModule->getFunction("IndexClassFloatVec"), {var_name, idx});
      return V;
    }

    if (Type=="object_vec")
      return var_name;
  }


  if (Type=="str_vec")
  {
    return nullptr;
  }
  if (Type=="float_vec")
  {


    V = Builder->CreateCall(TheModule->getFunction("IndexFloatVec"), {V, idx});

    return V;
  }

  if (Type=="tensor")
  {
    std::cout << "vec idx of tensor, idx type: " << Idx[0]->GetType() << "\n";

    if (Idx[0]->GetType()!="tensor")
    {
      /*
      std::vector<Value *> idx_calc_args;
      idx_calc_args.push_back(var_name);
      for (int i=0; i<Idx.size(); i++)
        idx_calc_args.push_back(Idx[i]->codegen(scope_struct));
      Value *idx_at = Builder->CreateCall(TheModule->getFunction("CalculateIdxOffset"),
                          idx_calc_args);

      return Builder->CreateCall(TheModule->getFunction("IdxTensor"), {var_name, idx_at, scope_string, thread_id});
      */
      std::vector<Value *> idx_calc_args;
      idx_calc_args.push_back(var_name);
      idx_calc_args.push_back(Builder->CreateCall(TheModule->getFunction("get_scope_scope"), {scope_struct}));
      idx_calc_args.push_back(Builder->CreateCall(TheModule->getFunction("get_scope_thread_id"), {scope_struct}));
      for (int i=0; i<Idx.size(); i++)
        idx_calc_args.push_back(Idx[i]->codegen(scope_struct));

      return Builder->CreateCall(TheModule->getFunction("IdxTensor"), idx_calc_args);
    } else {
      VariableExprAST *idx = static_cast<VariableExprAST *>(Idx[0].get());
      Value *idx_tensor_name = idx->NameSolver->codegen(scope_struct);
      
      return Builder->CreateCall(TheModule->getFunction("IdxTensorWithTensor"), {var_name, idx_tensor_name, Builder->CreateCall(TheModule->getFunction("get_scope_thread_id"), {scope_struct})});
      
    }
    
  }

  std::string _error = "Unknown vector: " + Name + ".";
  LogErrorS(_error);
  std::cout << "Type " << Type << "\n";

  return ret;
}


Value *ObjectVecIdxExprAST::codegen(Value *scope_struct) {
  if (not ShallCodegen)
    return ConstantFP::get(*TheContext, APFloat(0.0f));
  // Look this variable up in the function.
  std::cout << "ObjectVecIdxExprAST codegen" << "\n";
  
  VecIdxExprAST *vec = static_cast<VecIdxExprAST *>(Vec.get());
  std::cout << "vec name " << vec->GetName() << "\n";
  std::cout << "ObjectVecIdxExprAST is vec: " << GetIsVec() << "\n";

  Value *idx = vec->Idx[0]->codegen(scope_struct);


  Value *var_name, *object_name, *object_var_name, *post_dot_str;
  var_name = Builder->CreateGlobalString(vec->GetName());
  post_dot_str = Builder->CreateGlobalString(_post_dot);
  
  std::string pre_dot = GetPreDot();
  bool is_self = GetSelf();
  bool is_attr = GetIsAttribute();
  
  
  if (is_self||is_attr)
  {
    // Gets from pre_dot if it is a class attribute
    if (is_attr) {
      object_name = Builder->CreateGlobalString(pre_dot);
      var_name = Builder->CreateGlobalString(Name);

      var_name = Builder->CreateCall(TheModule->getFunction("ConcatStr"),
                                                      {object_name, var_name});
    }
    if (is_self)
      var_name = Builder->CreateCall(TheModule->getFunction("ConcatStr"),
                                                      {Builder->CreateLoad(int8PtrTy, Builder->CreateCall(TheModule->getFunction("get_scope_first_arg"), {scope_struct})), var_name});
  }

  if (Type=="tensor")
    return Builder->CreateCall(TheModule->getFunction("object_vec_idxTensor"),
                                                      {var_name, idx, post_dot_str});
  if (Type=="object")
    return Builder->CreateCall(TheModule->getFunction("object_vec_idxObject"),
                                                      {var_name, idx, post_dot_str});


  return ConstantFP::get(*TheContext, APFloat(0.0f));
}








Value *BinaryTensorScalarExprAST::codegen(Value *scope_struct) {
  if (not ShallCodegen)
    return ConstantFP::get(*TheContext, APFloat(0.0f));

  Value *tensor_name = Builder->CreateGlobalString(LHS->GetName());



  std::string pre_dot = LHS->GetPreDot();
  bool is_self = LHS->GetSelf();
  bool is_attr = LHS->GetIsAttribute();

  if (is_attr) { // Gets from pre_dot if it is a class attribute
    Value * object_name = Builder->CreateGlobalString(pre_dot);

    tensor_name = Builder->CreateCall(TheModule->getFunction("ConcatStr"),
                                                      {object_name, tensor_name});
  }
  if (is_self)
    tensor_name = Builder->CreateCall(TheModule->getFunction("ConcatStr"),
                                                      {Builder->CreateCall(TheModule->getFunction("get_scope_first_arg"), {scope_struct}), tensor_name});
  if (!(is_self||is_attr))
    tensor_name = Builder->CreateCall(TheModule->getFunction("ConcatStr"),
                                            {Builder->CreateCall(TheModule->getFunction("get_scope_scope"), {scope_struct}), tensor_name});
    



  // Special case '=' because we don't want to emit the LHS as an expression.
  if (Op == '=') {
    seen_var_attr=true;
    // Assignment requires the LHS to be an identifier.
    // This assume we're building without RTTI because LLVM builds that way by
    // default.  If you build LLVM with RTTI this can be changed to a
    // dynamic_cast for automatic error checking.
    VariableExprAST *LHSE = static_cast<VariableExprAST *>(LHS.get());
    if (!LHSE)
      return LogErrorV("'=' destiny must be a var.");
    // Codegen the RHS.
    
    Value *Val = RHS->codegen(scope_struct);
    if (!Val)
      return nullptr;

    
    
    std::cout << "1 0 attr\n";
    


    //LogErrorS("Attribution from float into tensor is not possible.");    
    
    
      
    
    seen_var_attr=false;
    return Val;
  }


  std::cout << "\n\n\nTensor scalar for LHS: " << LHS->GetName() << " RHS: " << RHS->GetName() << "\n\n\n";
  Value *LtensorPtr = LHS->codegen(scope_struct);
  Value *R = RHS->codegen(scope_struct);
  std::cout << "\n\n\nTensor scalar post codegen" << "\n\n\n";



  if (!LtensorPtr || !R)
    return nullptr;



  /*
  std::cout << "\nTensorScalar, LHS is self: " << LHS->GetSelf() << "\n";
  Function *TheFunction = Builder->GetInsertBlock()->getParent();
  std::string functionName = TheFunction->getName().str();
  std::cout << "Fname: " << functionName << "\n\n";
  */
  



  switch (Op)
  {
  case '*':
    return Builder->CreateCall(TheModule->getFunction("CudaScalarMult"),
                               {LtensorPtr, R, scope_struct}, "cudascalarmult");
  case '/':
    return Builder->CreateCall(TheModule->getFunction("CudaScalarDiv"),
                               {LtensorPtr, R, scope_struct}, "cudascalardiv");
  case 77:
    return Builder->CreateCall(TheModule->getFunction("CudaReverseScalarDiv"),
                               {LtensorPtr, R, scope_struct}, "cudareversescalardiv");
  case '+':
    return Builder->CreateCall(TheModule->getFunction("CudaScalarAdd"),
                               {LtensorPtr, R, scope_struct}, "cudascalaradd");
  case '-':
    return Builder->CreateCall(TheModule->getFunction("CudaScalarSub"),
                               {LtensorPtr, R, scope_struct}, "cudascalarsub");
  case tok_equal:
    return Builder->CreateCall(TheModule->getFunction("CudaScalarEqual"),
                               {LtensorPtr, R, scope_struct}, "cudascalarequal");
  case tok_diff:
    return Builder->CreateCall(TheModule->getFunction("CudaScalarDiff"),
                               {LtensorPtr, R, scope_struct}, "cudascalardiff");
  case '<':
    return Builder->CreateCall(TheModule->getFunction("CudaScalarMinor"),
                               {LtensorPtr, R, scope_struct}, "cudascalarminor");
  case '>':
    return Builder->CreateCall(TheModule->getFunction("CudaScalarHigher"),
                               {LtensorPtr, R, scope_struct}, "cudascalarhigher");
  case tok_minor_eq:
    return Builder->CreateCall(TheModule->getFunction("CudaScalarMinorEq"),
                               {LtensorPtr, R, scope_struct}, "cudascalarminoreq");
  case tok_higher_eq:
    return Builder->CreateCall(TheModule->getFunction("CudaScalarHigherEq"),
                               {LtensorPtr, R, scope_struct}, "cudascalarhighereq");
  case ':':
    return LtensorPtr;
  case tok_space:
    return R;
  default:
    break;
  }
  

  // If it wasn't a builtin binary operator, it must be a user defined one. Emit
  // a call to it.
  Function *F = getFunction(std::string("binary") + Op);
  assert(F && "Operator not found.");

  Value *Ops[] = {LtensorPtr, R};
  return Builder->CreateCall(F, Ops, "binop");
}



Value *BinaryPinnedScalarExprAST::codegen(Value *scope_struct) {
  if (not ShallCodegen)
    return ConstantFP::get(*TheContext, APFloat(0.0f));

  Value *tensor_name;



  

  if (Op == '=') {
    seen_var_attr=true;

    
    Value *Val = RHS->codegen(scope_struct);
    if (!Val)
      return nullptr;
    
    std::cout << "2 0 attr\n";
    std::cout << "is vec: " << LHS->GetIsVec()  << "\n";


    


    VecIdxExprAST   *LHSE = static_cast<VecIdxExprAST *>(LHS.get());
    tensor_name = LHSE->NameSolver->codegen(scope_struct);

    if (!LHSE)
      return LogErrorV("'=' destiny must be a variable.");

    

    std::vector<Value *> idx_calc_args;

    idx_calc_args.push_back(tensor_name);

    for (int i=0; i<LHSE->Idx.size(); i++)
    {
      idx_calc_args.push_back(LHSE->Idx[i]->codegen(scope_struct));
    }

    Value *idx_at = Builder->CreateCall(TheModule->getFunction("CalculateIdxOffset"),
                          idx_calc_args);

    Builder->CreateCall(TheModule->getFunction("AttrPinnedOnIdx"),
                          {tensor_name, Val, idx_at});


    /*
    if (LHS->GetIsVec())
    {
      VecIdxExprAST   *LHSE = static_cast<VecIdxExprAST *>(LHS.get());
      if (!LHSE)
        return LogErrorV("'=' destiny must be a variable.");
      std::cout << "is vec: " << LHS->GetIsVec()  << "\n";

      Builder->CreateCall(TheModule->getFunction("AttrPinnedOnIdx"),
                          {tensor_name, LHSE->Idx->codegen(scope_struct), Val});
    }
      
    else
    {
      VariableExprAST *LHSE = static_cast<VariableExprAST *>(LHS.get());
      if (!LHSE)
        return LogErrorV("'=' destiny must be a variable.");
    }
    */

    


    
    
    
      
    
    seen_var_attr=false;
    return Val;
  }


  
  Value *LtensorPtr = LHS->codegen(scope_struct);
  Value *R = RHS->codegen(scope_struct);
  



  
  
  if (!LtensorPtr || !R)
    return nullptr;



  /*
  std::cout << "\nTensorScalar, LHS is self: " << LHS->GetSelf() << "\n";
  Function *TheFunction = Builder->GetInsertBlock()->getParent();
  std::string functionName = TheFunction->getName().str();
  std::cout << "Fname: " << functionName << "\n\n";
  */
  



  switch (Op)
  {
  case '*':
    return Builder->CreateCall(TheModule->getFunction("CudaScalarMult"),
                               {LtensorPtr, R, scope_struct}, "cudascalarmult");
  case '/':
    return Builder->CreateCall(TheModule->getFunction("CudaScalarDiv"),
                               {LtensorPtr, R, scope_struct}, "cudascalardiv");
  case 77:
    return Builder->CreateCall(TheModule->getFunction("CudaReverseScalarDiv"),
                               {LtensorPtr, R, scope_struct}, "cudareversescalardiv");
  case '+':
    return Builder->CreateCall(TheModule->getFunction("CudaScalarAdd"),
                               {LtensorPtr, R, scope_struct}, "cudascalaradd");
  case '-':
    return Builder->CreateCall(TheModule->getFunction("CudaScalarSub"),
                               {LtensorPtr, R, scope_struct}, "cudascalarsub");
  case ':':
    return LtensorPtr;
  case tok_space:
    return R;
  default:
    break;
  }
  

  // If it wasn't a builtin binary operator, it must be a user defined one. Emit
  // a call to it.
  Function *F = getFunction(std::string("binary") + Op);
  assert(F && "Operator not found.");

  Value *Ops[] = {LtensorPtr, R};
  return Builder->CreateCall(F, Ops, "binop");
}







Value *BinaryPinnedAndTensorExprAST::codegen(Value *scope_struct) {
  if (not ShallCodegen)
    return ConstantFP::get(*TheContext, APFloat(0.0f));

  Value *tensor_name;



  

  if (Op == '=') {
    seen_var_attr=true;

    
    Value *RtensorPtr = RHS->codegen(scope_struct);
    if (!RtensorPtr)
      return nullptr;
    
    std::cout << "2 0 attr\n";
    std::cout << "is vec: " << LHS->GetIsVec()  << "\n";


    


    VecIdxExprAST   *LHSE = static_cast<VecIdxExprAST *>(LHS.get());
    tensor_name = LHSE->NameSolver->codegen(scope_struct);

    if (!LHSE)
      return LogErrorV("'=' destiny must be a variable.");

    

    std::vector<Value *> idx_calc_args;

    idx_calc_args.push_back(tensor_name);
    idx_calc_args.push_back(RtensorPtr);
    idx_calc_args.push_back(Builder->CreateCall(TheModule->getFunction("get_scope_thread_id"), {scope_struct}));


    
    for (int i=0; i<LHSE->Idx.size(); i++)
      idx_calc_args.push_back(LHSE->Idx[i]->codegen(scope_struct));


    Builder->CreateCall(TheModule->getFunction("AttrPinnedFromTensorOnIdx"),
                         idx_calc_args);


    /*
    if (LHS->GetIsVec())
    {
      VecIdxExprAST   *LHSE = static_cast<VecIdxExprAST *>(LHS.get());
      if (!LHSE)
        return LogErrorV("'=' destiny must be a variable.");
      std::cout << "is vec: " << LHS->GetIsVec()  << "\n";

      Builder->CreateCall(TheModule->getFunction("AttrPinnedFromTensorOnIdx"),
                          {tensor_name, LHSE->Idx->codegen(scope_struct), Val});
    }
      
    else
    {
      VariableExprAST *LHSE = static_cast<VariableExprAST *>(LHS.get());
      if (!LHSE)
        return LogErrorV("'=' destiny must be a variable.");
    }
    */

      
    
    seen_var_attr=false;
    return RtensorPtr;
  }


  
  Value *LtensorPtr = LHS->codegen(scope_struct);
  Value *R = RHS->codegen(scope_struct);
  



  
  
  if (!LtensorPtr || !R)
    return nullptr;




  switch (Op)
  {
  case '*':
    return Builder->CreateCall(TheModule->getFunction("CudaScalarMult"),
                               {LtensorPtr, R, scope_struct}, "cudascalarmult");
  case '/':
    return Builder->CreateCall(TheModule->getFunction("CudaScalarDiv"),
                               {LtensorPtr, R, scope_struct}, "cudascalardiv");
  case 77:
    return Builder->CreateCall(TheModule->getFunction("CudaReverseScalarDiv"),
                               {LtensorPtr, R, scope_struct}, "cudareversescalardiv");
  case '+':
    return Builder->CreateCall(TheModule->getFunction("CudaScalarAdd"),
                               {LtensorPtr, R, scope_struct}, "cudascalaradd");
  case '-':
    return Builder->CreateCall(TheModule->getFunction("CudaScalarSub"),
                               {LtensorPtr, R, scope_struct}, "cudascalarsub");
  case ':':
    return LtensorPtr;
  case tok_space:
    return R;
  default:
    break;
  }
  

  // If it wasn't a builtin binary operator, it must be a user defined one. Emit
  // a call to it.
  Function *F = getFunction(std::string("binary") + Op);
  assert(F && "Operator not found.");

  Value *Ops[] = {LtensorPtr, R};
  return Builder->CreateCall(F, Ops, "binop");
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


    VariableExprAST *LHSE = static_cast<VariableExprAST *>(LHS.get());
    Value *Lvar_name = LHSE->NameSolver->codegen(scope_struct);


    NameSolverAST *name_solver = static_cast<NameSolverAST *>(LHSE->NameSolver.get());
    std::string Lname = std::get<0>(name_solver->Names[0]);
    std::string LType = LHS->GetType();


    if (!LHSE)
      return LogErrorV("'=' destiny must be a variable.");

    // Codegen the RHS.
    Value *Val = RHS->codegen(scope_struct);
    if (!Val)
    {
      seen_var_attr=false;
      return nullptr;
      
    }

    std::string store_op = LType + "_Store";



    if(LHS->GetIsVec())
    {
      VecIdxExprAST *LHSV = static_cast<VecIdxExprAST *>(LHS.get());

      store_op = store_op + "_Idx";
      
      Builder->CreateCall(TheModule->getFunction(store_op),
                                              {Lvar_name,
                                                LHSV->Idx[0]->codegen(scope_struct),
                                                Val, scope_struct});

    } else { 
        Builder->CreateCall(TheModule->getFunction(store_op), {Lvar_name, Val, scope_struct});
    }
    

    seen_var_attr=false;
    return ConstantFP::get(*TheContext, APFloat(0.0f));
  }


  

  Value *L = LHS->codegen(scope_struct);
  Value *R = RHS->codegen(scope_struct);
  
  if (!L || !R)
    return nullptr;


  // if (Elements=="float_float")
  // {

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
      case 77:
        return LogErrorV("GOTCHA");
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
      default:
        break;
      }
  // } else {
    // std::cout << "Codegen for operation: " << Operation << ".\n";

    

    // return ConstantFP::get(*TheContext, APFloat(0.0f));
  // }



  

  // If it wasn't a builtin binary operator, it must be a user defined one. Emit
  // a call to it.
  Function *F = getFunction(std::string("binary") + Op);
  assert(F && "Operator not found.");

  Value *Ops[] = {L, R};
  return Builder->CreateCall(F, Ops, "binop");
}



Value *BinaryTensorTensorExprAST::codegen(Value *scope_struct) {
  if (not ShallCodegen)
    return ConstantFP::get(*TheContext, APFloat(0.0f));

  Value *LtensorName = Builder->CreateGlobalString(LHS->GetName());
  Value *RtensorName = Builder->CreateGlobalString(RHS->GetName());
  Value *object_name;


  
  // if is attribution
  if (Op == '=') {
  
    seen_var_attr=true;

    Value *RtensorPtr = RHS->codegen(scope_struct);
    

    if (!LHS->GetIsVec())
    {
      VariableExprAST *LHSE = static_cast<VariableExprAST *>(LHS.get());
      LtensorName = LHSE->NameSolver->codegen(scope_struct);

      if (!LHSE)
        return LogErrorV("'=' left side expression must be a var.");
      
      //std::cout << "1 1 attr\n";


      Builder->CreateCall(TheModule->getFunction("AttrTensor"),
                          {LtensorName, RtensorPtr, scope_struct});
      //std::cout << "Post attr call\n\n";
    } else
    {
      std::cout << "1 1 INDEXED attr\n";

      VecIdxExprAST *LHSE = static_cast<VecIdxExprAST *>(LHS.get());
      LtensorName = LHSE->NameSolver->codegen(scope_struct);
      if (!LHSE)
        return LogErrorV("'=' left side expression must be a var.");

      if(LHSE->Idx[0]->GetType()!="tensor")
      {
        std::vector<Value *> idx_calc_args;
        idx_calc_args.push_back(LtensorName);
        for (int i=0; i<LHSE->Idx.size(); i++)
          idx_calc_args.push_back(LHSE->Idx[i]->codegen(scope_struct));
        Value *idx_at = Builder->CreateCall(TheModule->getFunction("CalculateIdxOffset"),
                              idx_calc_args);

        Builder->CreateCall(TheModule->getFunction("AttrTensorOnIdx"),
                            {LtensorName, RtensorPtr,
                             idx_at, Builder->CreateCall(TheModule->getFunction("get_scope_thread_id"), {scope_struct})});
      } else {
        VariableExprAST *idx = static_cast<VariableExprAST *>(LHSE->Idx[0].get());
        Value *idx_tensor_name = idx->NameSolver->codegen(scope_struct);
        
        Builder->CreateCall(TheModule->getFunction("AttrTensorOnIdxTensor"), {LtensorName, idx_tensor_name, RtensorPtr, Builder->CreateCall(TheModule->getFunction("get_scope_thread_id"), {scope_struct})});

      }
    }

    seen_var_attr=false;
    return ConstantFP::get(*TheContext, APFloat(0.0f));
  }




  std::string functionName = Builder->GetInsertBlock()->getParent()->getName().str();
  std::cout << "\nTensor Tensor for function: " << functionName << "\n";
  int forward_func = 0;
  if(ends_with(functionName, "forward"))
    forward_func = 1;
  forward_func = 1; // TODO: RemoveLastDim this line



  
  Value *LtensorPtr = LHS->codegen(scope_struct);
  Value *RtensorPtr = RHS->codegen(scope_struct);



  if (!LtensorPtr || !RtensorPtr)
    return nullptr;

  Function *CudaFn;

  std::cout << "Tensor tensor: " << LHS->GetName() << ", " << RHS->GetName() << "\n";
    


  Value *is_forward_func = ConstantInt::get(Type::getInt32Ty(*TheContext), forward_func);
  
  /*
  void *vec = &NamedDims[LHS->GetName()];
  Value* LLVMValue = ConstantInt::get(Type::getInt64Ty(*TheContext), reinterpret_cast<uint64_t>(vec));
  LLVMValue = Builder->CreateIntToPtr(LLVMValue, int8PtrTy);
  */

  
  Value *new_dims;

  switch (Op)
  {
  case '@':
    return Builder->CreateCall(TheModule->getFunction("CudaMult"),
                                    {is_forward_func,
                                     LtensorPtr, RtensorPtr, Builder->CreateCall(TheModule->getFunction("get_scope_thread_id"), {scope_struct})});
  case '/':
  {
    return Builder->CreateCall(TheModule->getFunction("CudaDiv"),
                                    {is_forward_func,
                                     LtensorPtr, RtensorPtr, Builder->CreateCall(TheModule->getFunction("get_scope_thread_id"), {scope_struct})});
  }
  case '+':
    return Builder->CreateCall(TheModule->getFunction("CudaAdd"),
                                    {is_forward_func,
                                     LtensorPtr, RtensorPtr, Builder->CreateCall(TheModule->getFunction("get_scope_thread_id"), {scope_struct})});
  case '*':
    return Builder->CreateCall(TheModule->getFunction("CudaHadamard"),
                                    {is_forward_func,
                                     LtensorPtr, RtensorPtr, Builder->CreateCall(TheModule->getFunction("get_scope_thread_id"), {scope_struct})});
  case '-':
    return Builder->CreateCall(TheModule->getFunction("CudaSub"),
                                    {is_forward_func,
                                     LtensorPtr, RtensorPtr, Builder->CreateCall(TheModule->getFunction("get_scope_thread_id"), {scope_struct})});
  case tok_equal:
    return Builder->CreateCall(TheModule->getFunction("CudaEqual"),
                               {is_forward_func, LtensorPtr, RtensorPtr, Builder->CreateCall(TheModule->getFunction("get_scope_thread_id"), {scope_struct})}, "cudaequal");
  case ':':
    return LtensorPtr;
  default:
    break;
  }
  

  std::string _error = "The operator " + ReverseToken(Op) + " is not implemented for operations between tensors";
  LogErrorS(_error);
  
  Function *F = getFunction(std::string("binary") + Op);
  assert(F && "Operator not found.");

  Value *Ops[] = {LtensorName, RtensorName};
  return Builder->CreateCall(F, Ops, "binop");
}




Value *CallExprAST::codegen(Value *scope_struct) {
  if (not ShallCodegen)
    return ConstantFP::get(*TheContext, APFloat(0.0f));
  // Look up the name in the global module table.
  std::string tgt_function = Callee;
  

  Function *TheFunction = Builder->GetInsertBlock()->getParent();
  std::string functionName = TheFunction->getName().str();
  std::string tgt_function_name;
  std::string msg;

  //std::cout << "\n\nFunction: " << tgt_function << "\n";

  int nested_function;
  if (functionName=="__anon_expr" || starts_with(functionName.c_str(), "__async_"))
  {
    nested_function=0;
  }
  else
    nested_function=1;


  bool has_scope = false;
  bool changed_first_arg = false;
  bool has_first_arg_copy = false;
  bool must_free_arg0 = false;

  Value *name;
  

  int thread = 0;

  msg = "---\nCallExpr " + Callee;
  p2t(msg);
  p2t("CallExpr Copy scope struct");


  // if (auto *inst = llvm::dyn_cast<llvm::Instruction>(scope_struct)) {
  //   std::cout << "--------------------------Operand belongs to function: " << inst->getFunction()->getName().str() << "\n";
  // } else {
  //     std::cout << "-------------------------Operand is not an Instruction (e.g., maybe a GlobalValue, Constant, etc.)\n";
  // }
  // std::cout << "----------------------Current function: " << functionName << ", codegen for: " << Callee <<  "\n";

  scope_struct = Builder->CreateCall(TheModule->getFunction("scope_struct_Copy"), {scope_struct}); 


  
  
  Value *first_arg, *scope_string, *previous_scope, *thread_id, *has_grad;


  //TODO: Solve scope_string discontinuity on async functions
  if (starts_with(functionName.c_str(), "__async_"))
  {
    msg = "\n\n\n\n\nCallExpr ASYNC\n\n\n\n\n";
    p2t(msg);

    scope_string = Builder->CreateCall(TheModule->getFunction("GetEmptyChar"), {});

    std::string copy = functionName;
    std::string prefix = "__async_";

    size_t pos = copy.find(prefix);
    copy.erase(pos, prefix.length());
    thread = std::stoi(copy);
    thread_id = ConstantInt::get(Type::getInt32Ty(*TheContext), thread);
    has_grad  = ConstantInt::get(Type::getInt32Ty(*TheContext), 1);

    Builder->CreateCall(TheModule->getFunction("set_scope_scope"), {scope_struct, scope_string});
    Builder->CreateCall(TheModule->getFunction("set_scope_thread_id"), {scope_struct, thread_id});
    Builder->CreateCall(TheModule->getFunction("set_scope_has_grad"), {scope_struct, has_grad}); 

  }
  



  // call("scope_struct_Print", {scope_struct});

  msg = "\n\n\nCallExpr Function name: " + functionName;
  p2t(msg);
  msg = "CallExpr THREAD IS: " + std::to_string(thread);
  msg = msg + "\n\n\n\n\n\n";
  p2t(msg);



  //Builder->CreateCall(TheModule->getFunction("FreeChar"), {previous_scope});
  previous_scope = Builder->CreateCall(TheModule->getFunction("CopyString"),
                                        {Builder->CreateCall(TheModule->getFunction("get_scope_scope"), {scope_struct})});


  Builder->CreateCall(TheModule->getFunction("set_scope_previous_scope"), {scope_struct, previous_scope});


  Value *_pre_dot_str = Builder->CreateGlobalString(_pre_dot);
  Value *first_arg_copy;




  

  if (isAttribute && !isSelf && !in_str(tgt_function, native_methods))
  { // e.g: model.forward()
    p2t("CallExpr Calling object method");
    if (nested_function)
    {
      first_arg_copy = Builder->CreateCall(TheModule->getFunction("CopyString"),
                                                    {Builder->CreateCall(TheModule->getFunction("get_scope_first_arg"), {scope_struct})});
      has_first_arg_copy = true;
    }
    
    
    first_arg = Builder->CreateCall(TheModule->getFunction("ConcatStr"),
                {Builder->CreateCall(TheModule->getFunction("get_scope_previous_scope"), {scope_struct}), _pre_dot_str});
                
    Builder->CreateCall(TheModule->getFunction("set_scope_first_arg"), {scope_struct, first_arg});

    changed_first_arg = true;
  }
  
  
  

  int target_args_size = Args.size();
  std::vector<Value *> ArgsV;


  if (in_str(Callee, threaded_tensor_functions))
  {
    msg = "\n\n\n\n\nCallExpr CALLEE " + Callee + " IS IN A THREAD" + "\n\n\n\n\n";
    p2t(msg);


    ArgsV.push_back(Builder->CreateCall(TheModule->getFunction("get_scope_thread_id"), {scope_struct}));

    target_args_size+=1;
  }
  



  msg = "CallExpr Call start mangle " + Callee;
  p2t(msg);

  bool is_self_of_nested_function = (nested_function==1 && isSelf);
  

  

  msg = "CallExpr Call name mangle";
  p2t(msg);

  // Handle self or object attribute expressions
  if(isSelf || isAttribute)
  {
    bool not_coding_language_method = (!in_str(tgt_function, native_methods));    

    

    if (not_coding_language_method)
      tgt_function = Class+tgt_function;

    if (!is_self_of_nested_function && not_coding_language_method)
    {

      _pre_dot_str = Builder->CreateCall(TheModule->getFunction("ConcatScopeAtCallExpr"),
                {Builder->CreateCall(TheModule->getFunction("get_scope_scope"), {scope_struct}), _pre_dot_str});

      first_arg = Builder->CreateCall(TheModule->getFunction("FirstArgOnDemand"),
                                                    {scope_struct,
                                                     _pre_dot_str,
                                                     Builder->CreateGlobalString(Class),
                                                     Builder->CreateGlobalString(Callee),
                                                     ConstantInt::get(Type::getInt32Ty(*TheContext), nested_function),
                                                     ConstantInt::get(Type::getInt32Ty(*TheContext), isSelf),
                                                     ConstantInt::get(Type::getInt32Ty(*TheContext), isAttribute)});

      Builder->CreateCall(TheModule->getFunction("set_scope_first_arg"), {scope_struct, first_arg});


      
    }
    if (is_self_of_nested_function && not_coding_language_method)
    { // object method inside object method
      first_arg_copy = Builder->CreateCall(TheModule->getFunction("CopyString"), {Builder->CreateCall(TheModule->getFunction("get_scope_first_arg"), {scope_struct})});

      first_arg = Builder->CreateCall(TheModule->getFunction("ConcatStr"),
                                                    {first_arg_copy,
                                                     _pre_dot_str});
      Builder->CreateCall(TheModule->getFunction("set_scope_first_arg"), {scope_struct, first_arg});
                                                      
      has_first_arg_copy = true;
    }
    changed_first_arg = not_coding_language_method;
    

    //name = NameSolver->codegen(scope_struct);
    
    if (CalleeOverride!="none"||in_str(Callee, native_methods))
    { // e.g: x.view()
    
      if (isSelf&&!isAttribute)
        ArgsV.push_back(Builder->CreateCall(TheModule->getFunction("get_scope_first_arg"), {scope_struct}));
      if (!isSelf&&isAttribute)
      {
        Value *arg = Builder->CreateCall(TheModule->getFunction("ConcatStr"),
                        {Builder->CreateCall(TheModule->getFunction("get_scope_previous_scope"), {scope_struct}), _pre_dot_str});
        ArgsV.push_back(arg);
        //must_free_arg0 = true; //TODO: break?
      }
      
      if (isSelf && isAttribute)
      { // e.g: self.can_load_.first_nonzero()
        // Extend first arg
        ArgsV.push_back(Builder->CreateCall(TheModule->getFunction("get_scope_first_arg"), {scope_struct}));
        ArgsV[0] = Builder->CreateCall(TheModule->getFunction("ConcatStr"),
                                        {ArgsV[0], _pre_dot_str});
        //must_free_arg0 = true; //TODO: break?
        
      }

      if (in_str(Callee, return_tensor_methods))
      {
        ArgsV[1] = Builder->CreateCall(TheModule->getFunction("tensor_Load"), {ArgsV[1], scope_struct});
        must_free_arg0 = false;
      }

    }
    else // Pass first_arg's reference for the derived AST nodes.
      ArgsV.push_back(Builder->CreateCall(TheModule->getFunction("get_scope_first_arg"), {scope_struct}));
    
    target_args_size+=1;
  }


  p2t("CallExpr Finish mangle, get scope info.");
  

  if (!(CalleeOverride!="none" || in_str(Callee, native_fn))||Callee=="print_scope") // user defined functions
  {
    has_scope = true;
    
    if(Callee!="print_scope")
    {
      scope_string = Builder->CreateCall(TheModule->getFunction("RandomStrOnDemand"), {});
      Builder->CreateCall(TheModule->getFunction("set_scope_scope"), {scope_struct, scope_string});

    }
    
    
    // std::cout << "Get scope" << ".\n";
    // ArgsV.push_back(Builder->CreateCall(TheModule->getFunction("get_scope_scope"), {scope_struct})); // Pass scope's reference for the derived AST nodes.
    // std::cout << "Get previous" << ".\n";
    // ArgsV.push_back(Builder->CreateCall(TheModule->getFunction("get_scope_previous_scope"), {scope_struct}));
    // std::cout << "Get thread id" << ".\n";
    // ArgsV.push_back(Builder->CreateCall(TheModule->getFunction("get_scope_thread_id"), {scope_struct}));
    // std::cout << "Get has grad" << ".\n";
    // ArgsV.push_back(Builder->CreateCall(TheModule->getFunction("get_scope_has_grad"), {scope_struct}));
    // std::cout << "Got all " << ".\n";    
    // target_args_size+=4;

    scope_struct = Builder->CreateCall(TheModule->getFunction("scope_struct_Copy"), {scope_struct}); 
    
    p2t("--------------------CallExpr Add scope struct");
    ArgsV.push_back(scope_struct);
    target_args_size+=1;
  }

  p2t("CallExpr require scope functions");

  if(in_str(tgt_function, require_scope_functions))
  {
    ArgsV.push_back(Builder->CreateCall(TheModule->getFunction("get_scope_scope"), {scope_struct})); // Pass scope's reference for the derived AST nodes.
    target_args_size+=1;
  }

  p2t("CallExpr got scope");

  
  

  // Detect function errors
  Function *CalleeF;
  if (!IsVarForward)
  {
    CalleeF = getFunction(tgt_function);
    if (!CalleeF)
    {
      std::string _error = "The referenced function "+ tgt_function +" was not yet declared.";
      return LogErrorV(_error);
    }

    tgt_function_name = CalleeF->getName().str();

    // If argument mismatch error.
    if ((CalleeF->arg_size()) != target_args_size && !in_str(tgt_function_name, vararg_methods))
    {
      //std::cout << "CalleeF->arg_size() " << CalleeF->arg_size() << " target_args_size " << target_args_size << "\n";
      std::string _error = "Incorrect parameters used on function " + tgt_function + " call.";
      return LogErrorV(_error);
    }
  }
  // std::cout << "\n\n\nCalling function: " << tgt_function <<"\n";
  msg = "CallExpr Calling function: " + tgt_function;
  p2t(msg);



  // Builder->CreateCall(TheModule->getFunction("get_scope_first_arg"), {scope_struct});

  // Get Arguments
  for (unsigned i = 0, e = Args.size(); i != e; ++i) {

    // msg = "CallExpr Call codegen for argument n°: " + i;
    // p2t(msg);

    //std::cout << "ARG: " << Args[i]->GetName() << " has self: " << Args[i]->GetSelf() << " and type: " << Args[i]->GetType() <<  "\n\n";

      

    // deal with firstarg on self.mcts(self.actions)

    
    first_arg = Builder->CreateCall(TheModule->getFunction("get_scope_first_arg"), {scope_struct});
    scope_string = Builder->CreateCall(TheModule->getFunction("get_scope_scope"), {scope_struct});
    previous_scope = Builder->CreateCall(TheModule->getFunction("get_scope_previous_scope"), {scope_struct});
    Value *_scope = (!in_str(tgt_function, native_methods)) ? previous_scope : scope_string;
    

    p2t("CallExpr Got scope. Now copy scope struct");
    
    

    Value *arg;
    Value *arg_scope = Builder->CreateCall(TheModule->getFunction("scope_struct_Copy"), {scope_struct});


    p2t("CallExpr Set scope");
    
    Builder->CreateCall(TheModule->getFunction("set_scope_scope"), {arg_scope, _scope});

    if ((Args[i]->GetType()=="tensor" || Args[i]->GetType()=="pinned_tensor") && Args[i]->GetIsVarLoad())
    {      
      VariableExprAST *Arg = static_cast<VariableExprAST *>(Args[i].get());
      std::cout << "Codegen tensor name solver" << ".\n";
      arg = Arg->NameSolver->codegen(arg_scope);
      std::cout << "name solver done" << ".\n";

      arg = Builder->CreateCall(TheModule->getFunction("tensor_Load"), {arg, scope_struct});
    }
    else
    {
      Value *fa = (isAttribute && !isSelf && !in_str(tgt_function, native_methods) && nested_function) ? first_arg_copy : first_arg;

      Builder->CreateCall(TheModule->getFunction("set_scope_first_arg"), {arg_scope, fa});

      p2t("CallExpr Non-tensor arg codegen");
      arg = Args[i]->codegen(arg_scope);
    }

    p2t("CallExpr push back arg");

  
    ArgsV.push_back(arg);


    if (!ArgsV.back())
      return nullptr;
  }



  

  Value *ret = ConstantFP::get(*TheContext, APFloat(0.0f));
  // std::cout << "\n\nCallExpr Create call: "  << tgt_function_name << " from parent: " << functionName << ", with override: " << CalleeOverride << " and " << ArgsV.size() << " args." << "\n\n";


  p2t("CallExpr Execute call");



  if (CalleeOverride=="none")
    ret = Builder->CreateCall(CalleeF, ArgsV, "calltmp");
  else
  {
    if(in_str(CalleeOverride, threaded_tensor_functions))
      ArgsV.push_back(Builder->CreateCall(TheModule->getFunction("get_scope_thread_id"), {scope_struct}));
    
    //std::cout << "Override: " << CalleeOverride << "\n";
    if (in_str(CalleeOverride, native_modules))
    {
      CalleeF = getFunction(CalleeOverride);
      Value *conv_name = Builder->CreateGlobalString(tgt_function);
      Value *is_attr = ConstantInt::get(Type::getInt32Ty(*TheContext), (int)(isSelf));
      ArgsV.push_back(conv_name);
      ArgsV.push_back(is_attr);
      
      if (CalleeF->arg_size() != ArgsV.size())
      {
        std::string _error = "Incorrect parameters used on function " + tgt_function + " call.";
        return LogErrorV(_error);
      }
      ret = Builder->CreateCall(CalleeF, ArgsV, "calltmp");

    }
    else if (CalleeOverride=="SplitString")
    {
      Value *V = Builder->CreateCall(TheModule->getFunction("str_Load"), 
                                     {Builder->CreateGlobalString(PreDot), scope_struct});
      
      ret = Builder->CreateCall(getFunction("SplitString"), 
                          {V, ArgsV[1]});

    }
    else if (CalleeOverride=="ToFloat")
    {
      //std::cout << "\n\nTO FLOAT HAS TYPE " << Args[0]->GetType() << "\n";
      if (Args[0]->GetType()=="str")
        ret = Builder->CreateCall(getFunction("StrToFloat"), 
                          {ArgsV[0]});

    } else
      ret = Builder->CreateCall(getFunction(CalleeOverride), ArgsV, "calltmp");
  }

  p2t("CallExpr clean scope"); 
  
  // Builder->CreateCall(TheModule->getFunction("FreeChar"), {previous_scope});
  
  // if (changed_first_arg)
  //   Builder->CreateCall(TheModule->getFunction("FreeChar"), {first_arg});
  
  // if (has_first_arg_copy)
  //   Builder->CreateCall(TheModule->getFunction("FreeChar"), {first_arg_copy});

  // if (has_scope)
  //   Builder->CreateCall(TheModule->getFunction("FreeChar"), {scope_string});

  // if (must_free_arg0)
  //   Builder->CreateCall(TheModule->getFunction("FreeChar"), {ArgsV[0]});
  
  return ret;
}