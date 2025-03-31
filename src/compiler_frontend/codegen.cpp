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



Value *NumberExprAST::codegen(Value *first_arg, Value *scope_str, Value *previous_scope, Value *thread_id, Value *has_grad) {
  if (not ShallCodegen)
    return ConstantFP::get(*TheContext, APFloat(0.0f));
  
  return ConstantFP::get(*TheContext, APFloat(Val));
}

Value *StringExprAST::codegen(Value *first_arg, Value *scope_str, Value *previous_scope, Value *thread_id, Value *has_grad) {
  if (not ShallCodegen)
    return ConstantFP::get(*TheContext, APFloat(0.0f));
  SetName(Val);
  return Builder->CreateGlobalString(Val);
}


// Create Float Var
Value *VarExprAST::codegen(Value *first_arg, Value *scope_str, Value *previous_scope, Value *thread_id, Value *has_grad) {
  if (not ShallCodegen)
    return ConstantFP::get(*TheContext, APFloat(0.0f));
  std::vector<Value *> OldBindings;

  Function *TheFunction = Builder->GetInsertBlock()->getParent();


  // Register all variables and emit their initializer.
  for (unsigned i = 0, e = VarNames.size(); i != e; ++i) {
    const std::string &VarName = VarNames[i].first;
    ExprAST *Init = VarNames[i].second.get();

    // Emit the initializer before adding the variable to scope, this prevents
    // the initializer from referencing the variable itself, and permits stuff
    // like this:
    //  var a = 1 in
    Value *InitVal;
    if (Init) {
      InitVal = Init->codegen(first_arg, scope_str, previous_scope, thread_id, has_grad);
      if (!InitVal)
        return nullptr;
    } else { // If not specified, use 0.0.
      InitVal = ConstantFP::get(*TheContext, APFloat(0.0));
    }



    Value *var_name = Builder->CreateGlobalString(VarName);
    var_name = Builder->CreateCall(TheModule->getFunction("ConcatStr"),
                                        {scope_str, var_name});


    Builder->CreateCall(TheModule->getFunction("AddFloatToScopeCleanList"),
                                        {scope_str, var_name});

    Builder->CreateCall(TheModule->getFunction("StoreOnDemand"),
                                        {var_name, InitVal});
                                                  
    
  }


  return ConstantFP::get(*TheContext, APFloat(0.0));
}











Value *DataExprAST::codegen(Value *first_arg, Value *scope_str, Value *previous_scope, Value *thread_id, Value *has_grad) {
  if (not ShallCodegen)
    return ConstantFP::get(*TheContext, APFloat(0.0f));

  std::cout << "EXECUTING DATA CODEGEN.\n";

  Function *TheFunction = Builder->GetInsertBlock()->getParent();

  // Register all variables and emit their initializer.
  for (unsigned i = 0, e = VarNames.size(); i != e; ++i) {
    const std::string &VarName = VarNames[i].first; 
    ExprAST *Init = VarNames[i].second.get();
    
    Value *var_name, *scopeless_name, *init;
    
    var_name = Builder->CreateCall(TheModule->getFunction("CopyString"),
                                            {Builder->CreateGlobalString(VarName)});

    bool is_self = GetSelf();
    bool is_attr = GetIsAttribute();

    if (is_self||is_attr)
      var_name = Builder->CreateCall(TheModule->getFunction("ConcatStrFreeRight"),
                                            {first_arg, var_name});
    scopeless_name = Builder->CreateCall(TheModule->getFunction("CopyString"),
                                            {var_name});
    if (!(is_self||is_attr))
      var_name = Builder->CreateCall(TheModule->getFunction("ConcatStrFreeRight"),
                                            {scope_str, var_name});





    Value *notes_vector = Builder->CreateCall(TheModule->getFunction("CreateNotesVector"),
                                            {});


    for (int j=0; j<Notes.size(); j++)
    {
      ExprAST *note = Notes[j].get();
      if (NumberExprAST* numExpr = dynamic_cast<NumberExprAST*>(note)) {
        
        notes_vector = Builder->CreateCall(TheModule->getFunction("Add_Float_To_NotesVector"),
                                                {notes_vector, note->codegen(first_arg, scope_str, previous_scope, thread_id, has_grad)});
                                                // {notes_vector});
      }
      else if (StringExprAST* expr = dynamic_cast<StringExprAST*>(note)) {
        Value *str_val = Builder->CreateCall(TheModule->getFunction("CopyString"),
                                            {note->codegen(first_arg, scope_str, previous_scope, thread_id, has_grad)});
        notes_vector = Builder->CreateCall(TheModule->getFunction("Add_String_To_NotesVector"),
                                                {notes_vector, str_val});
      }
      else if (VariableExprAST* expr = dynamic_cast<VariableExprAST*>(note)) {
        notes_vector = Builder->CreateCall(TheModule->getFunction("Add_Float_To_NotesVector"),
                                                {notes_vector, note->codegen(first_arg, scope_str, previous_scope, thread_id, has_grad)});
      }
      else {
        std::cout << "Could not find the data type\n";
      }

    }

    std::cout << "TYPE: " << Type << ".\n";

    std::string create_fn = Type + "_Create";
    std::cout << "CREATE FN: " <<  create_fn << ".\n";


    Builder->CreateCall(TheModule->getFunction(create_fn),
                                              {var_name, scopeless_name, Init->codegen(first_arg, scope_str, previous_scope, thread_id, has_grad), notes_vector,
                                               thread_id, scope_str});
    
    Builder->CreateCall(TheModule->getFunction("Dispose_NotesVector"), {notes_vector});

    
  }


  return ConstantFP::get(*TheContext, APFloat(0.0));
}






Value *IfExprAST::codegen(Value *first_arg, Value *scope_str, Value *previous_scope, Value *thread_id, Value *has_grad) {
  if (not ShallCodegen)
    return ConstantFP::get(*TheContext, APFloat(0.0f));


  Value *CondV = Cond->codegen(first_arg, scope_str, previous_scope, thread_id, has_grad);
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
    ThenV = then_body->codegen(first_arg, scope_str, previous_scope, thread_id, has_grad);
  

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
    ElseV = else_body->codegen(first_arg, scope_str, previous_scope, thread_id, has_grad);

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

Value *ForExprAST::codegen(Value *first_arg, Value *scope_str, Value *previous_scope, Value *thread_id, Value *has_grad) {
  if (not ShallCodegen)
    return ConstantFP::get(*TheContext, APFloat(0.0f));
  Function *TheFunction = Builder->GetInsertBlock()->getParent();

  // Create an alloca for the variable in the entry block.
  AllocaInst *Alloca = CreateEntryBlockAlloca(TheFunction, VarName);

  // Emit the start code first, without 'variable' in scope.
  Value *StartVal = Start->codegen(first_arg, scope_str, previous_scope, thread_id, has_grad);
  if (!StartVal)
    return nullptr;

  Value *_zero = ConstantFP::get(*TheContext, APFloat(0.0));



  Value *var_name = Builder->CreateGlobalString(VarName);
  var_name = Builder->CreateCall(TheModule->getFunction("ConcatStr"),
                                    {scope_str, var_name});

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
    StepVal = Step->codegen(first_arg, scope_str, previous_scope, thread_id, has_grad);
    if (!StepVal)
      return nullptr;
  } 


  // Compute the end condition.
  Value *EndCond = End->codegen(first_arg, scope_str, previous_scope, thread_id, has_grad);
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
    body->codegen(first_arg, scope_str, previous_scope, thread_id, has_grad);

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



Value *WhileExprAST::codegen(Value *first_arg, Value *scope_str, Value *previous_scope, Value *thread_id, Value *has_grad) {
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
  Value* condVal = Cond->codegen(first_arg, scope_str, previous_scope, thread_id, has_grad);
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
    body->codegen(first_arg, scope_str, previous_scope, thread_id, has_grad);

  // After the loop body, go back to the condition check
  Builder->CreateBr(CondBB);

  // Insert the after loop block
  Builder->SetInsertPoint(AfterBB);

  return Constant::getNullValue(Type::getFloatTy(*TheContext));
}




bool seen_var_attr = false;
Value *VariableExprAST::codegen(Value *first_arg, Value *scope_str, Value *previous_scope, Value *thread_id, Value *has_grad) {
  if (not ShallCodegen)
    return ConstantFP::get(*TheContext, APFloat(0.0f));
  // Look this variable up in the function.



  Function *TheFunction = Builder->GetInsertBlock()->getParent();
  //std::string functionName = TheFunction->getName().str();
  

  //std::cout << "Create value V" << "\n";
  Value * ret = ConstantFP::get(*TheContext, APFloat(0.0f));
  Value *V, *var_name;



  /*
  std::cout << "\nVARIABLE EXPR CODEGEN: " << Name << "\n";
  for (const auto &entry : NamedValues)
    std::cout << "NamedValues: " << entry.first << "\n";
  for (const auto &entry : NamedClassValues)
    std::cout << "NamedClassValues: " << entry.first << "\n";
  */

  std::string type = GetType();
  std::string pre_dot = GetPreDot();
  bool is_self = GetSelf();
  bool is_attr = GetIsAttribute();
  
  //std::string __print = "\n\nLOAD OF " + std::string(Name) + " ";
  //Builder->CreateCall(TheModule->getFunction("print"),
  //    {Builder->CreateGlobalString(__print), ConstantFP::get(*TheContext, APFloat(0.0f))});

  

  var_name = NameSolver->codegen(first_arg, scope_str, previous_scope, thread_id, has_grad);

  NameSolverAST *name_solver = static_cast<NameSolverAST *>(NameSolver.get());
  std::string Name = std::get<0>(name_solver->Names[name_solver->Names.size()-1]);
  

  if (is_self||is_attr)
  {
    if (type=="float")
    {
        V = Builder->CreateCall(TheModule->getFunction("LoadOnDemand"), {var_name});
        
        return V;
      
    }
    if (type=="str_vec")
    {
      V = Builder->CreateCall(TheModule->getFunction("LoadStrVecOnDemand"),
                                                      {var_name});
      //Builder->CreateCall(TheModule->getFunction("PrintStrVec"), {V});
      return V;
      
    }
    if (type=="float_vec"){
      V = Builder->CreateCall(TheModule->getFunction("LoadFloatVecOnDemand"),
                                                      {var_name});
      Builder->CreateCall(TheModule->getFunction("PrintFloatVec"), {V});
      return V;
    }
  }



  if (type=="float")
  {
    
    V = Builder->CreateCall(TheModule->getFunction("LoadOnDemand"), {var_name});
    

    //if (!seen_var_attr) //TODO: Solve this bug
    //  V = Builder->CreateCall(TheModule->getFunction("UnbugFloat"), {V}, "unbugfloat");

    return V;

  } else if (type=="object") {
    return var_name;

  } else if (type=="str") {

    
    //std::cout << "Type: " << Type << "\n\n";

    for (const auto &entry : NamedTensorsT)
    {
      std::cout << "Returning None because a tensor with name " << Name << " was found on strings map " << "\n";
      if (ends_with(entry.first, Name))
        return ret;
    }
    

    V = Builder->CreateCall(TheModule->getFunction("LoadStrOnDemand"), {var_name});

    //if (!seen_var_attr)
    //  Builder->CreateCall(TheModule->getFunction("PrintStr"), {V});
    
    return V;
  } else if (type=="str_vec") {





    return nullptr;
  } else if (NamedPinnedTensors.count(Name)>0) {
    //std::cout << "\nVariable Tensor " << Name << " Codegen.\n";
  

    if (!seen_var_attr)
      Builder->CreateCall(TheModule->getFunction("PrintTensor"), {thread_id, var_name});
    
    
    //Builder->CreateCall(TheModule->getFunction("PrintTensor"), {thread_id, var_name});

    return Builder->CreateCall(TheModule->getFunction("LoadTensor"), {var_name});
  } else if (type=="tensor") {
    //std::cout << "\nVariable Tensor " << Name << " Codegen.\n";


    if (!seen_var_attr)
    {
      Builder->CreateCall(TheModule->getFunction("PrintTensor"), {thread_id, var_name});
      return ConstantFP::get(*TheContext, APFloat(0.0f));
    }
    
    return Builder->CreateCall(TheModule->getFunction("LoadTensor"), {var_name});
  }
  else
  {
    std::string _error = "Variable " + Name + " does not exist";
    LogErrorS(_error);
  }
}



Value *VecIdxExprAST::codegen(Value *first_arg, Value *scope_str, Value *previous_scope, Value *thread_id, Value *has_grad) {
  if (not ShallCodegen)
    return ConstantFP::get(*TheContext, APFloat(0.0f));
  // Look this variable up in the function.

  std::cout << "Now Loading Vec indexation for type: " << Type << "  \n";


  Function *TheFunction = Builder->GetInsertBlock()->getParent();
  std::string functionName = TheFunction->getName().str();
  
  
  Value * ret = ConstantFP::get(*TheContext, APFloat(0.0f));
  Value *V, *idx;

  if (Type!="object_vec")
    idx = Idx[0]->codegen(first_arg, scope_str, previous_scope, thread_id, has_grad);


  Value *var_name, *object_name, *object_var_name;
  var_name = NameSolver->codegen(first_arg, scope_str, previous_scope, thread_id, has_grad);
  
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
        idx_calc_args.push_back(Idx[i]->codegen(first_arg, scope_str, previous_scope, thread_id, has_grad));
      Value *idx_at = Builder->CreateCall(TheModule->getFunction("CalculateIdxOffset"),
                          idx_calc_args);

      return Builder->CreateCall(TheModule->getFunction("IdxTensor"), {var_name, idx_at, scope_str, thread_id});
      */
      std::vector<Value *> idx_calc_args;
      idx_calc_args.push_back(var_name);
      idx_calc_args.push_back(scope_str);
      idx_calc_args.push_back(thread_id);
      for (int i=0; i<Idx.size(); i++)
        idx_calc_args.push_back(Idx[i]->codegen(first_arg, scope_str, previous_scope, thread_id, has_grad));

      return Builder->CreateCall(TheModule->getFunction("IdxTensor"), idx_calc_args);
    } else {
      VariableExprAST *idx = static_cast<VariableExprAST *>(Idx[0].get());
      Value *idx_tensor_name = idx->NameSolver->codegen(first_arg, scope_str, previous_scope, thread_id, has_grad);
      
      return Builder->CreateCall(TheModule->getFunction("IdxTensorWithTensor"), {var_name, idx_tensor_name, thread_id});
      
    }
    
  }

  std::string _error = "Unknown vector: " + Name + ".";
  LogErrorS(_error);
  std::cout << "Type " << Type << "\n";

  return ret;
}


Value *ObjectVecIdxExprAST::codegen(Value *first_arg, Value *scope_str, Value *previous_scope, Value *thread_id, Value *has_grad) {
  if (not ShallCodegen)
    return ConstantFP::get(*TheContext, APFloat(0.0f));
  // Look this variable up in the function.
  std::cout << "ObjectVecIdxExprAST codegen" << "\n";
  
  VecIdxExprAST *vec = static_cast<VecIdxExprAST *>(Vec.get());
  std::cout << "vec name " << vec->GetName() << "\n";
  std::cout << "ObjectVecIdxExprAST is vec: " << GetIsVec() << "\n";

  Value *idx = vec->Idx[0]->codegen(first_arg, scope_str, previous_scope, thread_id, has_grad);


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
                                                      {Builder->CreateLoad(int8PtrTy, first_arg), var_name});
  }

  if (Type=="tensor")
    return Builder->CreateCall(TheModule->getFunction("object_vec_idxTensor"),
                                                      {var_name, idx, post_dot_str});
  if (Type=="object")
    return Builder->CreateCall(TheModule->getFunction("object_vec_idxObject"),
                                                      {var_name, idx, post_dot_str});


  return ConstantFP::get(*TheContext, APFloat(0.0f));
}



Value *BinaryTensorScalarExprAST::codegen(Value *first_arg, Value *scope_str, Value *previous_scope, Value *thread_id, Value *has_grad) {
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
                                                      {first_arg, tensor_name});
  if (!(is_self||is_attr))
    tensor_name = Builder->CreateCall(TheModule->getFunction("ConcatStr"),
                                            {scope_str, tensor_name});
    



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
    
    Value *Val = RHS->codegen(first_arg, scope_str, previous_scope, thread_id, has_grad);
    if (!Val)
      return nullptr;

    
    
    std::cout << "1 0 attr\n";
    


    //LogErrorS("Attribution from float into tensor is not possible.");    
    
    
      
    
    seen_var_attr=false;
    return Val;
  }


  std::cout << "\n\n\nTensor scalar for LHS: " << LHS->GetName() << " RHS: " << RHS->GetName() << "\n\n\n";
  Value *LtensorPtr = LHS->codegen(first_arg, scope_str, previous_scope, thread_id, has_grad);
  Value *R = RHS->codegen(first_arg, scope_str, previous_scope, thread_id, has_grad);
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
                               {LtensorPtr, R, thread_id}, "cudascalarmult");
  case '/':
    return Builder->CreateCall(TheModule->getFunction("CudaScalarDiv"),
                               {LtensorPtr, R, thread_id}, "cudascalardiv");
  case 77:
    return Builder->CreateCall(TheModule->getFunction("CudaReverseScalarDiv"),
                               {LtensorPtr, R, thread_id}, "cudareversescalardiv");
  case '+':
    return Builder->CreateCall(TheModule->getFunction("CudaScalarAdd"),
                               {LtensorPtr, R, thread_id}, "cudascalaradd");
  case '-':
    return Builder->CreateCall(TheModule->getFunction("CudaScalarSub"),
                               {LtensorPtr, R, thread_id}, "cudascalarsub");
  case tok_equal:
    return Builder->CreateCall(TheModule->getFunction("CudaScalarEqual"),
                               {LtensorPtr, R, thread_id}, "cudascalarequal");
  case tok_diff:
    return Builder->CreateCall(TheModule->getFunction("CudaScalarDiff"),
                               {LtensorPtr, R, thread_id}, "cudascalardiff");
  case '<':
    return Builder->CreateCall(TheModule->getFunction("CudaScalarMinor"),
                               {LtensorPtr, R, thread_id}, "cudascalarminor");
  case '>':
    return Builder->CreateCall(TheModule->getFunction("CudaScalarHigher"),
                               {LtensorPtr, R, thread_id}, "cudascalarhigher");
  case tok_minor_eq:
    return Builder->CreateCall(TheModule->getFunction("CudaScalarMinorEq"),
                               {LtensorPtr, R, thread_id}, "cudascalarminoreq");
  case tok_higher_eq:
    return Builder->CreateCall(TheModule->getFunction("CudaScalarHigherEq"),
                               {LtensorPtr, R, thread_id}, "cudascalarhighereq");
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



Value *BinaryPinnedScalarExprAST::codegen(Value *first_arg, Value *scope_str, Value *previous_scope, Value *thread_id, Value *has_grad) {
  if (not ShallCodegen)
    return ConstantFP::get(*TheContext, APFloat(0.0f));

  Value *tensor_name;



  

  if (Op == '=') {
    seen_var_attr=true;

    
    Value *Val = RHS->codegen(first_arg, scope_str, previous_scope, thread_id, has_grad);
    if (!Val)
      return nullptr;
    
    std::cout << "2 0 attr\n";
    std::cout << "is vec: " << LHS->GetIsVec()  << "\n";


    


    VecIdxExprAST   *LHSE = static_cast<VecIdxExprAST *>(LHS.get());
    tensor_name = LHSE->NameSolver->codegen(first_arg, scope_str, previous_scope, thread_id, has_grad);

    if (!LHSE)
      return LogErrorV("'=' destiny must be a variable.");

    

    std::vector<Value *> idx_calc_args;

    idx_calc_args.push_back(tensor_name);

    for (int i=0; i<LHSE->Idx.size(); i++)
    {
      idx_calc_args.push_back(LHSE->Idx[i]->codegen(first_arg, scope_str, previous_scope, thread_id, has_grad));
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
                          {tensor_name, LHSE->Idx->codegen(first_arg, scope_str, previous_scope, thread_id, has_grad), Val});
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


  
  Value *LtensorPtr = LHS->codegen(first_arg, scope_str, previous_scope, thread_id, has_grad);
  Value *R = RHS->codegen(first_arg, scope_str, previous_scope, thread_id, has_grad);
  



  
  
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
                               {LtensorPtr, R, thread_id}, "cudascalarmult");
  case '/':
    return Builder->CreateCall(TheModule->getFunction("CudaScalarDiv"),
                               {LtensorPtr, R, thread_id}, "cudascalardiv");
  case 77:
    return Builder->CreateCall(TheModule->getFunction("CudaReverseScalarDiv"),
                               {LtensorPtr, R, thread_id}, "cudareversescalardiv");
  case '+':
    return Builder->CreateCall(TheModule->getFunction("CudaScalarAdd"),
                               {LtensorPtr, R, thread_id}, "cudascalaradd");
  case '-':
    return Builder->CreateCall(TheModule->getFunction("CudaScalarSub"),
                               {LtensorPtr, R, thread_id}, "cudascalarsub");
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







Value *BinaryPinnedAndTensorExprAST::codegen(Value *first_arg, Value *scope_str, Value *previous_scope, Value *thread_id, Value *has_grad) {
  if (not ShallCodegen)
    return ConstantFP::get(*TheContext, APFloat(0.0f));

  Value *tensor_name;



  

  if (Op == '=') {
    seen_var_attr=true;

    
    Value *RtensorPtr = RHS->codegen(first_arg, scope_str, previous_scope, thread_id, has_grad);
    if (!RtensorPtr)
      return nullptr;
    
    std::cout << "2 0 attr\n";
    std::cout << "is vec: " << LHS->GetIsVec()  << "\n";


    


    VecIdxExprAST   *LHSE = static_cast<VecIdxExprAST *>(LHS.get());
    tensor_name = LHSE->NameSolver->codegen(first_arg, scope_str, previous_scope, thread_id, has_grad);

    if (!LHSE)
      return LogErrorV("'=' destiny must be a variable.");

    

    std::vector<Value *> idx_calc_args;

    idx_calc_args.push_back(tensor_name);
    idx_calc_args.push_back(RtensorPtr);
    idx_calc_args.push_back(thread_id);

    for (int i=0; i<LHSE->Idx.size(); i++)
      idx_calc_args.push_back(LHSE->Idx[i]->codegen(first_arg, scope_str, previous_scope, thread_id, has_grad));


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
                          {tensor_name, LHSE->Idx->codegen(first_arg, scope_str, previous_scope, thread_id, has_grad), Val});
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


  
  Value *LtensorPtr = LHS->codegen(first_arg, scope_str, previous_scope, thread_id, has_grad);
  Value *R = RHS->codegen(first_arg, scope_str, previous_scope, thread_id, has_grad);
  



  
  
  if (!LtensorPtr || !R)
    return nullptr;




  switch (Op)
  {
  case '*':
    return Builder->CreateCall(TheModule->getFunction("CudaScalarMult"),
                               {LtensorPtr, R, thread_id}, "cudascalarmult");
  case '/':
    return Builder->CreateCall(TheModule->getFunction("CudaScalarDiv"),
                               {LtensorPtr, R, thread_id}, "cudascalardiv");
  case 77:
    return Builder->CreateCall(TheModule->getFunction("CudaReverseScalarDiv"),
                               {LtensorPtr, R, thread_id}, "cudareversescalardiv");
  case '+':
    return Builder->CreateCall(TheModule->getFunction("CudaScalarAdd"),
                               {LtensorPtr, R, thread_id}, "cudascalaradd");
  case '-':
    return Builder->CreateCall(TheModule->getFunction("CudaScalarSub"),
                               {LtensorPtr, R, thread_id}, "cudascalarsub");
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