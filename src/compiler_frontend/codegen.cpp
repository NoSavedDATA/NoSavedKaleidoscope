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


Value *StrVecExprAST::codegen(Value *first_arg, Value *scope_str, Value *previous_scope, Value *thread_id, Value *has_grad) {
  if (not ShallCodegen)
    return ConstantFP::get(*TheContext, APFloat(0.0f));

  //std::vector<AllocaInst *> OldBindings;

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


    AllocaInst *Alloca = CreateEntryBlockAlloca(TheFunction, VarName);
    Builder->CreateStore(InitVal, Alloca);
      
    // Remember the old variable binding so that we can restore the binding when
    // we unrecurse.
    //std::cout << "STRING CODEGEN FOR " << VarName << "\n";
    //OldBindings.push_back(NamedStrVecs[VarName]);

    
    // Remember this binding.

    if (Type=="str")
      NamedStrVecs[VarName] = Alloca;
    if (Type=="float")
      NamedFloatVecs[VarName] = Alloca;
    
    
  }


  //for (unsigned i = 0, e = VarNames.size(); i != e; ++i)
  //  NamedStrVecs[VarNames[i].first] = OldBindings[i];

  
  return ConstantFP::get(*TheContext, APFloat(0.0f));
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