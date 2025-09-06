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
    else if(type=="bool")
      load_type = Type::getInt1Ty(*TheContext);
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



  return ConstantFP::get(*TheContext, APFloat(0.0));
}







llvm::Type *get_type_from_str(std::string type)
{
  llvm::Type *llvm_type;
  if (type=="float")
    llvm_type = Type::getFloatTy(*TheContext);
  else if (type=="int")
    llvm_type = Type::getInt32Ty(*TheContext);
  else if (type=="bool")
    llvm_type = Type::getInt1Ty(*TheContext);
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




Value *UnkVarExprAST::codegen(Value *scope_struct) {
  if (not ShallCodegen)
    return ConstantFP::get(*TheContext, APFloat(0.0f));


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
      Value *void_ptr = Constant::getNullValue(Type::getInt8Ty(*TheContext)->getPointerTo());
      initial_value = callret(Type+"_channel_message", {scope_struct, void_ptr, initial_value});
    }




    if(in_str(Type, primary_data_tokens)&&!(is_self||is_attr))
    { 
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
 










    if(Type!="float"&&Type!="int"&&ClassSize.count(Type)==0)
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
      

    call("str_Delete", {var_name});
    call("str_Delete", {scopeless_name});
  }


  return ConstantFP::get(*TheContext, APFloat(0.0));
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
    
    llvm::Type *alloca_type = get_type_from_str(Type);
    AllocaInst *alloca = CreateEntryBlockAlloca(TheFunction, Name, alloca_type);
    Builder->CreateStore(initial_value, alloca);

    function_allocas[parser_struct.function_name][VarName] = alloca;
  }

  return ConstantFP::get(*TheContext, APFloat(0.0));
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



    Check_Is_Compatible_Data_Type(data_type, Init->GetDataTree(), parser_struct);    



    if(Init->GetIsMsg()) {
      Value *void_ptr = Constant::getNullValue(Type::getInt8Ty(*TheContext)->getPointerTo());
      initial_value = callret(Type+"_channel_message", {scope_struct, void_ptr, initial_value});
    }


    if(in_str(Type, primary_data_tokens)&&!(is_self||is_attr))
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




    Value *notes_vector;
    if (HasNotes) {
      notes_vector = callret("CreateNotesVector", {});
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
        else if (Nameable* expr = dynamic_cast<Nameable*>(note)) {
          
          
          if(expr->Depth==1&&data_typeVars[parser_struct.function_name].count(expr->Name)==0)
          { 
            Value *_str = callret("CopyString", {global_str(expr->Name)});
            call("MarkToSweep_Mark", {scope_struct, _str, global_str("str")});

            Value *str_val = callret("CopyString", {_str});
            notes_vector = callret("Add_To_NotesVector_str", {notes_vector, str_val});
          } else {
            std::string type = expr->GetDataTree().Type;
            notes_vector = callret("Add_To_NotesVector_"+type, {notes_vector, note->codegen(scope_struct)});
          }
          
        }
        else {
          std::cout << "Could not find the data type of a note in DataExpr of " << VarName << " \n";
        }
      }
    } else
      notes_vector = callret("nullptr_get", {});
  

    
    if(!IsStruct||Type=="list")
    {
      std::string create_fn = Type;
      create_fn = (create_fn=="tuple") ? "list" : create_fn;
      create_fn = create_fn + "_Create";

      initial_value = callret(create_fn, {scope_struct, var_name, scopeless_name, initial_value, notes_vector});
    }



    call("MarkToSweep_Mark", {scope_struct, initial_value, global_str(Extract_List_Suffix(Type))});
    if(is_self||is_attr)
      call("MarkToSweep_Unmark_Scopeless", {scope_struct, initial_value});
    else
      call("MarkToSweep_Unmark_Scopeful", {scope_struct, initial_value});


      

    if(is_self)
    {
      int object_ptr_offset = ClassVariables[parser_struct.class_name][VarName]; 
  
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
      

    if(HasNotes)
      call("Dispose_NotesVector", {notes_vector, scopeless_name});
    else
      call("str_Delete", {scopeless_name});
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
  
  if(Cond->GetDataTree().Type=="int")
    CondV = Builder->CreateICmpNE(
        CondV, const_int(0), "ifcond");
  if(Cond->GetDataTree().Type=="float")
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

  std::string start_type = Start->GetDataTree().Type;
  llvm::Type *llvm_type = get_type_from_str(start_type);
  

  // Create an alloca for the variable in the entry block.
  AllocaInst *control_var_alloca = CreateEntryBlockAlloca(TheFunction, VarName, llvm_type);

  function_allocas[parser_struct.function_name][VarName] = control_var_alloca;

  // Emit the start code first, without 'variable' in scope.
  Value *StartVal = Start->codegen(scope_struct);


  if (!StartVal)
    return nullptr;






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



  // Compute the end condition.
  Value *EndCond = End->codegen(scope_struct);
  if (!EndCond)
    return nullptr;

  // Convert condition to a bool by comparing equal to 0.0.








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
  if (start_type=="int")
    NextVal = Builder->CreateAdd(CurVal, StepVal, "nextvar"); // Increment  
  if (start_type=="float")
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


  std::string vec_type = UnmangleVec(data_type);
  Value *vec_value = callret(vec_type+"_Idx", {scope_struct, vec, CurIdx});
  if((vec_type=="list"||vec_type=="tuple")&&(Type=="float"||Type=="int"))
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







std::string VariableExprAST::GetType(bool from_assignment) {
  std::string type = Type;

  if (type=="None") {
    if (typeVars[parser_struct.function_name].find(Name) != typeVars[parser_struct.function_name].end())
      type = typeVars[parser_struct.function_name][Name];
    SetType(type);
    // LogBlue("variable " + parser_struct.function_name + "/" + Name +  " has type " + type);
  } else {
    if(CanBeString)
      SetType("str");
  }
  return type;
}

Data_Tree VariableExprAST::GetDataTree(bool from_assignment) {
  // if(!data_type.empty)
  //   return data_type;


  Data_Tree data_type;
  if (data_typeVars[parser_struct.function_name].find(Name) != data_typeVars[parser_struct.function_name].end())
    data_type = data_typeVars[parser_struct.function_name][Name];
  else
    LogError(-1, "Could not find variable " + Name + " on scope " + parser_struct.function_name);




  
  return data_type;
}


bool seen_var_attr = false;
Value *VariableExprAST::codegen(Value *scope_struct) {
  if (not ShallCodegen)
    return ConstantFP::get(*TheContext, APFloat(0.0f));
  // Look this variable up in the function.


  Function *TheFunction = Builder->GetInsertBlock()->getParent();
  

  Value *V;


  std::string type = GetType();
  
  if (type=="None") {
    if(!CanBeString)
    {
      LogError(parser_struct.line, "Variable " + Name + " was not found on scope " + parser_struct.function_name + ".");
      return ConstantFP::get(*TheContext, APFloat(0.0f));
    }
    Value *_str = callret("CopyString", {global_str(Name)});
    call("MarkToSweep_Mark", {scope_struct, _str, global_str("str")});
    return _str;
  }



  
  std::string pre_dot = GetPreDot();
  bool is_self = GetSelf();
  bool is_attr = GetIsAttribute();
    





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

    // LogBlue("operation: " + Operation + " with elements: " + Elements + " and return type " + type);
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


    // Codegen the RHS.
    Value *Val;
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
      
      Val = callret(Operation, {scope_struct, LHS->codegen(scope_struct), RHS->codegen(scope_struct)});
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
        std::string LType = LHSE->GetDataTree().Type;


        
        std::string store_trigger = LType + "_StoreTrigger";
        Value *Val_indexed = callret("assign_wise_list_Idx", {Val, const_int(i)});

        AllocaInst *alloca = function_allocas[parser_struct.function_name][Lname];
        


        Function *F = TheModule->getFunction(store_trigger);
        if (F)
        {
          Value *old_val = Builder->CreateLoad(int8PtrTy, alloca);
          Value *Lvar_name = global_str(LHSE->Name);
          call(store_trigger, {Lvar_name, old_val, Val_indexed, scope_struct});
        }

        if (LType=="int")
          Val_indexed = callret("to_int", {scope_struct, Val_indexed});
        if (LType=="float")
          Val_indexed = callret("to_float", {scope_struct, Val_indexed});
        //   Val_indexed = Builder->CreatePtrToInt(Val_indexed, Type::getInt32Ty(*TheContext));

        Builder->CreateStore(Val_indexed, alloca);

        
        if(!in_str(LType, primary_data_tokens))
          call("MarkToSweep_Mark", {scope_struct, Val_indexed, global_str(Extract_List_Suffix(LType))});
        
        
      }
      return ConstantFP::get(*TheContext, APFloat(0.0f));
    }




    std::string LType = LHS->GetDataTree().Type;
    std::string store_op = LType + "_Store";



    


    bool is_alloca = (!LHS->GetSelf()&&!LHS->GetIsAttribute());
    
    Value *Lvar_name;
      
    

    if(auto *LHSV = dynamic_cast<NameableIdx *>(LHS.get())) {
      
      Value *vec = LHSV->Inner->codegen(scope_struct);
      Data_Tree dt = LHSV->GetDataTree(true);
      std::string type = UnmangleVec(dt);

      Value *idx = Idx_Calc_Codegen(type, vec, LHSV->Idx, scope_struct);

      if(type=="list"||type=="dict") {
        std::string nested_type = dt.Nested_Data[0].Type;
        if (in_str(nested_type, primary_data_tokens))
          type = nested_type + "_" + type;
      }
      // LogBlue("New type is: " + type);
      
      if(ends_with(type, "dict"))
      {
        store_op = type+"_Store_Key";
        
        std::string RType = RHS->GetDataTree().Type;
        if (RType=="int" || RType=="float")
        {
          store_op = store_op + "_" + RType;

          call(store_op, {scope_struct, vec, idx, Val}); 
          return const_float(0);
        }
        
        call(store_op, {scope_struct, vec, idx, Val, global_str(RType)});
        return const_float(0);
      }

      
      call(type+"_Store_Idx", {vec, idx, Val, scope_struct});
      return const_float(0);
    } 

  
    

    

    std::string Lname = LHS->GetName();

    if (is_alloca)
    {
      Check_Is_Compatible_Data_Type(LHS->GetDataTree(), RHS->GetDataTree(), parser_struct);


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



      if(!in_str(LType, primary_data_tokens))
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

      if(!in_str(LType, primary_data_tokens))
        call("MarkToSweep_Unmark_Scopeful", {scope_struct, Val, global_str(LType)});
      if (LType=="float"&&RHS->GetType()=="int")
        Val = Builder->CreateUIToFP(Val, Type::getFloatTy(*TheContext), "floattmp");




      Builder->CreateStore(Val, alloca);
    } else
    {
      if(auto *LHSV = dynamic_cast<Nameable *>(LHS.get())) {
        Data_Tree L_dt = LHSV->GetDataTree();

        Check_Is_Compatible_Data_Type(L_dt, RHS->GetDataTree(), parser_struct);
        LType = L_dt.Type;


        LHSV->Load_Last=false;


        Value *obj_ptr = LHSV->codegen(scope_struct);
        
        if(LType=="float"||LType=="int")
          call("object_Attr_"+LType, {obj_ptr, Val});
        else
          call("tie_object_to_object", {obj_ptr, Val});
      } else {}
      

      if(!in_str(LType, primary_data_tokens))
        call("MarkToSweep_Unmark_Scopeless", {scope_struct, Val, global_str(LType)});   
    }
    

    seen_var_attr=false;
    return ConstantFP::get(*TheContext, APFloat(0.0f));
  }


  



  
  Value *L = LHS->codegen(scope_struct);
  Value *R = RHS->codegen(scope_struct);


  // GetType(); // gets the correct Elements and Operation.




  
  if (!L || !R)
    return nullptr;

  if (cast_L_to=="int_to_float") {
    L = Builder->CreateSIToFP(L, Type::getFloatTy(*TheContext), "lfp");
  }
  if (cast_R_to=="int_to_float") {
    R = Builder->CreateSIToFP(R, Type::getFloatTy(*TheContext), "lfp");
  }
  if (Operation=="tensor_int_div")
  {
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

    if(elements_type_return.count(Elements)>0)
    {
      std::string return_type = elements_type_return[Elements];

      if(!in_str(return_type, primary_data_tokens)&&return_type!="None")
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

































Value *UnaryExprAST::codegen(Value *scope_struct) {
  if (not ShallCodegen)
    return ConstantFP::get(*TheContext, APFloat(0.0f));
  Value *OperandV = Operand->codegen(scope_struct);
  if (!OperandV)
    return nullptr;
  
  
  
  std::string operand_type = Operand->GetDataTree().Type;
  if (Opcode=='-')
  {
    //std::cout << "\n\n\n\n\n\nIT'S A MINUS " << Operand->GetType() << "\n\n\n\n\n\n\n";
    if (operand_type=="tensor")
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

  std::cout << "Opcode: " << Opcode << "\n";

  if (Opcode==tok_not) {
    if(operand_type!="bool")
      LogError(parser_struct.line, "Cannot use not with type: " + operand_type);

    return Builder->CreateNot(OperandV, "logicalnot");
  }

  if (Opcode=='!')
  {
    return Builder->CreateCall(TheModule->getFunction("logical_not"), {OperandV});
  }
  if (Opcode==';')
    return OperandV;
    // return ConstantFP::get(Type::getFloatTy(*TheContext), 0);
  

  Function *F = getFunction(std::string("unary") + std::to_string(Opcode));
  if (!F)
    return LogErrorV(parser_struct.line,"Unknown unary operator.");

  return Builder->CreateCall(F, OperandV, "unop");
}






Function *codegenAsyncFunction(std::vector<std::unique_ptr<ExprAST>> &asyncBody, Value *scope_struct, Parser_Struct parser_struct, Value *barrier, std::string async_suffix) {
  


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


  // std::vector<Value *> previous_scope_values;
  std::vector<std::string> previous_scope_value_types;
  std::vector<std::string> previous_scope_value_names;


  for (auto &pair : function_allocas[parser_struct.function_name]) {

    std::string type;
    if (Object_toClass[parser_struct.function_name].count(pair.first)>0)
      type = "void";
    else
    {   
      type = UnmangleVec(data_typeVars[parser_struct.function_name][pair.first]);
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

  std::string async_scope = parser_struct.function_name + async_suffix;

  for(int i=0; i<previous_scope_value_names.size(); ++i) {
    std::string type = previous_scope_value_types[i];
    std::string var_name = previous_scope_value_names[i];

    Value *v = callret("emerge_"+type, {global_str(functionName), global_str(var_name)});

    llvm::Type *llvm_type = get_type_from_str(type);
    AllocaInst * alloca = CreateEntryBlockAlloca(asyncFun, var_name, llvm_type);
    Builder->CreateStore(v, alloca);
    function_allocas[async_scope][var_name] = alloca;
  }

  
  for (auto &body : asyncBody)
  {
    V = body->codegen(scope_struct_copy);
  }


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





Value *ChannelExprAST::codegen(Value *scope_struct) {
  Function *TheFunction = Builder->GetInsertBlock()->getParent();
  
  // std::string type = UnmangleVec(data_type);
  
    
  Value *void_ptr = Constant::getNullValue(Type::getInt8Ty(*TheContext)->getPointerTo());


  Value *initial_value = callret("channel_Create", {scope_struct, const_int(BufferSize)});




  llvm::Type *alloca_type = get_type_from_str("channel");
  AllocaInst *alloca = CreateEntryBlockAlloca(TheFunction, Name, alloca_type);
  Builder->CreateStore(initial_value, alloca);
  function_allocas[parser_struct.function_name][Name] = alloca;



  return const_float(0);
}



Value *GoExprAST::codegen(Value *scope_struct) {


  Value *barrier = callret("get_barrier", {const_int(1)});

  BasicBlock *CurrentBB = Builder->GetInsertBlock();


  
  Function *asyncFun = codegenAsyncFunction(Body, scope_struct, parser_struct, barrier, "_spawn");


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


  return const_float(0);
}












Value *AsyncExprAST::codegen(Value *scope_struct) {
  if (not ShallCodegen)
    return ConstantFP::get(*TheContext, APFloat(0.0f));

  
  // Create/Spawn Threads

  

  Value *barrier = callret("get_barrier", {const_int(1)});

  BasicBlock *CurrentBB = Builder->GetInsertBlock();


  Function *asyncFun = codegenAsyncFunction(Body, scope_struct, parser_struct, barrier, "_async");


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

  Function *asyncFun = codegenAsyncFunction(Body, scope_struct, parser_struct, barrier, "_asyncs");


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
  std::string type = UnmangleVec(Inner_Vec->GetDataTree());
  SetType(type);


  std::string split_fn = type + "_Split_Parallel";

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



  for (Value *pthreadPtr : thread_pointers)
  {
    Value *pthread = Builder->CreateLoad(pthreadTy, pthreadPtr);
    Builder->CreateCall(pthread_join, {pthread});
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

    if(!in_str(type, primary_data_tokens))
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
    
    if(!in_str(Vars[i]->GetDataTree().Type, primary_data_tokens))
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

  LogBlue("new tuple codegen");
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

  return callret("list_New", values);
}





Data_Tree NewVecExprAST::GetDataTree(bool from_assignment) {
  if(!data_type.empty)
    return data_type;

  data_type.Type = "tuple";
  for (int i=1; i<Values.size(); i=i+2)
  {
    std::string type = Values[i]->GetDataTree().Type;
    data_type.Nested_Data.push_back(Data_Tree(type));
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


    if(!in_str(type, primary_data_tokens))
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



inline std::vector<Value *> codegen_Argument_List(Parser_Struct parser_struct, std::vector<Value *> ArgsV, std::vector<std::unique_ptr<ExprAST>> Args, Value *scope_struct, std::string fn_name, bool is_nsk_fn, int arg_offset=1)
{

  // Get Arguments
  for (unsigned i = 0, e = Args.size(); i != e; ++i) {
    Value *arg; 

    arg = Args[i]->codegen(scope_struct);
    Data_Tree data_type = Args[i]->GetDataTree();
    std::string type = data_type.Type;

      

    if (!in_str(fn_name, {"to_int", "to_float"}))
    { 
      if (Function_Arg_Types.count(fn_name)>0)
      {
        int tgt_arg = i + arg_offset;

        std::string expected_type = Function_Arg_Types[fn_name][Function_Arg_Names[fn_name][tgt_arg]];
        Data_Tree expected_data_type = Function_Arg_DataTypes[fn_name][Function_Arg_Names[fn_name][tgt_arg]];
        int differences = expected_data_type.Compare(data_type);
        if (differences>0) { 
          LogError(parser_struct.line, "Got an incorrect type for argument " + Function_Arg_Names[fn_name][tgt_arg] + " of function " + fn_name);
          std::cout << "Passed\n   ";
          data_type.Print();
          std::cout << "\nExpected\n   ";
          expected_data_type.Print();
          std::cout << "\n\n";
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
    ExprAST *Init = VarNames[i].second.get();


    Value *var_name, *obj_name;// = Builder->CreateCall(TheModule->getFunction("GetEmptyChar"), {});
    
    
    if(!isSelf&&!isAttribute)
    {   
      AllocaInst *alloca = CreateEntryBlockAlloca(TheFunction, VarName, int8PtrTy);

      Value *ptr;
      if (Init==nullptr)
        ptr = callret("malloc", {const_int64(Size)});
      else
        ptr = Init->codegen(scope_struct);
      
      Builder->CreateStore(ptr, alloca);
      // Value *ptr = callret("posix_memalign", {alloca, const_int64(8), const_int64(Size)});
      // std::cout << "ADDING OBJECT " << VarName << " TO FUNCTION " << parser_struct.function_name << ".\n";
      function_allocas[parser_struct.function_name][VarName] = alloca;


      
      if (HasInit[i]) {
        Value *scope_struct_copy = callret("scope_struct_Copy", {scope_struct});
        call("set_scope_object", {scope_struct_copy, ptr});
        
        
        int arg_type_check_offset = 1;
        std::vector<Value *> ArgsV = {scope_struct_copy};

        std::string Callee = ClassName + "___init__";
        ArgsV = codegen_Argument_List(parser_struct, std::move(ArgsV), std::move(Args[i]), scope_struct, Callee, false, arg_type_check_offset);
        
        call(Callee, ArgsV);



        
        call("scope_struct_Delete", {scope_struct_copy});
      }
    }      
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



Value *NestedCallExprAST::codegen(Value *scope_struct) {

  

  
  bool is_nsk_fn = in_str(Callee, native_fn);

  
  Value *obj_ptr;
  Value *scope_struct_copy = callret("scope_struct_Copy", {scope_struct});

  call("set_scope_function_name", {scope_struct_copy, global_str(Callee)});


  // p2t("Calling: " + Callee);



  if(ends_with(Callee, "__init__")&&Inner_Expr->From_Self) // mallocs an object inside another
  {
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
  
    
  if(!in_str(Type, primary_data_tokens) && !in_str(Type, {"", "None"}))
    call("MarkToSweep_Mark", {scope_struct, ret, global_str(Extract_List_Suffix(Type))});
  
  return ret;
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
      return callret("get_scope_object", {scope_struct}); 
    return load_alloca(Name, type, parser_struct.function_name); 
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
  if(in_str(compound_type, compound_tokens)||ends_with(compound_type, "_vec"))
    type = inner_dt.Nested_Data[0].Type;
  else
    type = compound_type;



  Value *loaded_var = Inner->codegen(scope_struct);
  Value *idx = Idx_Calc_Codegen(compound_type, loaded_var, Idx, scope_struct);



  if (compound_type == "dict") {
    Value *ret_val = callret("dict_Query", {scope_struct, loaded_var, idx});
    if(type=="float"||type=="int")
      ret_val = callret("to_"+type, {scope_struct, ret_val});
    return ret_val;
  }
  
  if (!Idx->IsSlice) {
    std::string idx_fn = compound_type + "_Idx";

    Value *ret_val = callret(idx_fn, {scope_struct, loaded_var, idx});

    if(!(ends_with(compound_type,"_vec"))&&(type=="float"||type=="int"))
      ret_val = callret("to_"+type, {scope_struct, ret_val});
    
    return ret_val;
  } else {
    std::string slice_fn = compound_type + "_Slice";    
    Value *ret =  callret(slice_fn, {scope_struct, loaded_var, idx});
    call("Delete_Ptr", {idx});
    return ret;
  }
}

inline bool Check_Args_Count(const std::string &Callee, int target_args_size, Parser_Struct parser_struct) {
  Function *CalleeF;
  CalleeF = getFunction(Callee);
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


Value *NameableCall::codegen(Value *scope_struct) {  

  int arg_type_check_offset=1, target_args_size=Args.size()+1;
  bool is_nsk_fn = in_str(Callee, native_methods);
  Value *scope_struct_copy = callret("scope_struct_Copy", {scope_struct});

  call("set_scope_function_name", {scope_struct_copy, global_str(Callee)});


  std::vector<Value*> ArgsV = {scope_struct_copy};

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
        std::string obj_class = Inner->Inner->GetDataTree().Type;
        
        int size = ClassSize[obj_class];

        Value *new_ptr = callret("malloc", {const_int64(size)});


        call("tie_object_to_object", {obj_ptr, new_ptr});
        
        obj_ptr = new_ptr;

      }
      call("set_scope_object", {scope_struct_copy, obj_ptr});
    }
    else{
      ArgsV.push_back(obj_ptr);
      arg_type_check_offset++;
      target_args_size++;
    }
  }
  

  if(!Check_Args_Count(Callee, target_args_size, parser_struct))
    return const_float(0);


  ArgsV = codegen_Argument_List(parser_struct, std::move(ArgsV), std::move(Args), scope_struct, Callee, is_nsk_fn, arg_type_check_offset);
  



  // LogBlue("call " + Callee);
  Value *ret = callret(Callee, ArgsV);
  call("scope_struct_Delete", {scope_struct_copy});    



  if (ReturnType=="")
    GetDataTree();

  if(!in_str(ReturnType, primary_data_tokens)&&ReturnType!="")
    call("MarkToSweep_Mark", {scope_struct, ret, global_str(ReturnType)});
  

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
  


  Value *scope_struct_copy = callret("scope_struct_Copy", {scope_struct});
  Value *first_arg, *scope_string;




  
  
  Value *_pre_dot_str = global_str(_pre_dot);
  Value *first_arg_copy;


  int target_args_size = Args.size();
  std::vector<Value *> ArgsV; 
  

  


  

  bool is_self_of_nested_function = (nested_function==1 && isSelf);
  bool is_user_cpp_function = in_str(tgt_function, user_cpp_functions);
  bool is_nsk_fn = in_str(tgt_function, native_methods);
  

  
  
  call("set_scope_function_name", {scope_struct_copy, global_str(tgt_function)});



  


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
 


  Function *CalleeF;
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
    std::string _error = "Incorrect parameters used on function " + tgt_function + " call.\n\t    Expected " +  std::to_string(CalleeF->arg_size()-1) + " arguments, got " + std::to_string(target_args_size-1);
    return LogErrorV(parser_struct.line, _error);
  }





  // Sends the non-changed scope_struct to load/codegen the arguments from the argument list
  ArgsV = codegen_Argument_List(parser_struct, std::move(ArgsV), std::move(Args), scope_struct, tgt_function, is_nsk_fn);
  // Always include scope on the beggining
  ArgsV.insert(ArgsV.begin(), scope_struct_copy);



 
  

  
  Value *ret;
  if (CalleeOverride=="none")
  {
    ret = Builder->CreateCall(CalleeF, ArgsV, "calltmp");
    call("scope_struct_Delete", {scope_struct_copy});

    if(!in_str(Type, primary_data_tokens)&&Type!="")
      call("MarkToSweep_Mark", {scope_struct, ret, global_str(Extract_List_Suffix(Type))});
    
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
      ret = Builder->CreateCall(getFunction(CalleeOverride), ArgsV, "calltmp");

    call("scope_struct_Delete", {scope_struct_copy});    

    if(!in_str(Type, primary_data_tokens)&&Type!="")
      call("MarkToSweep_Mark", {scope_struct, ret, global_str(Extract_List_Suffix(Type))});
    
    return ret;
  }  
}



