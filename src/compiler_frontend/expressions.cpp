

#include "llvm/IR/Value.h"


#include <filesystem>
#include <fstream>
#include <map>
#include <string>
#include <vector>

#include "include.h"
#include "../data_types/data_tree.h"


using namespace llvm;
namespace fs = std::filesystem;



std::vector<std::string> imported_libs;
std::map<std::string, std::vector<std::string>> lib_submodules;


//===----------------------------------------------------------------------===//
// Abstract Syntax Tree (aka Parse Tree)
//===----------------------------------------------------------------------===//
void ExprAST::SetType(std::string Type) {
  this->Type=Type;
  this->ReturnType=Type;
}
std::string ExprAST::GetType(bool from_assignment) {
  return Type;
}
Data_Tree ExprAST::GetDataTree(bool from_assignment) {
  return Data_Tree(this->Type);
}
void ExprAST::SetReturnType(std::string ReturnType) {
  this->ReturnType=ReturnType;
}

void ExprAST::SetIsVarLoad(bool isVarLoad) {
  this->isVarLoad=isVarLoad;
}
bool ExprAST::GetIsVarLoad() {
  return isVarLoad;
}

bool ExprAST::GetNameSolveToLast() {
  return NameSolveToLast;
}
void ExprAST::SetNameSolveToLast(bool NameSolveToLast) {
  this->NameSolveToLast=NameSolveToLast;
}

void ExprAST::SetSelf(bool Self) {
  this->isSelf=Self;
}
bool ExprAST::GetSelf() {
  return isSelf;
}

void ExprAST::SetSolverIncludeScope(bool SolverIncludeScope) {
  this->SolverIncludeScope=SolverIncludeScope;
}
bool ExprAST::GetSolverIncludeScope() {
  return SolverIncludeScope;
}

void ExprAST::SetIsAttribute(bool Attribute) {
  this->isAttribute=Attribute;
}
bool ExprAST::GetIsAttribute() {
  return isAttribute;
}


void ExprAST::SetIsMsg(bool isMessage) {
  this->isMessage=isMessage;
}
bool ExprAST::GetIsMsg() {
  return isMessage;
}


void ExprAST::SetPreDot(std::string pre_dot) {
  this->_pre_dot=pre_dot;
}
std::string ExprAST::GetPreDot() {
  return _pre_dot;
}

std::string ExprAST::GetName() {
  return Name;
}
void ExprAST::SetName(std::string Name) {
  this->Name=Name;
}


void ExprAST::SetIsVec(bool isVec) {
  this->isVec=isVec;
}
bool ExprAST::GetIsVec() {
  return isVec;
}


void ExprAST::SetIsList(bool isList) {
  this->isList=isList;
}
bool ExprAST::GetIsList() {
  return isList;
}


// Tensor related
    
  
NameSolverAST::NameSolverAST(std::vector<std::tuple<std::string, int, std::vector<std::unique_ptr<ExprAST>>>> Names)
                : Names(std::move(Names)) {} 
 
  
  /// NumberExprAST - Expression class for numeric literals like "1.0".
NumberExprAST::NumberExprAST(float Val) : Val(Val) {
  this->SetType("float");
} 

IntExprAST::IntExprAST(int Val) : Val(Val) {
  this->SetType("int");
} 

BoolExprAST::BoolExprAST(bool Val) : Val(Val) {
  this->SetType("bool");
} 
  
  

  
StringExprAST::StringExprAST(std::string Val) : Val(Val) {
  this->SetType("str");
} 
  

NullPtrExprAST::NullPtrExprAST() {
  this->SetType("nullptr");
} 


VariableListExprAST::VariableListExprAST(std::vector<std::unique_ptr<Nameable>> ExprList)
                      : ExprList(std::move(ExprList)) {
  this->SetIsList(true);
} 

  
  /// VariableExprAST - Expression class for referencing a variable, like "a".
VariableExprAST::VariableExprAST(std::unique_ptr<ExprAST> NameSolver, bool CanBeString, const std::string &Name, Parser_Struct parser_struct)
                                : CanBeString(CanBeString), Name(Name), parser_struct(parser_struct) {
  this->isVarLoad = true;
  this->NameSolver = std::move(NameSolver);
}
  
const std::string &VariableExprAST::getName() const { return Name; }
std::string VariableExprAST::GetName()  {
  return Name;
}


  

  
  
  
  
  
/// VarExprAST - Expression class for var/in
VarExprAST::VarExprAST(
    std::vector<std::pair<std::string, std::unique_ptr<ExprAST>>> VarNames,
    std::string Type)
    : VarNames(std::move(VarNames)), Type(Type) {}
  
  


NewTupleExprAST::NewTupleExprAST(
    std::vector<std::unique_ptr<ExprAST>> Values)
    : Values(std::move(Values)) {}


  
NewVecExprAST::NewVecExprAST(
    std::vector<std::unique_ptr<ExprAST>> Values,
    std::string Type)
    : Values(std::move(Values)), Type(Type) 
{
  this->SetType(Type);
}


NewDictExprAST::NewDictExprAST(
    std::vector<std::unique_ptr<ExprAST>> Keys,
    std::vector<std::unique_ptr<ExprAST>> Values,
    std::string Type, Parser_Struct parser_struct)
    : Keys(std::move(Keys)), Values(std::move(Values)), Type(Type), parser_struct(parser_struct)
{
  this->SetType(Type);
}
  
  

ObjectExprAST::ObjectExprAST(
    Parser_Struct parser_struct,
  std::vector<std::pair<std::string, std::unique_ptr<ExprAST>>> VarNames,
  std::vector<bool> HasInit,
  std::vector<std::vector<std::unique_ptr<ExprAST>>> Args,
  std::string Type,
  std::unique_ptr<ExprAST> Init, int Size, std::string ClassName)
  : parser_struct(parser_struct), HasInit(HasInit), Args(std::move(Args)), VarExprAST(std::move(VarNames), std::move(Type)), Init(std::move(Init)), Size(Size), ClassName(ClassName)
{
  
}



void Print_Names_Str(std::vector<std::string> names_vec) {


  if(names_vec.size()==0)
    return;
  if (names_vec.size() == 1) {
    std::cout << "\n\nName: " << names_vec[0] << "\n\n\n";
    return;
  }

  
  std::cout << "\n\nName 2 strs: ";
  for (int i=0; i<=names_vec.size()-2; ++i)
    std::cout << names_vec[i] << "."; 
  std::cout << names_vec[names_vec.size()-1] << "\n\n\n"; 
}


NameableExprAST::NameableExprAST() {}

SelfExprAST::SelfExprAST() {
  Expr_String = {"self"};
  End_of_Recursion=true;
  Name="self";
  height=1;
  From_Self=true;
}
EmptyStrExprAST::EmptyStrExprAST() {
  Expr_String = {};
  End_of_Recursion=true;
  height=0;
}
NestedStrExprAST::NestedStrExprAST(std::unique_ptr<NameableExprAST> Inner_Expr, std::string name, Parser_Struct parser_struct) : parser_struct(parser_struct)
                                    {
  this->Inner_Expr = std::move(Inner_Expr);
  this->Inner_Expr->IsLeaf=false;
  Name = name;
  
  From_Self = this->Inner_Expr->From_Self;
  height=this->Inner_Expr->height+1;

  Expr_String = this->Inner_Expr->Expr_String;
  Expr_String.push_back(name);
}

NestedVectorIdxExprAST::NestedVectorIdxExprAST(std::unique_ptr<NameableExprAST> Inner_Expr, std::string name, Parser_Struct parser_struct, std::unique_ptr<IndexExprAST> Idx, std::string type)
                                        : parser_struct(parser_struct), Idx(std::move(Idx)) {
  this->Inner_Expr = std::move(Inner_Expr);
  this->Inner_Expr->IsLeaf=false;
  this->Name = name;
  this->SetType(type);
  
  height=this->Inner_Expr->height+1;

  Expr_String = this->Inner_Expr->Expr_String;
  Expr_String.push_back(name);
  // Print_Names_Str(Expr_String);
}



NestedCallExprAST::NestedCallExprAST(std::unique_ptr<NameableExprAST> Inner_Expr, std::string Callee, Parser_Struct parser_struct,
  std::vector<std::unique_ptr<ExprAST>> Args)
  : Inner_Expr(std::move(Inner_Expr)), Callee(Callee), parser_struct(parser_struct), Args(std::move(Args)) {
}


NestedVariableExprAST::NestedVariableExprAST(std::unique_ptr<NameableExprAST> Inner_Expr, Parser_Struct parser_struct, std::string type, Data_Tree data_type)
      : Inner_Expr(std::move(Inner_Expr)), parser_struct(parser_struct) {
  this->SetType(type);

  this->Name = this->Inner_Expr->Name;
  this->data_type = data_type;
}
 
UnkVarExprAST::UnkVarExprAST(
  Parser_Struct parser_struct,
  std::vector<std::pair<std::string, std::unique_ptr<ExprAST>>> VarNames,
  std::string Type,
  std::vector<std::unique_ptr<ExprAST>> Notes)
  : parser_struct(parser_struct), VarExprAST(std::move(VarNames), std::move(Type)),
                Notes(std::move(Notes)) {

  for (unsigned i = 0, e = this->VarNames.size(); i != e; ++i) {
    const std::string &VarName = this->VarNames[i].first; 
    ExprAST *Init = this->VarNames[i].second.get();

    Data_Tree dt = Init->GetDataTree();

    if(Init->GetIsMsg()) {
      if(dt.Nested_Data.size()==0)
        LogBlue("Failed to receive message from " + Init->GetName() + ". Is it a channel?");
      dt = dt.Nested_Data[0];
    }
        
    data_typeVars[parser_struct.function_name][VarName] = dt;
  }
}


TupleExprAST::TupleExprAST(
  Parser_Struct parser_struct,
  std::vector<std::pair<std::string, std::unique_ptr<ExprAST>>> VarNames,
  std::string Type,
  Data_Tree data_type) : parser_struct(parser_struct), VarExprAST(std::move(VarNames), std::move(Type)), data_type(data_type) {

    
  for (unsigned i = 0, e = this->VarNames.size(); i != e; ++i) {
    const std::string &VarName = this->VarNames[i].first; 
    ExprAST *Init = this->VarNames[i].second.get();

    std::string init_type = Init->GetType();
    Data_Tree other_type = Init->GetDataTree();
    
    if(this->Type=="tuple")
      Check_Is_Compatible_Data_Type(data_type, other_type, parser_struct);
    
    data_typeVars[parser_struct.function_name][VarName] = data_type;
  }
}

ListExprAST::ListExprAST(
  Parser_Struct parser_struct,
  std::vector<std::pair<std::string, std::unique_ptr<ExprAST>>> VarNames,
  std::string Type,
  Data_Tree data_type) : parser_struct(parser_struct), VarExprAST(std::move(VarNames), std::move(Type)), data_type(data_type) {

    
  for (unsigned i = 0, e = this->VarNames.size(); i != e; ++i) {
    const std::string &VarName = this->VarNames[i].first; 
    ExprAST *Init = this->VarNames[i].second.get();

    std::string init_type = Init->GetType();
    Data_Tree other_type = Init->GetDataTree();
    
    Check_Is_Compatible_Data_Type(data_type, other_type, parser_struct);
    
    data_typeVars[parser_struct.function_name][VarName] = data_type;
  }
}

DictExprAST::DictExprAST(
  Parser_Struct parser_struct,
  std::vector<std::pair<std::string, std::unique_ptr<ExprAST>>> VarNames,
  std::string Type,
  Data_Tree data_type) : parser_struct(parser_struct), VarExprAST(std::move(VarNames), std::move(Type)), data_type(data_type) {

    
  for (unsigned i = 0, e = this->VarNames.size(); i != e; ++i) {
    const std::string &VarName = this->VarNames[i].first; 
    ExprAST *Init = this->VarNames[i].second.get();

    std::string init_type = Init->GetType();
    Data_Tree other_type = Init->GetDataTree();
    
    Check_Is_Compatible_Data_Type(data_type, other_type, parser_struct);
    
    data_typeVars[parser_struct.function_name][VarName] = data_type;
  }
}

  
DataExprAST::DataExprAST(
  Parser_Struct parser_struct,
  std::vector<std::pair<std::string, std::unique_ptr<ExprAST>>> VarNames,
  std::string Type, Data_Tree data_type, bool HasNotes, bool IsStruct,
  std::vector<std::unique_ptr<ExprAST>> Notes)
  : parser_struct(parser_struct), VarExprAST(std::move(VarNames), std::move(Type)), data_type(data_type), HasNotes(HasNotes), IsStruct(IsStruct),
                Notes(std::move(Notes))
{  
  for (unsigned i = 0, e = this->VarNames.size(); i != e; ++i) {
    if(this->isSelf)
      continue;    
    const std::string &VarName = this->VarNames[i].first;  
    data_typeVars[parser_struct.function_name][VarName] = data_type;
  }
}


LibImportExprAST::LibImportExprAST(std::string LibName, bool IsDefault, Parser_Struct parser_struct)
  : LibName(LibName), IsDefault(IsDefault), parser_struct(parser_struct) {


  std::string ai_path = LibName+".ai";

  if (!(in_str(LibName, imported_libs))) {
    // std::cout << "import " << LibName << ".\n";
    
    bool has_nsk_ai=false, has_so_lib=false;
    std::string lib_path = std::getenv("NSK_LIBS");

    std::string lib_dir = lib_path + "/" + LibName;
    std::string so_lib_path = lib_dir + "/lib.so";

    if(fs::exists(so_lib_path))
    {
      has_so_lib=true;
      LibParser *lib_parser = new LibParser(lib_dir);
      
      lib_parser->ParseLibs();
      lib_parser->ImportLibs(so_lib_path, LibName, IsDefault);
    }



    std::string include_path = lib_dir + "/include.ai";
    if(fs::exists(include_path)) {
      has_nsk_ai=true;
      get_tok_util_space();
      tokenizer.importFile(include_path, 0);
    } else 
      getNextToken(); // eat lib name
    


    if(!(has_nsk_ai||has_so_lib))
      LogError(parser_struct.line, "Failed to import library: " + LibName + ".\n\t    Could not find .ai or lib.so file.");
    else
      imported_libs.push_back(LibName);
  }
}

  
  
  
  
  
  
  
  
  
  
  
  
  
  /// UnaryExprAST - Expression class for a unary operator.
UnaryExprAST::UnaryExprAST(int Opcode, std::unique_ptr<ExprAST> Operand, Parser_Struct parser_struct)
    : Opcode(Opcode), Operand(std::move(Operand)), parser_struct(parser_struct) {}
  
  



Data_Tree BinaryExprAST::GetDataTree(bool from_assignment) {

  std::string LType = LHS->GetDataTree().Type, RType = RHS->GetDataTree().Type;
  if ((LType=="list"||RType=="list") && Op!='=')
    LogError(parser_struct.line, "Tuple elements type are unknown during parsing type. Please load the element into a static type variable first.");
  
  Elements = LType + "_" + RType;    


  if (Elements=="int_float") {
    Elements = "float_float"; 
    cast_L_to="int_to_float";
  }
  if (Elements=="float_int") {
    Elements = "float_float"; 
    cast_R_to="int_to_float";
  }
  
  std::string operation = op_map[Op];
  Operation = Elements + "_" + operation;
  

  std::string type;
  if (Operation=="int_int_div")
    type = "float";
  else if (ops_type_return.count(Operation)>0)
  {
    type = ops_type_return[Operation];
  }
  else if (elements_type_return.count(Elements)>0)
    type = elements_type_return[Elements];
  else {}

  return Data_Tree(type);
}

  
  /// binaryexprAST - Expression class for a binary operator.
BinaryExprAST::BinaryExprAST(char Op, std::unique_ptr<ExprAST> LHS,
              std::unique_ptr<ExprAST> RHS, Parser_Struct parser_struct)
    : Op(Op), LHS(std::move(LHS)), RHS(std::move(RHS)), parser_struct(parser_struct) {}
  
  
  


  
  
  
  
  
  
  
  
  
  
  
  
  
CallExprAST::CallExprAST(std::unique_ptr<ExprAST> NameSolver,
            const std::string &Callee, const std::string &Name,
            std::vector<std::unique_ptr<ExprAST>> Args,
            const std::string &Class,
            const std::string &PreDot,
            const std::string &Load_Type,
            bool IsVarForward,
            const std::string &CalleeOverride,
            const std::string &Scope_Random_Str,
            const std::string &LoadOf,
            Parser_Struct parser_struct
          )
    : NameSolver(std::move(NameSolver)), Callee(Callee), Name(Name), Args(std::move(Args)), Class(Class),
      PreDot(PreDot), Load_Type(Load_Type), IsVarForward(IsVarForward), CalleeOverride(CalleeOverride),
      Scope_Random_Str(Scope_Random_Str), LoadOf(LoadOf), parser_struct(parser_struct) {
  SetType("float");
}



  


RetExprAST::RetExprAST(std::vector<std::unique_ptr<ExprAST>> Vars)
    : Vars(std::move(Vars)) {}
    
  
  
  
  
  /// IfExprAST - Expression class for if/then/else.
IfExprAST::IfExprAST(std::unique_ptr<ExprAST> Cond,
          std::vector<std::unique_ptr<ExprAST>> Then,
          std::vector<std::unique_ptr<ExprAST>> Else)
    : Cond(std::move(Cond)), Then(std::move(Then)), Else(std::move(Else)) {}
  
  
/// ForExprAST - Expression class for for.
ForExprAST::ForExprAST(const std::string &VarName, std::unique_ptr<ExprAST> Start,
          std::unique_ptr<ExprAST> End, std::unique_ptr<ExprAST> Step,
          std::vector<std::unique_ptr<ExprAST>> Body, Parser_Struct parser_struct)
    : VarName(VarName), Start(std::move(Start)), End(std::move(End)),
      Step(std::move(Step)), Body(std::move(Body)), parser_struct(parser_struct) {}
  

/// ForExprAST - Expression class for for.
ForEachExprAST::ForEachExprAST(const std::string &VarName, std::unique_ptr<ExprAST> Vec,
          std::vector<std::unique_ptr<ExprAST>> Body, Parser_Struct parser_struct, Data_Tree data_type)
    : VarName(VarName), Vec(std::move(Vec)), Body(std::move(Body)), parser_struct(parser_struct) {

    this->data_type = data_type;
    Type = data_type.Nested_Data[0].Type;
}


  
  /// WhileExprAST - Expression class for while.
WhileExprAST::WhileExprAST(std::unique_ptr<ExprAST> Cond, std::vector<std::unique_ptr<ExprAST>> Body)
  : Cond(std::move(Cond)), Body(std::move(Body)) {}



IndexExprAST::IndexExprAST(std::vector<std::unique_ptr<ExprAST>> Idxs, std::vector<std::unique_ptr<ExprAST>> Second_Idxs, bool IsSlice)
            : Idxs(std::move(Idxs)), Second_Idxs(std::move(Second_Idxs)), IsSlice(IsSlice) {
  Size = this->Idxs.size();
}
  



ChannelExprAST::ChannelExprAST(Parser_Struct parser_struct, Data_Tree data_type, std::string Name, int BufferSize, bool isSelf) : parser_struct(parser_struct), BufferSize(BufferSize) {
  this->data_type = data_type;
  this->Name = Name;
  this->isSelf = isSelf;
}

GoExprAST::GoExprAST(std::vector<std::unique_ptr<ExprAST>> Body, Parser_Struct parser_struct) : Body(std::move(Body)), parser_struct(parser_struct) {  

}
  
  /// AsyncExprAST - Expression class for async.
AsyncExprAST::AsyncExprAST(std::vector<std::unique_ptr<ExprAST>> Body, Parser_Struct parser_struct)
  : Body(std::move(Body)), parser_struct(parser_struct) {}
  
AsyncsExprAST::AsyncsExprAST(std::vector<std::unique_ptr<ExprAST>> Body, int AsyncsCount, Parser_Struct parser_struct)
  : Body(std::move(Body)), AsyncsCount(AsyncsCount), parser_struct(parser_struct) {}

IncThreadIdExprAST::IncThreadIdExprAST()
{}



Data_Tree SplitParallelExprAST::GetDataTree(bool from_assignment) {
  return Inner_Vec->GetDataTree();
}

SplitParallelExprAST::SplitParallelExprAST(std::unique_ptr<ExprAST> Inner_Vec) : Inner_Vec(std::move(Inner_Vec)) {
}

Data_Tree SplitStridedParallelExprAST::GetDataTree(bool from_assignment) {
  return Inner_Vec->GetDataTree();
}

SplitStridedParallelExprAST::SplitStridedParallelExprAST(std::unique_ptr<ExprAST> Inner_Vec) : Inner_Vec(std::move(Inner_Vec)) {
}
  
  /// FinishExprAST - Expression class for finish/async.
FinishExprAST::FinishExprAST(std::vector<std::unique_ptr<ExprAST>> Bodies,
              std::vector<bool> IsAsync)
        : Bodies(std::move(Bodies)), IsAsync(std::move(IsAsync)) {}
  
  
  
  
  /// LockExprAST
LockExprAST::LockExprAST(std::vector<std::unique_ptr<ExprAST>> Bodies,
            std::string Name)
        : Bodies(std::move(Bodies)), Name(Name) {}

  
NoGradExprAST::NoGradExprAST(std::vector<std::unique_ptr<ExprAST>> Bodies)
        : Bodies(std::move(Bodies)) {}
  
  
  
MainExprAST::MainExprAST(std::vector<std::unique_ptr<ExprAST>> Bodies)
        : Bodies(std::move(Bodies)) {}
  
  
  
  
PrototypeAST::PrototypeAST(const std::string &Name, const std::string &Return_Type, const std::string &Class, const std::string &Method,
              std::vector<std::string> Args,
              std::vector<std::string> Types,
              bool IsOperator, unsigned Prec)
      : Name(Name), Return_Type(Return_Type), Class(Class), Method(Method), Args(std::move(Args)), Types(std::move(Types)),
        IsOperator(IsOperator), Precedence(Prec) {}

const std::string &PrototypeAST::getName() const { return Name; }
const std::string &PrototypeAST::getClass() const { return Class; }
const std::string &PrototypeAST::getMethod() const { return Method; }

bool PrototypeAST::isUnaryOp() const { return IsOperator && Args.size() == 1; }
bool PrototypeAST::isBinaryOp() const { return IsOperator && Args.size() == 2; }

char PrototypeAST::getOperatorName() const {
  assert(isUnaryOp() || isBinaryOp());
  return Name[Name.size() - 1];
}



unsigned PrototypeAST::getBinaryPrecedence() const { return Precedence; }





Nameable *Nameable::InnerMost() {
  if (Inner->Depth==0)
    return this;
  return Inner->InnerMost(); 
}


std::string Nameable::GetLibCallee() {
  if (Inner->Depth==0)
    return Name;
  return Inner->GetLibCallee() + "__" + Name;
}


Data_Tree NameableIdx::GetDataTree(bool from_assignment) {
  

  Data_Tree inner_dt = Inner->GetDataTree();

  std::string compound_type = inner_dt.Type;
  

  if (from_assignment || !in_str(compound_type, {"vec", "list", "dict", "tuple"}))
    return inner_dt; //e.g: for list_Store_Idx

  if(compound_type=="tuple") {
    if (IntExprAST *expr = dynamic_cast<IntExprAST*>(Idx->Idxs[0].get())) {

      int idx = expr->Val;
      if (idx>=inner_dt.Nested_Data.size())
        LogError(parser_struct.line, "Tuple index out of range. Index at: " + std::to_string(idx) + ", but the tuple size is " + std::to_string(inner_dt.Nested_Data.size()));

      return Data_Tree(inner_dt.Nested_Data[idx].Type);
    } else
      LogError(parser_struct.line, "Can only index tuple with a constant integer.");
  }

  return inner_dt.Nested_Data[0];
}


Data_Tree NameableCall::GetDataTree(bool from_assignment) { 
  if (data_type.Type!="")
    return data_type;

  Data_Tree ret = functions_return_data_type[Callee];


  std::string ret_type = ret.Type;
  if (ends_with(ret_type, "_vec")) {
    Data_Tree return_dt = Data_Tree("vec");
    return_dt.Nested_Data.push_back(remove_suffix(ret_type, "_vec"));
    ret = return_dt;
  }

  if(Callee=="zip") {

    Data_Tree return_dt = Data_Tree("list");
    return_dt.Nested_Data.push_back(Data_Tree("list"));

    for(int i=0; i<Args.size(); ++i) {

      std::string type = Args[i]->GetDataTree().Nested_Data[0].Type;
      return_dt.Nested_Data[0].Nested_Data.push_back(Data_Tree(type));
    }

    ret = return_dt;
  }

  data_type = ret;
  ReturnType = ret.Type;

  return ret;
}


Data_Tree Nameable::GetDataTree(bool from_assignment) {
  
  if(Depth==1) {
    if(Name=="self")
      return Data_Tree(parser_struct.class_name);
    else if(data_typeVars[parser_struct.function_name].find(Name)!=data_typeVars[parser_struct.function_name].end())
        return data_typeVars[parser_struct.function_name][Name];
    else
      LogError(parser_struct.line, "Could not find variable " + Name + " on scope " + parser_struct.function_name + ".");
  }

  
  std::string scope = Inner->GetDataTree().Type;


  if(data_typeVars[scope].find(Name)!=data_typeVars[scope].end())
    return data_typeVars[scope][Name];
  else
    LogError(parser_struct.line, "Could not find variable " + Name + " on scope " + scope+". Depth: " + std::to_string(Depth));
}




Nameable::Nameable(Parser_Struct parser_struct) : parser_struct(parser_struct) {}

Nameable::Nameable(Parser_Struct parser_struct, std::string Name, int Depth) : parser_struct(parser_struct), Depth(Depth) {
  this->Name = Name;
  this->isAttribute = Depth>1;
  this->isSelf = (Depth==1&&Name=="self");
}


void Nameable::AddNested(std::unique_ptr<Nameable> Inner) {
  this->Inner = std::move(Inner);
  this->Inner->IsLeaf = false;
  this->isSelf = this->isSelf||this->Inner->isSelf;
}

NameableRoot::NameableRoot(Parser_Struct parser_struct) : Nameable(parser_struct) {
  Depth = 0;
  Name = "";
}

NameableCall::NameableCall(Parser_Struct parser_struct, std::unique_ptr<Nameable> Inner, std::vector<std::unique_ptr<ExprAST>> Args) : Nameable(parser_struct), Args(std::move(Args)) {
  this->Inner = std::move(Inner);
  this->Inner->IsLeaf = false;
  this->isSelf = this->Inner->isSelf;

  
  Depth = this->Inner->Depth;

  Callee = this->Inner->Name;

  
  if (Depth==1 && lib_function_remaps.count(Callee)>0)
    Callee = lib_function_remaps[Callee];

  if(Depth>1) {    
    std::string inner_most_name = this->Inner->InnerMost()->Name;


    if (in_str(inner_most_name, imported_libs))
    {
      FromLib=true;
      Callee = this->Inner->GetLibCallee();
    }
    else {  

      Data_Tree inner_dt = this->Inner->Inner->GetDataTree();
      if(data_typeVars[inner_dt.Type].find(Callee)!=data_typeVars[inner_dt.Type].end()) // self.linear1(x)    
        Callee = UnmangleVec(data_typeVars[inner_dt.Type][Callee]);
      else { // x.view()
        this->Inner = std::move(this->Inner->Inner);
        Callee = UnmangleVec(inner_dt) + "_" + Callee;
      }
    }

    
    // LogBlue("Callee is " + Callee);
  }


  if (in_str(Callee, vararg_methods))
  {
    if (Callee=="zip") {
      GetDataTree();
      this->Args.push_back(std::make_unique<NullPtrExprAST>());
    }
    else
      this->Args.push_back(std::make_unique<IntExprAST>(TERMINATE_VARARG));
  }
}


NameableIdx::NameableIdx(Parser_Struct parser_struct, std::unique_ptr<Nameable> Inner, std::unique_ptr<IndexExprAST> Idx) : Nameable(parser_struct), Idx(std::move(Idx)) {
  this->Inner = std::move(Inner); 
  this->Inner->IsLeaf = false;
  this->isSelf = this->Inner->isSelf;
}