

#include "llvm/IR/Value.h"


#include <map>
#include <vector>

#include "include.h"


#include <filesystem>
#include <fstream>


using namespace llvm;
namespace fs = std::filesystem;



//===----------------------------------------------------------------------===//
// Abstract Syntax Tree (aka Parse Tree)
//===----------------------------------------------------------------------===//
void ExprAST::SetType(std::string Type) {
  this->Type=Type;
  this->ReturnType=Type;
}
std::string ExprAST::GetType() {
  return Type;
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
  
  
  
  
StringExprAST::StringExprAST(std::string Val) : Val(Val) {
  this->SetType("str");
} 
  

NullPtrExprAST::NullPtrExprAST() {
  this->SetType("nullptr");
} 


VariableListExprAST::VariableListExprAST(std::vector<std::unique_ptr<ExprAST>> ExprList)
                      : ExprList(std::move(ExprList)) {
  this->SetIsList(true);
} 

  
  /// VariableExprAST - Expression class for referencing a variable, like "a".
VariableExprAST::VariableExprAST(std::unique_ptr<ExprAST> NameSolver, std::string Type, const std::string &Name, Parser_Struct parser_struct)
                                : Name(Name), parser_struct(parser_struct) {
  this->isVarLoad = true;
  this->NameSolver = std::move(NameSolver);
  this->SetType(Type);
  this->NameSolver->SetType(Type);

}
  
const std::string &VariableExprAST::getName() const { return Name; }
std::string VariableExprAST::GetName()  {
  return Name;
}
  
  
VecIdxExprAST::VecIdxExprAST(std::unique_ptr<ExprAST> Loaded_Var, std::vector<std::unique_ptr<ExprAST>> Idx, std::string Type)
              : Loaded_Var(std::move(Loaded_Var)), Idx(std::move(Idx)) {
  this->isVarLoad = true; //todo: remove this?
  this->SetType(Type);
}
  
const std::string &VecIdxExprAST::getName() const { return Name; }
std::string VecIdxExprAST::GetName()  { return Name; }

  
  
  
ObjectVecIdxExprAST::ObjectVecIdxExprAST(std::unique_ptr<ExprAST> Vec, std::string _post_dot, std::unique_ptr<ExprAST> Idx)
              : Vec(std::move(Vec)), _post_dot(_post_dot), Idx(std::move(Idx)) {
  this->isVarLoad = true;
}
  
  
/// VarExprAST - Expression class for var/in
VarExprAST::VarExprAST(
    std::vector<std::pair<std::string, std::unique_ptr<ExprAST>>> VarNames,
    std::string Type)
    : VarNames(std::move(VarNames)), Type(Type) {}
  
  


  
NewVecExprAST::NewVecExprAST(
    std::vector<std::unique_ptr<ExprAST>> Values,
    std::string Type)
    : Values(std::move(Values)), Type(Type) 
    {
      this->SetType(Type);
    }
  
  

ObjectExprAST::ObjectExprAST(
    Parser_Struct parser_struct,
  std::vector<std::pair<std::string, std::unique_ptr<ExprAST>>> VarNames,
  std::string Type,
  std::unique_ptr<ExprAST> Init, int Size)
  : parser_struct(parser_struct), VarExprAST(std::move(VarNames), std::move(Type)), Init(std::move(Init)), Size(Size) {}




 
  
DataExprAST::DataExprAST(
  Parser_Struct parser_struct,
  std::vector<std::pair<std::string, std::unique_ptr<ExprAST>>> VarNames,
  std::string Type,
  std::vector<std::unique_ptr<ExprAST>> Notes)
  : parser_struct(parser_struct), VarExprAST(std::move(VarNames), std::move(Type)),
                Notes(std::move(Notes)) {}


LibImportExprAST::LibImportExprAST(std::string LibName)
  : LibName(LibName) {


  std::string ai_path = LibName+".ai";

  if (fs::exists(ai_path)) {

    std::cout << "Importing high-level lib " << ai_path << ".\n";
    LibName_HighLevel = ai_path;

  } else {

    
    std::string lib_path = std::getenv("NSK_LIBS");

    std::string lib_dir = lib_path + "/" + LibName;
    std::string so_lib_path = lib_dir + "/lib.so";

    if(!fs::exists(so_lib_path))
    { 
        LogError("- Failed to import library;\n\t    - " + so_lib_path + " file not found.");
        return;
    }


    LibParser *lib_parser = new LibParser(lib_dir);
    
    lib_parser->ParseLibs();
    lib_parser->ImportLibs(so_lib_path);
  }
}

  
  
  
  
  
  
  
LSTMExprAST::LSTMExprAST(
  std::vector<std::pair<std::string, std::unique_ptr<ExprAST>>> VarNames,
  std::string Type,
  std::unique_ptr<ExprAST> C, std::unique_ptr<ExprAST> OC,
  const std::string &TensorInit)
  : VarExprAST(std::move(VarNames), std::move(Type)),
                C(std::move(C)), OC(std::move(OC)),
                TensorInit(TensorInit) {}
  
  
  
EmbeddingExprAST::EmbeddingExprAST(
  std::vector<std::pair<std::string, std::unique_ptr<ExprAST>>> VarNames,
  std::string Type,
  std::unique_ptr<ExprAST> C, std::unique_ptr<ExprAST> OC,
  const std::string &TensorInit)
  : VarExprAST(std::move(VarNames), std::move(Type)),
                C(std::move(C)), OC(std::move(OC)),
                TensorInit(TensorInit) {}
  
  
  
  
LinearExprAST::LinearExprAST(
  std::vector<std::pair<std::string, std::unique_ptr<ExprAST>>> VarNames,
  std::string Type,
  std::unique_ptr<ExprAST> C, std::unique_ptr<ExprAST> OC,
  std::vector<int> Notators,
  const std::string &TensorInit)
  : VarExprAST(std::move(VarNames), std::move(Type)),
                C(std::move(C)), OC(std::move(OC)),
                Notators(std::move(Notators)),
                TensorInit(TensorInit) {}
  
  
  
  
MHSAExprAST::MHSAExprAST(
  std::vector<std::pair<std::string, std::unique_ptr<ExprAST>>> VarNames,
  std::string Type,
  std::unique_ptr<ExprAST> nh, std::unique_ptr<ExprAST> C, std::unique_ptr<ExprAST> T,
  std::vector<int> Notators,
  const std::string &TensorInit)
  : VarExprAST(std::move(VarNames), std::move(Type)),
                nh(std::move(nh)), C(std::move(C)), T(std::move(T)),
                Notators(std::move(Notators)),
                TensorInit(TensorInit) {}
  
  
  
  
ReluExprAST::ReluExprAST(
  std::vector<std::pair<std::string, std::unique_ptr<ExprAST>>> VarNames,
  std::string Type)
  : VarExprAST(std::move(VarNames), std::move(Type)) {}
  
  
  
  /// UnaryExprAST - Expression class for a unary operator.
UnaryExprAST::UnaryExprAST(char Opcode, std::unique_ptr<ExprAST> Operand)
    : Opcode(Opcode), Operand(std::move(Operand)) {}
  
  
  
  /// BinaryExprAST - Expression class for a binary operator.
BinaryExprAST::BinaryExprAST(char Op, std::string Elements, std::string Operation, std::unique_ptr<ExprAST> LHS,
              std::unique_ptr<ExprAST> RHS, Parser_Struct parser_struct)
    : Op(Op), Elements(Elements), Operation(Operation), LHS(std::move(LHS)), RHS(std::move(RHS)), parser_struct(parser_struct) {}
  
  
  


  
  
  
  
  
  
  
BinaryObjExprAST::BinaryObjExprAST(char Op, std::unique_ptr<ExprAST> LHS,
              std::unique_ptr<ExprAST> RHS)
    : Op(Op), LHS(std::move(LHS)), RHS(std::move(RHS)) {}
  
  
  
  
  
  
  
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



ChainCallExprAST::ChainCallExprAST(const std::string &Call_Of,
                    std::vector<std::unique_ptr<ExprAST>> Args,
                    std::unique_ptr<ExprAST> Inner_Call)
    : Call_Of(Call_Of), Args(std::move(Args)), Inner_Call(std::move(Inner_Call)) {
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
          std::vector<std::unique_ptr<ExprAST>> Body, Parser_Struct parser_struct)
    : VarName(VarName), Vec(std::move(Vec)), Body(std::move(Body)), parser_struct(parser_struct) {}

  
  /// WhileExprAST - Expression class for while.
WhileExprAST::WhileExprAST(std::unique_ptr<ExprAST> Cond, std::vector<std::unique_ptr<ExprAST>> Body)
  : Cond(std::move(Cond)), Body(std::move(Body)) {}
  
  
  
  /// AsyncExprAST - Expression class for async.
AsyncExprAST::AsyncExprAST(std::vector<std::unique_ptr<ExprAST>> Body)
  : Body(std::move(Body)) {}
  
AsyncsExprAST::AsyncsExprAST(std::vector<std::unique_ptr<ExprAST>> Body, int AsyncsCount)
  : Body(std::move(Body)), AsyncsCount(AsyncsCount) {}

IncThreadIdExprAST::IncThreadIdExprAST()
{}

SplitParallelExprAST::SplitParallelExprAST(std::unique_ptr<ExprAST> Inner_Vec) : Inner_Vec(std::move(Inner_Vec)) {
  // Type = Inner_Vec->GetType();
}


SplitStridedParallelExprAST::SplitStridedParallelExprAST(std::unique_ptr<ExprAST> Inner_Vec) : Inner_Vec(std::move(Inner_Vec)) {
  // Type = Inner_Vec->GetType();
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
