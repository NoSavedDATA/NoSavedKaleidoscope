

#include "llvm/IR/Value.h"


#include <map>
#include <vector>

#include "include.h"




using namespace llvm;



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
std::vector<float> ExprAST::GetDims() {
  return Dims;
}
void ExprAST::SetDims(std::vector<float> Dims) {
  this->Dims=Dims;
}
Value *ExprAST::GetTensorPtr() {
  return TensorPtr;
}
    
  
NameSolverAST::NameSolverAST(std::vector<std::tuple<std::string, int, std::vector<std::unique_ptr<ExprAST>>>> Names)
                : Names(std::move(Names)) {} 
 
  
  /// NumberExprAST - Expression class for numeric literals like "1.0".
NumberExprAST::NumberExprAST(float Val) : Val(Val) {
  this->SetType("float");
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
VariableExprAST::VariableExprAST(std::unique_ptr<ExprAST> NameSolver, std::string Type) {
  this->isVarLoad = true;
  this->NameSolver = std::move(NameSolver);
  this->SetType(Type);
  this->NameSolver->SetType(Type);
}
  
const std::string &VariableExprAST::getName() const { return Name; }
std::string VariableExprAST::GetName()  {
  return Name;
}
  
  
VecIdxExprAST::VecIdxExprAST(std::unique_ptr<ExprAST> NameSolver, std::vector<std::unique_ptr<ExprAST>> Idx, std::string Type)
              : Idx(std::move(Idx)) {
  this->isVarLoad = true;
  this->NameSolver = std::move(NameSolver);
  this->SetType(Type);
  this->NameSolver->SetType(Type);
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
  std::vector<std::pair<std::string, std::unique_ptr<ExprAST>>> VarNames,
  std::string Type,
  std::unique_ptr<ExprAST> Init)
  : VarExprAST(std::move(VarNames), std::move(Type)), Init(std::move(Init)) {}




 
  
DataExprAST::DataExprAST(
  std::vector<std::pair<std::string, std::unique_ptr<ExprAST>>> VarNames,
  std::string Type,
  std::vector<std::unique_ptr<ExprAST>> Notes)
  : VarExprAST(std::move(VarNames), std::move(Type)),
                Notes(std::move(Notes)) {}

  
  
  
  
  
  
MaxPool2dExprAST::MaxPool2dExprAST(
  std::vector<std::pair<std::string, std::unique_ptr<ExprAST>>> VarNames,
  std::string Type,
  std::unique_ptr<ExprAST> Ks,
  std::unique_ptr<ExprAST> Stride, std::unique_ptr<ExprAST> Padding)
  : VarExprAST(std::move(VarNames), std::move(Type)),
                Ks(std::move(Ks)),
                Stride(std::move(Stride)), Padding(std::move(Padding)) {}

  
  
BatchNorm2dExprAST::BatchNorm2dExprAST(
  std::vector<std::pair<std::string, std::unique_ptr<ExprAST>>> VarNames,
  std::string Type,
  std::unique_ptr<ExprAST> C)
  : VarExprAST(std::move(VarNames), std::move(Type)),
                C(std::move(C)) {}

  
  
BN2dReluExprAST::BN2dReluExprAST(
  std::vector<std::pair<std::string, std::unique_ptr<ExprAST>>> VarNames,
  std::string Type,
  std::unique_ptr<ExprAST> C)
  : VarExprAST(std::move(VarNames), std::move(Type)),
                C(std::move(C)) {}
  
  
  
  
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
              std::unique_ptr<ExprAST> RHS)
    : Op(Op), Elements(Elements), Operation(Operation), LHS(std::move(LHS)), RHS(std::move(RHS)) {}
  
  
  
BinaryTensorScalarExprAST::BinaryTensorScalarExprAST(char Op, std::unique_ptr<ExprAST> LHS,
              std::unique_ptr<ExprAST> RHS)
    : Op(Op), LHS(std::move(LHS)), RHS(std::move(RHS)) {}
  
  

BinaryTensorTensorExprAST::BinaryTensorTensorExprAST(char Op, std::unique_ptr<ExprAST> LHS,
              std::unique_ptr<ExprAST> RHS)
    : Op(Op), LHS(std::move(LHS)), RHS(std::move(RHS)) {}


  
  
  
  
  
  
  
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
            const std::string &CalleeOverride)
    : NameSolver(std::move(NameSolver)), Callee(Callee), Name(Name), Args(std::move(Args)), Class(Class),
      PreDot(PreDot), Load_Type(Load_Type), IsVarForward(IsVarForward), CalleeOverride(CalleeOverride) {
  SetType("float");
}



ChainCallExprAST::ChainCallExprAST(const std::string &Call_Of,
                    std::vector<std::unique_ptr<ExprAST>> Args,
                    std::unique_ptr<ExprAST> Inner_Call)
    : Call_Of(Call_Of), Args(std::move(Args)), Inner_Call(std::move(Inner_Call)) {
  SetType("float");
}
  
  
ReturnExprAST::ReturnExprAST(std::vector<std::unique_ptr<ExprAST>> Vars, std::vector<bool> IsAs,
              std::vector<std::unique_ptr<ExprAST>> Destiny)
    : Vars(std::move(Vars)), IsAs(std::move(IsAs)), Destiny(std::move(Destiny)) {}
  
  
  
  /// IfExprAST - Expression class for if/then/else.
IfExprAST::IfExprAST(std::unique_ptr<ExprAST> Cond,
          std::vector<std::unique_ptr<ExprAST>> Then,
          std::vector<std::unique_ptr<ExprAST>> Else)
    : Cond(std::move(Cond)), Then(std::move(Then)), Else(std::move(Else)) {}
  
  
/// ForExprAST - Expression class for for.
ForExprAST::ForExprAST(const std::string &VarName, std::unique_ptr<ExprAST> Start,
          std::unique_ptr<ExprAST> End, std::unique_ptr<ExprAST> Step,
          std::vector<std::unique_ptr<ExprAST>> Body)
    : VarName(VarName), Start(std::move(Start)), End(std::move(End)),
      Step(std::move(Step)), Body(std::move(Body)) {}
  
  
  /// WhileExprAST - Expression class for while.
WhileExprAST::WhileExprAST(std::unique_ptr<ExprAST> Cond, std::vector<std::unique_ptr<ExprAST>> Body)
  : Cond(std::move(Cond)), Body(std::move(Body)) {}
  
  
  
  /// AsyncExprAST - Expression class for async.
AsyncExprAST::AsyncExprAST(std::vector<std::unique_ptr<ExprAST>> Body)
  : Body(std::move(Body)) {}
  
  
  
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
  
  
  
  
  
  
  
PrototypeAST::PrototypeAST(const std::string &Name, const std::string &Class, const std::string &Method,
              std::vector<std::string> Args,
              std::vector<std::string> Types,
              bool IsOperator, unsigned Prec)
      : Name(Name), Class(Class), Method(Method), Args(std::move(Args)), Types(std::move(Types)),
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
