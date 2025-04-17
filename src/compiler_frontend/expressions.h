#pragma once

#include "llvm/IR/Value.h"


using namespace llvm;

//===----------------------------------------------------------------------===//
// Abstract Syntax Tree (aka Parse Tree)
//===----------------------------------------------------------------------===//

/// ExprAST - Base class for all expression nodes.
class ExprAST {
  public:
    virtual ~ExprAST() = default;
    std::vector<float> Dims = {-1.0f};
    std::string Type = "None";
    std::string ReturnType = "None";
    std::string Name = "Unnamed";
    bool isSelf = false;
    bool isAttribute = false;
    std::string _pre_dot = "";
    bool isVec = false;
    bool isVarLoad = false;
    bool SolverIncludeScope = true;
    bool NameSolveToLast = true;
  
    Value *TensorPtr;
  
  
    virtual Value *codegen(Value *scope_struct) = 0;
    virtual void SetType(std::string Type);
    virtual std::string GetType();
    virtual void SetReturnType(std::string ReturnType);
  
    virtual void SetIsVarLoad(bool isVarLoad);
    virtual bool GetIsVarLoad();
  
    virtual bool GetNameSolveToLast(); 
    virtual void SetNameSolveToLast(bool NameSolveToLast); 
  
    virtual void SetSelf(bool Self); 
    virtual bool GetSelf(); 
  
    virtual void SetSolverIncludeScope(bool SolverIncludeScope); 
    virtual bool GetSolverIncludeScope(); 
  
    virtual void SetIsAttribute(bool Attribute); 
    virtual bool GetIsAttribute(); 
    
  
    virtual void SetPreDot(std::string pre_dot); 
    virtual std::string GetPreDot(); 
  
    virtual std::string GetName(); 
    virtual void SetName(std::string Name); 
  
    
    virtual void SetIsVec(bool isVec); 
    virtual bool GetIsVec(); 
  
    // Tensor related
    virtual std::vector<float> GetDims(); 
    virtual void SetDims(std::vector<float> Dims); 
    virtual Value *GetTensorPtr(); 
    
  };
  
  
  
  class NameSolverAST : public ExprAST {
  
    public:
      std::vector<std::tuple<std::string, int, std::vector<std::unique_ptr<ExprAST>>>> Names;
      NameSolverAST(std::vector<std::tuple<std::string, int, std::vector<std::unique_ptr<ExprAST>>>> Names);
    
  
    Value *codegen(Value *scope_struct) override;
  };
  
  
  /// NumberExprAST - Expression class for numeric literals like "1.0".
  class NumberExprAST : public ExprAST {
    float Val;
  
    public:
      NumberExprAST(float Val); 
  
    Value *codegen(Value *scope_struct) override;
  };
  
  
  
  class StringExprAST : public ExprAST {
    std::string Val;
  
    public:
      StringExprAST(std::string Val); 
  
    Value *codegen(Value *scope_struct) override;
  };
  
  
  
  /// VariableExprAST - Expression class for referencing a variable, like "a".
  class VariableExprAST : public ExprAST {
  
    public:
      std::unique_ptr<ExprAST> NameSolver;
      VariableExprAST(std::unique_ptr<ExprAST> NameSolver, std::string Type); 
  
      Value *codegen(Value *scope_struct) override;
      const std::string &getName() const; 
      std::string GetName() override; 
  };
  
  
  class VecIdxExprAST : public ExprAST {
    
    public:
      std::unique_ptr<ExprAST> NameSolver;
      std::vector<std::unique_ptr<ExprAST>> Idx;
  
      VecIdxExprAST(std::unique_ptr<ExprAST> NameSolver, std::vector<std::unique_ptr<ExprAST>> Idx, std::string Type);
  
      Value *codegen(Value *scope_struct) override;
      const std::string &getName() const;
      std::string GetName() override;
  };
  
  
  
  class ObjectVecIdxExprAST : public ExprAST {
  
    public:
      std::unique_ptr<ExprAST> Vec, Idx;
      std::string _post_dot;
  
      ObjectVecIdxExprAST(std::unique_ptr<ExprAST> Vec, std::string _post_dot, std::unique_ptr<ExprAST> Idx);
  
      Value *codegen(Value *scope_struct) override;
  };
  
  /// VarExprAST - Expression class for var/in
  class VarExprAST : public ExprAST {
  
    public:
      std::vector<std::pair<std::string, std::unique_ptr<ExprAST>>> VarNames;
      
      std::string Type;
      VarExprAST(
          std::vector<std::pair<std::string, std::unique_ptr<ExprAST>>> VarNames,
          std::string Type);
  
    Value *codegen(Value *scope_struct) override;
  };
  
  
  
  class StrVecExprAST : public ExprAST {
  
    public:
      std::vector<std::pair<std::string, std::unique_ptr<ExprAST>>> VarNames;
      std::string Type;
      
      StrVecExprAST(
          std::vector<std::pair<std::string, std::unique_ptr<ExprAST>>> VarNames,
          std::string Type);
  
    Value *codegen(Value *scope_struct) override;
  };
  
  
  class NewVecExprAST : public ExprAST {
  
    public:
      std::vector<std::unique_ptr<ExprAST>> Values;
      std::string Type;
      
      NewVecExprAST(
          std::vector<std::unique_ptr<ExprAST>> Values,
          std::string Type);
  
    Value *codegen(Value *scope_struct) override;
  };
  
  
  class ObjectExprAST : public VarExprAST {
  
  public:
    std::unique_ptr<ExprAST> Init;
  
    ObjectExprAST(
        std::vector<std::pair<std::string, std::unique_ptr<ExprAST>>> VarNames,
        std::string Type,
        std::unique_ptr<ExprAST> Init);
  
    Value *codegen(Value *scope_struct) override;
  };
  
  
   
  
  
  
  class DataExprAST : public VarExprAST {
    public:
      std::vector<std::unique_ptr<ExprAST>> Notes;
  
      DataExprAST(
        std::vector<std::pair<std::string, std::unique_ptr<ExprAST>>> VarNames,
        std::string Type,
        std::vector<std::unique_ptr<ExprAST>> Notes);
  
    Value *codegen(Value *scope_struct) override;
  };


  
  class Conv2dExprAST : public VarExprAST {
    public:
      std::unique_ptr<ExprAST> C, OC, Ks, Stride, Padding;
      std::string TensorInit;
  
      Conv2dExprAST(
        std::vector<std::pair<std::string, std::unique_ptr<ExprAST>>> VarNames,
        std::string Type,
        std::unique_ptr<ExprAST> C, std::unique_ptr<ExprAST> OC, std::unique_ptr<ExprAST> Ks,
        std::unique_ptr<ExprAST> Stride, std::unique_ptr<ExprAST> Padding,
        const std::string &TensorInit);
  
    Value *codegen(Value *scope_struct) override;
  };
  
  
  
  class MaxPool2dExprAST : public VarExprAST {
    public:
      std::unique_ptr<ExprAST> Ks, Stride, Padding;
  
      MaxPool2dExprAST(
        std::vector<std::pair<std::string, std::unique_ptr<ExprAST>>> VarNames,
        std::string Type,
        std::unique_ptr<ExprAST> Ks,
        std::unique_ptr<ExprAST> Stride, std::unique_ptr<ExprAST> Padding);
  
    Value *codegen(Value *scope_struct) override;
  };
  
  
  class BatchNorm2dExprAST : public VarExprAST {
    public:
      std::unique_ptr<ExprAST> C;
  
      BatchNorm2dExprAST(
        std::vector<std::pair<std::string, std::unique_ptr<ExprAST>>> VarNames,
        std::string Type,
        std::unique_ptr<ExprAST> C);
  
    Value *codegen(Value *scope_struct) override;
  };
  
  
  class BN2dReluExprAST : public VarExprAST {
    public:
      std::unique_ptr<ExprAST> C;
  
      BN2dReluExprAST(
        std::vector<std::pair<std::string, std::unique_ptr<ExprAST>>> VarNames,
        std::string Type,
        std::unique_ptr<ExprAST> C);
  
    Value *codegen(Value *scope_struct) override;
  };
  
  
  
  class LSTMExprAST : public VarExprAST {
    public:
      std::unique_ptr<ExprAST> C, OC;
      std::string TensorInit;
  
      LSTMExprAST(
        std::vector<std::pair<std::string, std::unique_ptr<ExprAST>>> VarNames,
        std::string Type,
        std::unique_ptr<ExprAST> C, std::unique_ptr<ExprAST> OC,
        const std::string &TensorInit);
  
    Value *codegen(Value *scope_struct) override;
  };
  
  
  class EmbeddingExprAST : public VarExprAST {
    public:
      std::unique_ptr<ExprAST> C, OC;
      std::string TensorInit;
  
      EmbeddingExprAST(
        std::vector<std::pair<std::string, std::unique_ptr<ExprAST>>> VarNames,
        std::string Type,
        std::unique_ptr<ExprAST> C, std::unique_ptr<ExprAST> OC,
        const std::string &TensorInit);
  
    Value *codegen(Value *scope_struct) override;
  };
  
  
  
  class LinearExprAST : public VarExprAST {
    public:
      std::unique_ptr<ExprAST> C, OC;
      std::string TensorInit;
      std::vector<int> Notators;
  
      LinearExprAST(
        std::vector<std::pair<std::string, std::unique_ptr<ExprAST>>> VarNames,
        std::string Type,
        std::unique_ptr<ExprAST> C, std::unique_ptr<ExprAST> OC,
        std::vector<int> Notators,
        const std::string &TensorInit);
  
    Value *codegen(Value *scope_struct) override;
  };
  
  
  
  class MHSAExprAST : public VarExprAST {
    public:
      std::unique_ptr<ExprAST> nh, C, T;
      std::string TensorInit;
      std::vector<int> Notators;
  
      MHSAExprAST(
        std::vector<std::pair<std::string, std::unique_ptr<ExprAST>>> VarNames,
        std::string Type,
        std::unique_ptr<ExprAST> nh, std::unique_ptr<ExprAST> C, std::unique_ptr<ExprAST> T,
        std::vector<int> Notators,
        const std::string &TensorInit);
  
    Value *codegen(Value *scope_struct) override;
  };
  
  
  
  class ReluExprAST : public VarExprAST {
    public:
  
      ReluExprAST(
        std::vector<std::pair<std::string, std::unique_ptr<ExprAST>>> VarNames,
        std::string Type);
  
    Value *codegen(Value *scope_struct) override;
  };
  
  
  /// UnaryExprAST - Expression class for a unary operator.
  class UnaryExprAST : public ExprAST {
    char Opcode;
    std::unique_ptr<ExprAST> Operand;
  
  public:
    UnaryExprAST(char Opcode, std::unique_ptr<ExprAST> Operand);
  
    Value *codegen(Value *scope_struct) override;
  };
  
  
  /// BinaryExprAST - Expression class for a binary operator.
  class BinaryExprAST : public ExprAST {
    char Op;
    std::unique_ptr<ExprAST> LHS, RHS;
    std::string Elements, Operation;
  
  public:
    BinaryExprAST(char Op, std::string Elements, std::string Operation, std::unique_ptr<ExprAST> LHS,
                  std::unique_ptr<ExprAST> RHS);
  
    Value *codegen(Value *scope_struct) override;
  };
  
  
  class BinaryTensorScalarExprAST : public ExprAST {
    char Op;
    std::unique_ptr<ExprAST> LHS, RHS;
  
  public:
    BinaryTensorScalarExprAST(char Op, std::unique_ptr<ExprAST> LHS,
                  std::unique_ptr<ExprAST> RHS);
  
    Value *codegen(Value *scope_struct) override;
  };
  
  
  class BinaryTensorTensorExprAST : public ExprAST {
    char Op;
    std::unique_ptr<ExprAST> LHS, RHS;
  
  public:
    BinaryTensorTensorExprAST(char Op, std::unique_ptr<ExprAST> LHS,
                  std::unique_ptr<ExprAST> RHS);
  
    Value *codegen(Value *scope_struct) override;
  };
  
  
  
  
  
  class BinaryObjExprAST : public ExprAST {
    char Op;
    std::unique_ptr<ExprAST> LHS, RHS;
  
  public:
    BinaryObjExprAST(char Op, std::unique_ptr<ExprAST> LHS,
                  std::unique_ptr<ExprAST> RHS);
  
    Value *codegen(Value *scope_struct) override;
  };
  
  
  
  
  
  /// CallExprAST - Expression class for function calls.
  class CallExprAST : public ExprAST {
    std::string Callee;
    std::vector<std::unique_ptr<ExprAST>> Args;
    std::string Class;
    std::string PreDot;
    bool IsVarForward;
    std::string CalleeOverride;
    std::unique_ptr<ExprAST> NameSolver;
  
    public:
      CallExprAST(std::unique_ptr<ExprAST> NameSolver,
                  const std::string &Callee,
                  std::vector<std::unique_ptr<ExprAST>> Args,
                  const std::string &Class,
                  const std::string &PreDot,
                  bool IsVarForward,
                  const std::string &CalleeOverride);
  
    Value *codegen(Value *scope_struct) override;
  };
  
  class ReturnExprAST : public ExprAST {
  
    public:
      std::vector<std::unique_ptr<ExprAST>> Vars;
      std::vector<bool> IsAs;
      std::vector<std::unique_ptr<ExprAST>> Destiny;
      
      ReturnExprAST(std::vector<std::unique_ptr<ExprAST>> Vars, std::vector<bool> IsAs,
                    std::vector<std::unique_ptr<ExprAST>> Destiny);
  
    Value *codegen(Value *scope_struct) override;
  };
  
  
  /// IfExprAST - Expression class for if/then/else.
  class IfExprAST : public ExprAST {
    std::unique_ptr<ExprAST> Cond;
    std::vector<std::unique_ptr<ExprAST>> Then, Else;
  
    public:
      IfExprAST(std::unique_ptr<ExprAST> Cond,
                std::vector<std::unique_ptr<ExprAST>> Then,
                std::vector<std::unique_ptr<ExprAST>> Else);
  
    Value *codegen(Value *scope_struct) override;
  };
  
  /// ForExprAST - Expression class for for.
  class ForExprAST : public ExprAST {
    std::string VarName;
    std::unique_ptr<ExprAST> Start, End, Step;
    std::vector<std::unique_ptr<ExprAST>> Body;
  
    public:
      ForExprAST(const std::string &VarName, std::unique_ptr<ExprAST> Start,
                std::unique_ptr<ExprAST> End, std::unique_ptr<ExprAST> Step,
                std::vector<std::unique_ptr<ExprAST>> Body);
  
    Value *codegen(Value *scope_struct) override;
  };
  
  /// WhileExprAST - Expression class for while.
  class WhileExprAST : public ExprAST {
    std::unique_ptr<ExprAST> Cond;
    std::vector<std::unique_ptr<ExprAST>> Body;
  
    public:
      WhileExprAST(std::unique_ptr<ExprAST> Cond, std::vector<std::unique_ptr<ExprAST>> Body);
  
    Value* codegen(Value *scope_struct) override;
  };
  
  
  /// AsyncExprAST - Expression class for async.
  class AsyncExprAST : public ExprAST {
    std::vector<std::unique_ptr<ExprAST>> Body;
  
    public:
      AsyncExprAST(std::vector<std::unique_ptr<ExprAST>> Body);
  
    Value* codegen(Value *scope_struct) override;
  };
  
  
  /// FinishExprAST - Expression class for finish/async.
  class FinishExprAST : public ExprAST {
    std::vector<std::unique_ptr<ExprAST>> Bodies;
    std::vector<bool> IsAsync;
  
    public:
      FinishExprAST(std::vector<std::unique_ptr<ExprAST>> Bodies,
                    std::vector<bool> IsAsync);
  
  
    Value* codegen(Value *scope_struct) override;
  };
  
  
  /// LockExprAST
  class LockExprAST : public ExprAST {
    std::vector<std::unique_ptr<ExprAST>> Bodies;
    std::string Name;
  
    public:
      LockExprAST(std::vector<std::unique_ptr<ExprAST>> Bodies,
                  std::string Name);
  
  
    Value* codegen(Value *scope_struct) override;
  };
  /// NoGradExprAST
  class NoGradExprAST : public ExprAST {
    std::vector<std::unique_ptr<ExprAST>> Bodies;
  
    public:
      NoGradExprAST(std::vector<std::unique_ptr<ExprAST>> Bodies);
  
  
    Value* codegen(Value *scope_struct) override;
  };
  
  
  
  
  
  /// PrototypeAST - This class represents the "prototype" for a function,
  /// which captures its name, and its argument names (thus implicitly the number
  /// of arguments the function takes), as well as if it is an operator.
  class PrototypeAST {
  
    std::string Name;
    std::string Class;
    std::string Method;
  
    std::vector<std::string> Args;
    std::vector<std::string> Types;
    bool IsOperator;
    unsigned Precedence; // Precedence if a binary op.
  
    public:
      PrototypeAST(const std::string &Name, const std::string &Class, const std::string &Method,
                  std::vector<std::string> Args,
                  std::vector<std::string> Types,
                  bool IsOperator = false, unsigned Prec = 0);
  
    Function *codegen();
    const std::string &getName() const; 
    const std::string &getClass() const; 
    const std::string &getMethod() const; 
  
    bool isUnaryOp() const; 
    bool isBinaryOp() const; 
  
    char getOperatorName() const; 
  
  
  
    unsigned getBinaryPrecedence() const; 
  };


