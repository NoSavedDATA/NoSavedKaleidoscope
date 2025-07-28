#pragma once

#include <string>
#include <vector>

#include "llvm/IR/Value.h"
#include "parser_struct.h"

using namespace llvm;

//===----------------------------------------------------------------------===//
// Abstract Syntax Tree (aka Parse Tree)
//===----------------------------------------------------------------------===//

/// ExprAST - Base class for all expression nodes.
class ExprAST {
  public:
    virtual ~ExprAST() = default;
    std::string Type = "None";
    std::string ReturnType = "None";
    std::string Name = "Unnamed";
    bool isSelf = false;
    bool isAttribute = false;
    std::string _pre_dot = "";
    bool isVec = false;
    bool isList = false;
    bool isVarLoad = false;
    bool SolverIncludeScope = true;
    bool NameSolveToLast = true;
  

  
  
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
  
    
    virtual void SetIsVec(bool); 
    virtual bool GetIsVec(); 
  
    virtual void SetIsList(bool); 
    virtual bool GetIsList(); 

    
  };
  
class IndexExprAST : public ExprAST {
  
  public:
    int Size;
    std::vector<std::unique_ptr<ExprAST>> Idxs;
    std::vector<std::unique_ptr<ExprAST>> Second_Idxs;
    bool IsSlice=false;

    IndexExprAST(std::vector<std::unique_ptr<ExprAST>>, std::vector<std::unique_ptr<ExprAST>>, bool);

    Value *codegen(Value *scope_struct) override;
    int size() {
      return Size;
    }
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
  
class IntExprAST : public ExprAST {
  int Val;
  public:
    IntExprAST(int Val); 
  Value *codegen(Value *scope_struct) override;
}; 

  
class StringExprAST : public ExprAST {
  std::string Val;

  public:
    StringExprAST(std::string Val); 

  Value *codegen(Value *scope_struct) override;
};
  


class NullPtrExprAST : public ExprAST {

  public:
    NullPtrExprAST(); 

  Value *codegen(Value *scope_struct) override;
};

 


class VariableListExprAST : public ExprAST {
  public:
    std::vector<std::unique_ptr<ExprAST>> ExprList;
    VariableListExprAST(std::vector<std::unique_ptr<ExprAST>> ExprList); 

  Value *codegen(Value *scope_struct) override;
};

  
/// VariableExprAST - Expression class for referencing a variable, like "a".
class VariableExprAST : public ExprAST {

  public:
    std::unique_ptr<ExprAST> NameSolver;
    std::string Name;
    Parser_Struct parser_struct;
    VariableExprAST(std::unique_ptr<ExprAST> NameSolver, std::string Type, const std::string &Name, Parser_Struct parser_struct);

    Value *codegen(Value *scope_struct) override;
    const std::string &getName() const; 
    std::string GetName() override; 
};


class VecIdxExprAST : public ExprAST {
  
  public:
    std::unique_ptr<ExprAST> Loaded_Var;
    std::unique_ptr<IndexExprAST> Idx;

    VecIdxExprAST(std::unique_ptr<ExprAST> Loaded_Var, std::unique_ptr<IndexExprAST> Idx, std::string Type);

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
  Parser_Struct parser_struct;
  std::unique_ptr<ExprAST> Init;
  int Size;

  ObjectExprAST(
      Parser_Struct parser_struct,
      std::vector<std::pair<std::string, std::unique_ptr<ExprAST>>> VarNames,
      std::string Type,
      std::unique_ptr<ExprAST> Init, int Size);

  Value *codegen(Value *scope_struct) override;
};
  
  
   
  
  
  
class DataExprAST : public VarExprAST {
  public:
    std::vector<std::unique_ptr<ExprAST>> Notes;
    Parser_Struct parser_struct;

    DataExprAST(
      Parser_Struct,
      std::vector<std::pair<std::string, std::unique_ptr<ExprAST>>> VarNames,
      std::string Type,
      std::vector<std::unique_ptr<ExprAST>> Notes);

  Value *codegen(Value *scope_struct) override;
};


  
class LibImportExprAST : public ExprAST {
  public:
    std::string LibName;
    bool IsDefault;

  LibImportExprAST(std::string, bool); 

  Value *codegen(Value *) override;
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
  Parser_Struct parser_struct;

public:
  BinaryExprAST(char Op, std::string Elements, std::string Operation, std::unique_ptr<ExprAST> LHS,
                std::unique_ptr<ExprAST> RHS, Parser_Struct);

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
  
  
class NameableExprAST : public ExprAST {
  public:
  std::vector<std::string> Expr_String = {};
  std::unique_ptr<NameableExprAST> Inner_Expr;
  std::string Name="";
  bool End_of_Recursion=false, skip=false, IsLeaf=true, Load_Last=true, From_Self=false;
  int height=1;
  NameableExprAST();

  Value *codegen(Value *scope_struct) override;
};


class EmptyStrExprAST : public NameableExprAST {
  
  public:
    EmptyStrExprAST();
    Value *codegen(Value *scope_struct) override;
};


class SelfExprAST : public NameableExprAST {
  
  public:
    SelfExprAST();
    Value *codegen(Value *scope_struct) override;
};


class NestedStrExprAST : public NameableExprAST {
  Parser_Struct parser_struct;
  
  public:
    NestedStrExprAST(std::unique_ptr<NameableExprAST>, std::string, Parser_Struct);
    Value *codegen(Value *scope_struct) override;
};
  

class NestedVectorIdxExprAST : public NameableExprAST {
  Parser_Struct parser_struct;
  
  public:
    std::unique_ptr<IndexExprAST> Idx;
    NestedVectorIdxExprAST(std::unique_ptr<NameableExprAST>, std::string, Parser_Struct, std::unique_ptr<IndexExprAST> Idx, std::string);
    Value *codegen(Value *scope_struct) override;
};


void Print_Names_Str(std::vector<std::string>);

class NestedVariableExprAST : public ExprAST {
  
  public:
    Parser_Struct parser_struct;
    std::unique_ptr<NameableExprAST> Inner_Expr;
    bool Load_Val = true;
    
    NestedVariableExprAST(std::unique_ptr<NameableExprAST>, Parser_Struct, std::string);
    Value *codegen(Value *scope_struct) override;
};

class NestedCallExprAST : public ExprAST {
  std::unique_ptr<NameableExprAST> Inner_Expr;
  std::string Callee;
  std::vector<std::unique_ptr<ExprAST>> Args;
  Parser_Struct parser_struct;

  public:
    NestedCallExprAST(std::unique_ptr<NameableExprAST> Inner_Expr, std::string Callee, Parser_Struct parser_struct,
                            std::vector<std::unique_ptr<ExprAST>> Args);
    Value *codegen(Value *scope_struct) override;
};
  
  
/// CallExprAST - Expression class for function calls.
class CallExprAST : public ExprAST {
  std::vector<std::string> Expr_String;
  std::string Callee;
  std::vector<std::unique_ptr<ExprAST>> Args;
  std::string Class;
  std::string Name;
  std::string PreDot;
  std::string LoadOf;
  std::string Scope_Random_Str;
  std::string Load_Type="none";
  bool IsVarForward;
  std::string CalleeOverride;
  std::unique_ptr<ExprAST> NameSolver;
  Parser_Struct parser_struct;

  public:
    CallExprAST(std::unique_ptr<ExprAST> NameSolver,
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
              );

  Value *codegen(Value *scope_struct) override;
};


class ChainCallExprAST : public ExprAST {

  std::string Call_Of;
  
  std::vector<std::unique_ptr<ExprAST>> Args;
  std::unique_ptr<ExprAST> Inner_Call;

  public:
    ChainCallExprAST(const std::string &Call_Of,
                    std::vector<std::unique_ptr<ExprAST>> Args,
                    std::unique_ptr<ExprAST> Inner_Call);

  Value *codegen(Value *scope_struct) override;
};
  

class RetExprAST : public ExprAST {

  public:
    std::vector<std::unique_ptr<ExprAST>> Vars;
    
    RetExprAST(std::vector<std::unique_ptr<ExprAST>> Vars);

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
  Parser_Struct parser_struct;

  public:
    ForExprAST(const std::string &VarName, std::unique_ptr<ExprAST> Start,
              std::unique_ptr<ExprAST> End, std::unique_ptr<ExprAST> Step,
              std::vector<std::unique_ptr<ExprAST>> Body, Parser_Struct);

  Value *codegen(Value *scope_struct) override;
};

/// ForExprAST - Expression class for for.
class ForEachExprAST : public ExprAST {
  std::string VarName, VecType;
  std::unique_ptr<ExprAST> Vec;
  std::vector<std::unique_ptr<ExprAST>> Body;
  Parser_Struct parser_struct;

  public:
    ForEachExprAST(const std::string &VarName, std::unique_ptr<ExprAST> Vec,
              std::vector<std::unique_ptr<ExprAST>> Body, Parser_Struct, std::string Type, std::string VecType);

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
  Parser_Struct parser_struct;
  std::vector<std::unique_ptr<ExprAST>> Body;

  public:
    AsyncExprAST(std::vector<std::unique_ptr<ExprAST>> Body, Parser_Struct parser_struct);

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


class AsyncsExprAST : public ExprAST {
  Parser_Struct parser_struct;
  std::vector<std::unique_ptr<ExprAST>> Body;
  int AsyncsCount;

  public:
    AsyncsExprAST(std::vector<std::unique_ptr<ExprAST>> Body, int AsyncsCount, Parser_Struct parser_struct);

  Value* codegen(Value *scope_struct) override;
};
  


class IncThreadIdExprAST : public ExprAST {
  public:
    IncThreadIdExprAST();
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


class SplitParallelExprAST : public ExprAST {
  std::unique_ptr<ExprAST> Inner_Vec;
  public:
    SplitParallelExprAST(std::unique_ptr<ExprAST> Inner_Vec);
  Value* codegen(Value *scope_struct) override;
};

class SplitStridedParallelExprAST : public ExprAST {
  std::unique_ptr<ExprAST> Inner_Vec;
  public:
    SplitStridedParallelExprAST(std::unique_ptr<ExprAST> Inner_Vec);
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
  
    std::string Name, Return_Type, Class, Method;
  
    std::vector<std::string> Args;
    std::vector<std::string> Types;
    bool IsOperator;
    unsigned Precedence; // Precedence if a binary op.
  
    public:
      PrototypeAST(const std::string &Name, const std::string &Return_Type, const std::string &Class, const std::string &Method,
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


