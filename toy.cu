#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetMachine.h"
#include <algorithm>
#include <cassert>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <memory>
#include <string>
#include <iostream>
#include <numeric>
#include <string>
#include <utility>
#include <vector>
#include <iomanip>
#include <math.h>
#include <fenv.h>
#include <tuple>


#include <stdio.h>
#include <stdlib.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cublasLt.h>
#include <omp.h>


#include "include/KaleidoscopeJIT.h"
#include "include/cu_commons.h"


static cublasHandle_t cublas_handle;
static cublasLtHandle_t cublaslt_handle;

// cuBLAS workspace. Hardcoding to 32MiB but only Hopper needs 32, for others 4 is OK
static size_t cublaslt_workspace_size = 32 * 1024 * 1024; // 32 MB
static void* cublaslt_workspace = NULL;
static cublasComputeType_t cublas_compute_type;


cublasComputeType_t cublas_compute = CUBLAS_COMPUTE_32F;

#define CUBLAS_LOWP CUDA_R_32F
#define PRECISION_MODE PRECISION_FP32



using namespace llvm;
using namespace llvm::orc;

#if __CUDA_ARCH__ == 800 || __CUDA_ARCH__ >= 900
#define MAX_1024_THREADS_BLOCKS 2
#else
#define MAX_1024_THREADS_BLOCKS 1
#endif

// Error Colors

// \033[0m default
// \033[31m red
// \033[33m yellow
// \033[95m purple


//===----------------------------------------------------------------------===//
// Lexer
//===----------------------------------------------------------------------===//


// The lexer returns tokens [0-255] if it is an unknown character, otherwise one
// of these for known things.
enum Token {
  tok_eof = -1,

  // functions/classes
  tok_def = -2,
  tok_class = -77,
  tok_self = -78,
  tok_class_attr = -79,
  tok_extern = -3,

  // primary
  tok_identifier = -4,
  tok_number = -5,
  tok_str = -40,

  // control
  tok_if = -6,
  tok_then = -7,
  tok_else = -8,
  tok_for = -9,
  tok_while = -10,
  tok_tab = 9,

  // operators
  tok_binary = -11,
  tok_unary = -12,


  tok_space = -14,

  
  // var definition
  tok_var = -15,
  tok_tensor = -16,
  tok_attr_var = -17,
  tok_attr_tensor = -18,

  // function ops
  tok_log = -30
};

static std::string IdentifierStr; // Filled in if tok_identifier
static float NumVal;             // Filled in if tok_number
int LineCounter;

/// get_token - Return the next token from standard input.
static int get_token() {
  static int LastChar = ' ';

  // Skip any whitespace and backspace.
  //while (LastChar==32 || LastChar==tok_tab)
  while (LastChar==32 || LastChar==tok_tab)
    LastChar = getchar();
  //while (isspace(LastChar))
    
  if (LastChar=='"')
  {

    LastChar = getchar();
    IdentifierStr = LastChar;

    bool name_ok=true;
    while (name_ok)
    {
      LastChar = getchar();
      
      if(LastChar!='"')
        IdentifierStr += LastChar;
      else
        name_ok = false;

    }
    LastChar = getchar();
    
    return tok_str;
  }

  if (isalpha(LastChar) || LastChar=='_') { // identifier: [a-zA-Z][a-zA-Z0-9]*
    IdentifierStr = LastChar;
    bool name_ok=true;
    while (name_ok)
    {
      LastChar = getchar();
      
      if(isalnum(LastChar) || LastChar=='_')
        IdentifierStr += LastChar;
      else
        name_ok = false;

      if (IdentifierStr == "tensor" && LastChar=='[')
      {
        LastChar = getchar();
        return tok_tensor;
      }
      if (LastChar=='.')
      {
        LastChar = getchar();
        if (IdentifierStr == "self")
          return tok_self;
        return tok_class_attr;
      }
    }

    if (IdentifierStr == "def")
      return tok_def;
    if (IdentifierStr == "class")
      return tok_class;
    if (IdentifierStr == "extern")
      return tok_extern;
    if (IdentifierStr == "if")
      return tok_if;
    if (IdentifierStr == "then")
      return tok_then;
    if (IdentifierStr == "else")
      return tok_else;
    if (IdentifierStr == "for")
      return tok_for;
    if (IdentifierStr == "while")
      return tok_while;
    if (IdentifierStr == "binary")
      return tok_binary;
    if (IdentifierStr == "unary")
      return tok_unary;
    if (IdentifierStr == "var")
      return tok_var;
    if (IdentifierStr == "log")
      return tok_log;
    
    return tok_identifier;
  }

  if (isdigit(LastChar) || LastChar == '.') { // Number: [-.]+[0-9.]+
    std::string NumStr;
    if (LastChar == '-') { // Check for optional minus sign
      NumStr += LastChar;
      LastChar = getchar();
    }
    do {
      NumStr += LastChar;
      LastChar = getchar();
    } while (isdigit(LastChar) || LastChar == '.');

    NumVal = strtod(NumStr.c_str(), nullptr);
    return tok_number;
  }

  if (LastChar == '#') {
    // Comment until end of line.
    do
      LastChar = getchar();
    while (LastChar != EOF && LastChar != '\n' && LastChar != '\r');

    if (LastChar != EOF)
      return get_token();
  }

  // Check for end of file.  Don't eat the EOF.
  if (LastChar == EOF)
    return tok_eof;

  // Otherwise, just return the character as its ascii value.
  int ThisChar = LastChar;
  LastChar = getchar();
  int otherChar = LastChar;


  
  if(ThisChar==10)
  {
    LineCounter += 1;
    return tok_space;
  }

  if((ThisChar==47)&&(otherChar == 47)){
    LastChar = getchar();
    return 77; //
  }
  return ThisChar;
}

//===----------------------------------------------------------------------===//
// Abstract Syntax Tree (aka Parse Tree)
//===----------------------------------------------------------------------===//

/// ExprAST - Base class for all expression nodes.
class ExprAST {
public:
  virtual ~ExprAST() = default;
  std::vector<float> Dims = {-1.0f};
  std::string Type = "None";
  std::string Name = "Unnamed";
  std::string isSelf = "false";

  virtual Value *codegen() = 0;
  virtual void SetType(std::string Type) {
    this->Type=Type;
  }
  virtual std::string GetType() {
    return Type;
  }
  virtual void SetSelf(std::string Self) {
    this->isSelf=Self;
  }
  virtual std::string GetSelf() {
    return isSelf;
  }
  virtual std::string GetName() {
    return Name;
  }
  virtual void SetName(std::string Name) {
    this->Name=Name;
  }
  virtual std::vector<float> GetDims() {
    return Dims;
  }
  virtual void SetDims(std::vector<float> Dims) {
    this->Dims=Dims;
  }
};

/// NumberExprAST - Expression class for numeric literals like "1.0".
class NumberExprAST : public ExprAST {
  float Val;

  public:
    NumberExprAST(float Val) : Val(Val) {} //{std::cout << "number created";}
  std::string Type = "num";

  Value *codegen() override;
};



class StringExprAST : public ExprAST {
  std::string Val;

  public:
    StringExprAST(std::string Val) : Val(Val) {} //{std::cout << "number created";}
  std::string Type = "str";

  Value *codegen() override;
};



/// VariableExprAST - Expression class for referencing a variable, like "a".
class VariableExprAST : public ExprAST {
  std::string Name;

  public:
    VariableExprAST(const std::string &Name) : Name(Name) {}

    Value *codegen() override;
    const std::string &getName() const { return Name; }
    std::string GetName() override {
    return Name;
  }
};


/// VarExprAST - Expression class for var/in
class VarExprAST : public ExprAST {

  public:
    std::vector<std::pair<std::string, std::unique_ptr<ExprAST>>> VarNames;
    std::unique_ptr<ExprAST> Body;
    std::string Type;
    VarExprAST(
        std::vector<std::pair<std::string, std::unique_ptr<ExprAST>>> VarNames,
        std::unique_ptr<ExprAST> Body,
        std::string Type)
        : VarNames(std::move(VarNames)), Body(std::move(Body)), Type(Type) {}

  Value *codegen() override;
};

class TensorExprAST : public VarExprAST {
  public:
    std::vector<std::unique_ptr<ExprAST>> V_Dims;
    std::string TensorInit;

    TensorExprAST(
      std::vector<std::pair<std::string, std::unique_ptr<ExprAST>>> VarNames,
      std::unique_ptr<ExprAST> Body,
      std::string Type,
      std::vector<std::unique_ptr<ExprAST>> V_Dims,
      const std::string &TensorInit)
      : VarExprAST(std::move(VarNames), std::move(Body), std::move(Type)),
                   V_Dims(std::move(V_Dims)), TensorInit(TensorInit) {}

  Value *codegen() override;
};

class LogExprAST : public ExprAST {
  std::string Name;

  public:
    LogExprAST(const std::string &Name) : Name(Name) {}

    Value *codegen() override;
    std::string GetName() override {
      return Name;
    }
};

class LossBackwardExprAST : public ExprAST {
  public:
    LossBackwardExprAST() {}
    Value *codegen() override;
};


/// UnaryExprAST - Expression class for a unary operator.
class UnaryExprAST : public ExprAST {
  char Opcode;
  std::unique_ptr<ExprAST> Operand;

public:
  UnaryExprAST(char Opcode, std::unique_ptr<ExprAST> Operand)
      : Opcode(Opcode), Operand(std::move(Operand)) {}

  Value *codegen() override;
};



/// BinaryExprAST - Expression class for a binary operator.
class BinaryExprAST : public ExprAST {
  char Op;
  std::unique_ptr<ExprAST> LHS, RHS;

public:
  BinaryExprAST(char Op, std::unique_ptr<ExprAST> LHS,
                std::unique_ptr<ExprAST> RHS)
      : Op(Op), LHS(std::move(LHS)), RHS(std::move(RHS)) {}

  Value *codegen() override;
};


class BinaryTensorScalarExprAST : public ExprAST {
  char Op;
  std::unique_ptr<ExprAST> LHS, RHS;

public:
  BinaryTensorScalarExprAST(char Op, std::unique_ptr<ExprAST> LHS,
                std::unique_ptr<ExprAST> RHS)
      : Op(Op), LHS(std::move(LHS)), RHS(std::move(RHS)) {}

  Value *codegen() override;
};


class BinaryTensorTensorExprAST : public ExprAST {
  char Op;
  std::unique_ptr<ExprAST> LHS, RHS;

public:
  BinaryTensorTensorExprAST(char Op, std::unique_ptr<ExprAST> LHS,
                std::unique_ptr<ExprAST> RHS)
      : Op(Op), LHS(std::move(LHS)), RHS(std::move(RHS)) {}

  Value *codegen() override;
};


/// CallExprAST - Expression class for function calls.
class CallExprAST : public ExprAST {
  std::string Callee;
  std::vector<std::unique_ptr<ExprAST>> Args;
  std::string Class;
  std::string Pre_dot;

  public:
    CallExprAST(const std::string &Callee,
                std::vector<std::unique_ptr<ExprAST>> Args,
                const std::string &Class,
                const std::string &Pre_dot)
        : Callee(Callee), Args(std::move(Args)), Class(Class), Pre_dot(Pre_dot) {}

  Value *codegen() override;
};

/// IfExprAST - Expression class for if/then/else.
class IfExprAST : public ExprAST {
  std::unique_ptr<ExprAST> Cond, Then, Else;

  public:
    IfExprAST(std::unique_ptr<ExprAST> Cond, std::unique_ptr<ExprAST> Then,
              std::unique_ptr<ExprAST> Else)
        : Cond(std::move(Cond)), Then(std::move(Then)), Else(std::move(Else)) {}

  Value *codegen() override;
};

/// ForExprAST - Expression class for for.
class ForExprAST : public ExprAST {
  std::string VarName;
  std::unique_ptr<ExprAST> Start, End, Step, Body;

  public:
    ForExprAST(const std::string &VarName, std::unique_ptr<ExprAST> Start,
              std::unique_ptr<ExprAST> End, std::unique_ptr<ExprAST> Step,
              std::unique_ptr<ExprAST> Body)
        : VarName(VarName), Start(std::move(Start)), End(std::move(End)),
          Step(std::move(Step)), Body(std::move(Body)) {}

  Value *codegen() override;
};

/// WhileExprAST - Expression class for while.
class WhileExprAST : public ExprAST {
	std::unique_ptr<ExprAST> Cond, Body;

  public:
    WhileExprAST(std::unique_ptr<ExprAST> Cond, std::unique_ptr<ExprAST> Body)
      : Cond(std::move(Cond)), Body(std::move(Body)) {}

	Value* codegen() override;
};


/// PrototypeAST - This class represents the "prototype" for a function,
/// which captures its name, and its argument names (thus implicitly the number
/// of arguments the function takes), as well as if it is an operator.
class PrototypeAST {
  std::string Name;
  std::vector<std::string> Args;
  std::vector<std::string> Types;
  bool IsOperator;
  unsigned Precedence; // Precedence if a binary op.

  public:
    PrototypeAST(const std::string &Name, std::vector<std::string> Args,
                std::vector<std::string> Types,
                bool IsOperator = false, unsigned Prec = 0)
        : Name(Name), Args(std::move(Args)), Types(std::move(Types)),
          IsOperator(IsOperator), Precedence(Prec) {}

  Function *codegen();
  const std::string &getName() const { return Name; }

  bool isUnaryOp() const { return IsOperator && Args.size() == 1; }
  bool isBinaryOp() const { return IsOperator && Args.size() == 2; }

  char getOperatorName() const {
    assert(isUnaryOp() || isBinaryOp());
    return Name[Name.size() - 1];
  }

  unsigned getBinaryPrecedence() const { return Precedence; }
};


class ClassAST : public ExprAST {
  std::vector<std::unique_ptr<FunctionAST>> Functions;

  public:
    ClassAST(std::vector<std::unique_ptr<FunctionAST>> Functions)
        : Functions(std::move(Functions)) {}
  
  const PrototypeAST& getProto(int i) const;
  const std::string& getName(int i) const;
  
  Value *codegen();
};

//===----------------------------------------------------------------------===//
// Parser
//===----------------------------------------------------------------------===//

/// CurTok/getNextToken - Provide a simple token buffer.  CurTok is the current
/// token the parser is looking at.  getNextToken reads another token from the
/// lexer and updates CurTok with its results.
static int CurTok;
static int getNextToken() { return CurTok = get_token(); }

/// BinopPrecedence - This holds the precedence for each binary operator that is
/// defined.
static std::map<char, int> BinopPrecedence;

/// get_tokenPrecedence - Get the precedence of the pending binary operator token.
static int get_tokenPrecedence() {
  if (CurTok==tok_space)
    return 1;

  if (!isascii(CurTok))
    return -1;

  // Make sure it's a declared binop.
  int TokPrec = BinopPrecedence[CurTok];
  if (TokPrec <= 0)
    return -1;
  return TokPrec;
}



/// LogError* - These are little helper functions for error handling.
//std::unique_ptr<ExprAST> LogError(const char *Str) {
std::unique_ptr<ExprAST> LogErrorS(std::string Str) {
  //fprintf(stderr, "\033[31m Erro: \033[0m%s\n", Str);
  if (Str!=" ")
    std::cout << "\nLinha: " << LineCounter << "\n   \033[31m Erro: \033[0m " << Str << "\n\n";
  
  
  return nullptr;
}

std::unique_ptr<ExprAST> LogError(std::string Str) {
  //fprintf(stderr, "\033[31m Erro: \033[0m%s\n", Str);
  LogErrorS(Str);

  while(CurTok!=tok_space && CurTok!=';' && CurTok!=',' && CurTok!=')')
    getNextToken();
  
  return nullptr;
}

std::unique_ptr<ExprAST> LogErrorBreakLine(std::string Str) {
  //fprintf(stderr, "\033[31m Erro: \033[0m%s\n", Str);
  LogErrorS(Str);

  while(CurTok!=tok_space && CurTok!=';')
    getNextToken();
  
  return nullptr;
}


void LogWarning(const char *Str) {
  std::cout << "\nLinha: " << LineCounter << "\n   \033[33m Aviso: \033[0m " << Str << "\n\n";
}

// Modified LogError function with token parameter
std::unique_ptr<ExprAST> LogErrorT(int CurTok) {
  //char buf[100];
  //snprintf(buf, sizeof(buf), "token %d inesperado.", CurTok);
  //fprintf(stderr, "\033[31mErro: \033[0m%s\n", buf);
  std::cout << "\nLinha: " << LineCounter << "\n   \033[31m Erro: \033[0mtoken " << IdentifierStr << " inesperado.\n\n";
  return nullptr;
}


std::unique_ptr<PrototypeAST> LogErrorP(const char *Str) {
  LogError(Str);
  while(CurTok!=tok_space && CurTok!=';')
    getNextToken();
  return nullptr;
}


std::unique_ptr<PrototypeAST> LogErrorP_to_comma(const char *Str) {
  LogError(Str);
  while(CurTok!=tok_space && CurTok!=';' && CurTok!=',' && CurTok!=')')
  {
    std::cout << "LogErrorP: " << IdentifierStr << "\n";
    
    getNextToken();
    }
  return nullptr;
}

Value *LogErrorV(const char *Str) {
  LogError(Str);
  return nullptr;
}

static std::unique_ptr<ExprAST> ParseExpression(int tabcount=0);

/// numberexpr ::= number
static std::unique_ptr<ExprAST> ParseNumberExpr(int tabcount=0) {
  auto Result = std::make_unique<NumberExprAST>(NumVal);
  getNextToken(); // consume the number
  return std::move(Result);
}

static std::unique_ptr<ExprAST> ParseStringExpr(int tabcount=0) {
  auto Result = std::make_unique<StringExprAST>(IdentifierStr);
  getNextToken(); // consume the number
  return std::move(Result);
}

/// parenexpr ::= '(' expression ')'
static std::unique_ptr<ExprAST> ParseParenExpr() {
  getNextToken(); // eat (.
  auto V = ParseExpression();
  if (!V)
    return nullptr;

  if (CurTok != ')')
    return LogError("Esperado ')' na expressão em paranteses");
  
  std::cout << "Close brackets\n";
  getNextToken(); // eat ).
  return V;
}

std::vector<std::string> tensorVars;



static std::vector<std::string> Classes;
static std::map<std::string, std::string> Object_toClass;


/// identifierexpr
///   ::= identifier
///   ::= identifier '(' expression* ')'
static std::unique_ptr<ExprAST> ParseIdentifierExpr(int tabcount=0) {
  
  for(int i=0; i<Classes.size(); i++)
    if(IdentifierStr==Classes[i]) 
    {
      getNextToken();
      std::cout << "Object name: " << IdentifierStr << " and Class: " << Classes[i]<< "\n";
      Object_toClass[IdentifierStr] = Classes[i];
      
      getNextToken();
      return std::move(std::make_unique<NumberExprAST>(0.0f));
    }

  std::string IdName = IdentifierStr;

  getNextToken(); // eat identifier.
  
  if (CurTok != '(') // Simple variable ref.
  {
    auto aux = std::make_unique<VariableExprAST>(IdName);
    if (std::find(tensorVars.begin(), tensorVars.end(), IdentifierStr) != tensorVars.end())
      aux->SetType("tensor");
    std::cout << "identifier type: " << aux->GetType() <<  "\n";
    return aux;
  }

  

  // Call.
  getNextToken(); // eat (
  std::vector<std::unique_ptr<ExprAST>> Args;
  if (CurTok != ')') {
    while (true) {
      std::cout << "Parse a class expr.\n";
      if (auto Arg = ParseExpression())
        Args.push_back(std::move(Arg));
      else
        return nullptr;
      

      if (CurTok == ')')
        break;

      if (CurTok != ',')
        return LogError("Esperado ')' ou ',' na lista de argumentos");
      getNextToken();
    }
  }

  // Eat the ')'.
  getNextToken();

  return std::make_unique<CallExprAST>(IdName, std::move(Args), "None", "None");
}

/// ifexpr ::= 'if' expression 'then' expression 'else' expression
static std::unique_ptr<ExprAST> ParseIfExpr(int tabcount=1) {
  
  //std::cout << tabcount << " " << CurTok << "token if atual\n";
  if(CurTok==tok_space)
    getNextToken();

  getNextToken(); // eat the if.
  //CurTok = '(';
  //std::cout << CurTok << "token if posterior\n";
  

  //std::cout << CurTok << " Cond token \n";
  // condition.
  auto Cond = ParseExpression(tabcount+1);
  if (!Cond)
    return nullptr;

  if(CurTok==tok_space)
    getNextToken();


  //std::cout << "If then token " << CurTok << "\n";
  auto Then = ParseExpression(tabcount+1);
  //std::cout << "Then finished \n";
  if (!Then)
  {
    //std::cout << "Then is null \n";
    return nullptr;
  }
  
  //std::cout << "If else token " << CurTok << "\n";
  
  if(CurTok==tok_space)
    getNextToken();


  if (CurTok != tok_else){
    auto Else = std::make_unique<NumberExprAST>(0);
    getNextToken();
    return std::make_unique<IfExprAST>(std::move(Cond), std::move(Then),
                                      std::move(Else));
  }
  else {
    getNextToken();

    auto Else = ParseExpression(tabcount+1);
    if (!Else)
      return nullptr;
    
    return std::make_unique<IfExprAST>(std::move(Cond), std::move(Then),
                                      std::move(Else));
  }
}

/// forexpr ::= 'for' identifier '=' expr ',' expr (',' expr)? 'in' expression
static std::unique_ptr<ExprAST> ParseForExpr() {
  getNextToken(); // eat the for.

  if (CurTok != tok_identifier)
    return LogError("identificador da variável de controle esperado depois do for.");

  std::string IdName = IdentifierStr;
  getNextToken(); // eat identifier.

  if (CurTok != '=')
    return LogError("Esperada atribuição do valor inicial do for.");
  getNextToken(); // eat '='.

  auto Start = ParseExpression(0);
  if (!Start)
    return nullptr;
  if (CurTok != ',')
    return LogError("Esperado ',' depois de atribuir valor inicial do for.");
  getNextToken();

  auto End = ParseExpression(0);
  if (!End)
    return nullptr;

  std::unique_ptr<ExprAST> Step = std::make_unique<NumberExprAST>(1.0);
  if (CurTok == ',') { // The step value is optional.
    getNextToken();
    auto aux = ParseExpression(0);
    if (aux)
      Step = std::move(aux);
  }

  
  auto Body = ParseExpression();
  if (!Body)
    return nullptr;

  return std::make_unique<ForExprAST>(IdName, std::move(Start), std::move(End),
                                       std::move(Step), std::move(Body));
}


/// whileexpr ::= 'while' identifier '=' expr ',' expr (',' expr)? 'in' expression
static std::unique_ptr<ExprAST> ParseWhileExpr() {
  getNextToken(); // eat the while.

  if (CurTok != tok_identifier)
    return LogError("Identificador da variável de controle esperado depois do while.");


  auto Cond = ParseExpression(0);
  if (!Cond)
    return nullptr;
  
  auto Body = ParseExpression();
  if (!Body)
    return nullptr;

  return std::make_unique<WhileExprAST>(std::move(Cond), std::move(Body));
}


/// varexpr ::= 'var' identifier ('=' expression)?
//                    (',' identifier ('=' expression)?)* 'in' expression
static std::unique_ptr<ExprAST> ParseVarExpr() {
  getNextToken(); // eat the var.
  

  // mem2reg is alloca-driven: it looks for allocas and if it can handle them, it promotes them. It DOES NOT APPLY TO GLOBAL variables or heap allocations.
  // mem2reg only promotes allocas whose uses are direct loads and stores. If the address of the stack object is passed to a function,
  //or if any funny pointer arithmetic is involved, the alloca will not be promoted.

  std::vector<std::pair<std::string, std::unique_ptr<ExprAST>>> VarNames;

  // At least one variable name is required.
  if (CurTok != tok_identifier)
    return LogError("Esperado identificador após var.");

  while (true) {
    std::string Name = IdentifierStr;
    getNextToken(); // eat identifier.

    // Read the optional initializer.
    std::unique_ptr<ExprAST> Init = nullptr;
    if (CurTok == '=') {
      getNextToken(); // eat the '='.

      Init = ParseExpression();
      if (!Init)
        return nullptr;
    }

    VarNames.push_back(std::make_pair(Name, std::move(Init)));

    // End of var list, exit loop.
    if (CurTok != ',')
      break;
    getNextToken(); // eat the ','.

    if (CurTok != tok_identifier)
      return LogError("Esperado um ou mais identificadores após var.");
  }

  auto Body = ParseExpression();
  if (!Body)
    return nullptr;

  return std::make_unique<VarExprAST>(std::move(VarNames), std::move(Body), "var");
}


static std::unique_ptr<ExprAST> ParseSelfExpr() {

  std::string pre_dot = IdentifierStr;
  std::string object_class;
  bool is_class_attr=false;

  //std::cout << "CLASS ATTR IS: " << IdentifierStr << "\n";

  if (CurTok!=tok_self)
  {
    is_class_attr = true;
    pre_dot="";

    
    while (CurTok==tok_class_attr)
    {
      object_class=IdentifierStr;
      pre_dot+=IdentifierStr;
      getNextToken();
    }
    
    //std::cout << "Search object method: " << IdentifierStr <<  "\n";
    if (Object_toClass.find(object_class) != Object_toClass.end())
    {
      //std::cout << "Found object to class for\n";
      object_class = Object_toClass[object_class]; 
    }
  } else
    getNextToken(); // eat object or self token.
  
  //std::cout << "post: " << IdentifierStr << " token: " << CurTok << "\n";
  std::cout << "Pre-dot: " << pre_dot << " Post-dot: " << IdentifierStr  << "\n";


  std::string IdName = IdentifierStr;

  getNextToken(); // eat identifier.

  
  if (pre_dot=="loss" and IdName=="backward")
  {
    if (CurTok != '(')
      LogErrorBreakLine("Precisa do '(' na chamada do backward");
    getNextToken();
    if (CurTok != ')')
      LogErrorBreakLine("Precisa do ')' na chamada do backward");
    getNextToken();
    return std::make_unique<LossBackwardExprAST>();
  }

  if (CurTok != '(') // Simple variable ref.
  {
    auto aux = std::make_unique<VariableExprAST>(IdName);
    if (std::find(tensorVars.begin(), tensorVars.end(), IdentifierStr) != tensorVars.end())
      aux->SetType("tensor");
    if (is_class_attr)
      aux->SetSelf(pre_dot);
    if (pre_dot=="self")
      aux->SetSelf("true");
    return aux;
  }

  // Call.
  getNextToken(); // eat (
  std::vector<std::unique_ptr<ExprAST>> Args;
  if (CurTok != ')') {
    while (true) {
      if (auto Arg = ParseExpression())
        Args.push_back(std::move(Arg));
      else
        return nullptr;

      if (CurTok == ')')
        break;

      if (CurTok != ',')
        return LogError("Esperado ')' ou ',' na lista de argumentos");
      getNextToken();
    }
  }

  // Eat the ')'.
  getNextToken();

  std::cout << "Call expr pre-dot: " << pre_dot << "\n";
  return std::make_unique<CallExprAST>(IdName, std::move(Args), object_class, pre_dot);
}


static std::unique_ptr<ExprAST> ParseTensorExpr() {
  // TODO: Allow finishing line with ";"
  getNextToken(); // eat the tensor.
  
  std::vector<std::unique_ptr<ExprAST>> dims;
  std::vector<std::pair<std::string, std::unique_ptr<ExprAST>>> VarNames;
  std::string init = "randu";
  //std::make_unique<NumberExprAST>(NumVal)
  
  while (true) {
    if (CurTok != tok_number && CurTok != tok_identifier && CurTok != tok_self)
      return LogError("Esperado número da dimensão do tensor.");
    
    if (CurTok==tok_number)
    {
      if (std::fmod(NumVal, 1.0) != 0)
        LogWarning("A dimensão do tensor precisa ser int. Não pode ser float ou double.");
    
      dims.push_back(std::make_unique<NumberExprAST>( (float)((int)round(NumVal)) ));
      getNextToken();
    } else if (CurTok==tok_identifier)
      if (IdentifierStr=="zeros" || IdentifierStr=="xavu" || IdentifierStr=="xavn" || IdentifierStr=="ones")
      {
        init = IdentifierStr;
        getNextToken();
      } else
        dims.push_back(std::move(ParseIdentifierExpr()));
    else {
      dims.push_back(std::move(ParseSelfExpr()));
    }

    
    if (CurTok != ',')
      break;
    getNextToken(); // eat the ','.
  }

  
  if (CurTok != ']')
    return LogError("Faltou fechar ].");
    getNextToken();


  std::string pre_dot = "false";
  if (CurTok == tok_self)
  {
    pre_dot = "true";
    getNextToken();
  }
  if (CurTok == tok_class_attr)
  {
    pre_dot = IdentifierStr;
    std::cout << "Obj attr tensor: " << pre_dot << "\n";
    getNextToken();
  }

  if (CurTok != tok_identifier)
    return LogError("Esperado identificador após var.");

  while (true) {
    std::string Name = IdentifierStr;
    tensorVars.push_back(IdentifierStr);
    getNextToken(); // eat identifier.

    // Read the optional initializer.
    std::unique_ptr<ExprAST> Init = nullptr;
    if (CurTok == '=') {
      getNextToken(); // eat the '='.

      Init = ParseExpression();
      if (!Init)
        return nullptr;
    }

    VarNames.push_back(std::make_pair(Name, std::move(Init)));

    // End of var list, exit loop.
    if (CurTok != ',')
      break;
    getNextToken(); // eat the ','.

    if (CurTok != tok_identifier)
      return LogError("Esperado um ou mais identificadores após var.");
  }

  auto Body = ParseExpression();
  if (!Body)
    return nullptr;

  auto aux = std::make_unique<TensorExprAST>(std::move(VarNames), std::move(Body), "tensor",
                                             std::move(dims), init);
  aux->SetSelf(pre_dot);
  return aux;
}



static std::unique_ptr<ExprAST> ParseLogExpr() {
  getNextToken(); // eat the log.
  

  if (CurTok != '(')
    return LogError("Esperado ( após a função log.");
  getNextToken();
  

  if (CurTok != tok_identifier)
    return LogError("Esperado tensor à função log.");
  
  std::string Name = IdentifierStr;
  getNextToken();
  

  if (CurTok != ')')
    return LogError("Esperado ) na função log.");
  getNextToken();

  auto aux = std::make_unique<LogExprAST>(std::move(Name));
  aux->SetType("tensor");
  return aux;
}



/// primary
///   ::= identifierexpr
///   ::= numberexpr
///   ::= parenexpr
///   ::= ifexpr
///   ::= forexpr
///   ::= varexpr
static std::unique_ptr<ExprAST> ParsePrimary(int tabcount=0) {
  while(CurTok==tok_tab)
    getNextToken();
  switch (CurTok) {
  default:
    //std::cout << CurTok << " token atual de erro esperando expressão\n";
    return LogErrorT(CurTok);
  case tok_identifier:
    return ParseIdentifierExpr(tabcount);
  case tok_class_attr:
    return ParseSelfExpr();
  case tok_self:
    return ParseSelfExpr();
  case tok_number:
    return ParseNumberExpr(tabcount);
  case tok_str:
    return ParseStringExpr();
  case '(':
    return ParseParenExpr();
  case tok_if:
    return ParseIfExpr();
  case tok_for:
    return ParseForExpr();
  case tok_while:
    return ParseWhileExpr();
  case tok_var:
    return ParseVarExpr();
  case tok_tensor:
    return ParseTensorExpr();
  case tok_log:
    return ParseLogExpr();
  case tok_tab:
    getNextToken();
    return ParsePrimary();
  case tok_space:
    getNextToken();
    return ParsePrimary();
  }
}

/// unary
///   ::= primary
///   ::= '!' unary
static std::unique_ptr<ExprAST> ParseUnary(int tabcount=0) {
  
  while((CurTok==tok_tab)||(CurTok==tok_space))
    getNextToken();
  // If the current token is not an operator, it must be a primary expr.
  
  //std::cout << "Unary current token " << CurTok << "\n";
  if (!isascii(CurTok) || CurTok == '(' || CurTok == ',')
  {
    //std::cout << "Returning, non-ascii found.\n";
    return ParsePrimary(tabcount);
  }
  
  
  // If this is a unary operator, read it.
  int Opc = CurTok;
  
  getNextToken();
  if (auto Operand = ParseUnary(tabcount))
    return std::make_unique<UnaryExprAST>(Opc, std::move(Operand));
  return nullptr;
}

/// binoprhs
///   ::= ('+' unary)*


static std::tuple<std::unique_ptr<ExprAST>, int> ParseBinOpRHS(int ExprPrec,
                                              std::unique_ptr<ExprAST> LHS,
                                              int tabcount=0) {
  
  // If this is a binop, find its precedence.
  int RhsTok = 0;
  int LhsTok = 0;

  int L_cuda = 0;
  int R_cuda = 0;

  std::string LName, RName;
  if (LHS->GetType()=="tensor")
    L_cuda = 1;

  while (true) {
    
    
    int TokPrec = get_tokenPrecedence();

    // If this is a binop that binds at least as tightly as the current binop,
    // consume it, otherwise we are done.
    

    if (TokPrec==BinopPrecedence[':'])
    {
      getNextToken();
      return std::make_tuple(std::move(LHS),L_cuda);
    }
    if (TokPrec < ExprPrec)
      return std::make_tuple(std::move(LHS),L_cuda);

    
    int BinOp = CurTok;

    if(CurTok==':')
    {
      getNextToken();
      return std::make_tuple(std::move(LHS),L_cuda);
    }

    if (CurTok==')')
      return std::make_tuple(std::move(LHS),L_cuda);

    
    getNextToken(); // eat binop
    if (CurTok==tok_number)
      RName = std::to_string(NumVal);
    else
      RName = IdentifierStr;

    // Get the Right Hand Side token

    int seen_tabs = 0;
    
    
    while(CurTok==tok_tab)
    {
      getNextToken();
      seen_tabs+=1;
    }

    

    
    RhsTok = CurTok;


    //std::cout << "Before RHS " << LhsTok << " " << BinOp << " " << CurTok << " " << seen_tabs << "/" << tabcount << " " << RName << " \n";

    
    if((BinOp==tok_space) && (!( CurTok==tok_identifier || CurTok==tok_number || CurTok==tok_self || CurTok==tok_class_attr || CurTok==tok_var || CurTok==tok_tensor)))
    {
      
      std::cout << "SPACE WITHOUT NUMBER OR VAR " << CurTok << " " << IdentifierStr << "\n";
      return std::make_tuple(std::move(LHS),L_cuda);
    }
    
    


    auto RHS = ParseUnary(); // Returns an identifier, number or expression result
    if (RHS->GetType()=="tensor")
      R_cuda=1;
    
    /*
    if(BinOp==tok_space)
    {
      std::cout << "FOUND SPACE HEREEE\n\n";
      return RHS;
    }
    */
    if (!RHS)
    {
      //std::cout << "RETURNING NULL Parse Unary \n";
      return std::make_tuple(nullptr,0);
    }

    
    if ((CurTok==tok_space)&&(seen_tabs<tabcount)&&(seen_tabs>0))
    {
      //std::cout << "DIMNISHING IJFNASEJHFBEAIUYSBFESABHFGIYBUEASFBEIAUSBFYEASUIBFYAEUSB\n";
      //LHS = std::move(RHS); //RETORNA O LADO DIREITO COMO O PRÓPRIO ELSE
      
      //LHS = ParseBinOpRHS(TokPrec + 1, std::move(LHS), tabcount);

      //RHS = ParseBinOpRHS(TokPrec + 1, std::move(RHS), tabcount);
      //LHS = std::make_unique<BinaryExprAST>(tok_space, std::move(LHS), std::move(RHS));
      
      //LHS = std::make_unique<BinaryExprAST>(BinOp, std::move(LHS), std::move(RHS));
      
      //return LHS;// RETORNA A VARIÁVEL COM ERRO DE INDEX
      return std::make_tuple(std::move(RHS),R_cuda);
    } else {


      // If BinOp binds less tightly with RHS than the operator after RHS, let
      // the pending operator take RHS as its LHS.
      int NextPrec = get_tokenPrecedence();
        
      // || ((seen_tabs<tabcount)&&(seen_tabs>0))
      if (TokPrec < NextPrec){
        //std::cout << NextPrec << " Next Prec\n";
        
        auto tuple = ParseBinOpRHS(TokPrec + 1, std::move(RHS), tabcount);
        RHS = std::move(std::get<0>(tuple));
        R_cuda = std::get<1>(tuple);

        //std::cout << "Error after RHS parse \n";
        if (!RHS)
        {
          //std::cout << "RETURNING NULL Recursive Bin Op \n";
          return std::make_tuple(nullptr,0);
        }
      }

      
      //std::cout << LhsTok << " " << BinOp << " " << RhsTok << "\n" << CurTok <<  " " << RName << "\n\n";
      
      //if(BinOp==64) // @

      

      if (L_cuda==1 && R_cuda==0)
      {
        LHS = std::make_unique<BinaryTensorScalarExprAST>(BinOp,
                                                      std::move(LHS), std::move(RHS));
        
      }
      else if (L_cuda==0 && R_cuda==1)
      {
        std::cout << "Reverse LHS and RHS\n";
        //std::cout << "Bin op: " << BinOp << "\n";
        if (BinOp==47)
          return std::make_tuple(LogError("Divisão de escalar por tensor."),0);

        if (BinOp==45) // inversion of 1 - tensor
        {
          RHS = std::make_unique<BinaryTensorScalarExprAST>(42,
                                                    std::move(RHS),
                                                    std::move(std::make_unique<NumberExprAST>(-1.0f)));
                                                    //std::move(LHS)
                                                    
          LHS = std::make_unique<BinaryTensorScalarExprAST>(43,
                                                    std::move(RHS), std::move(LHS));
        } else
          LHS = std::make_unique<BinaryTensorScalarExprAST>(BinOp,
                                                    std::move(RHS), std::move(LHS));
        L_cuda=1;
        R_cuda=0;
      }
      else if (L_cuda==1 && R_cuda==1)
      { 
        LHS = std::make_unique<BinaryTensorTensorExprAST>(BinOp,
                                                      std::move(LHS), std::move(RHS));
        R_cuda=0;
      }
      else
        LHS = std::make_unique<BinaryExprAST>(BinOp, std::move(LHS), std::move(RHS));

      LhsTok = RhsTok;    
  }
}
}


/// expression
///   ::= unary binoprhs
///
static std::unique_ptr<ExprAST> ParseExpression(int tabcount) {
  //std::cout << "Parse Expression tabcount " << tabcount << "\n";
  //std::cout << "Parse Expression\n";
  
  auto LHS = ParseUnary(tabcount);
  if (!LHS)
    return nullptr;

  return std::get<0>(ParseBinOpRHS(0, std::move(LHS), tabcount));
}

/// prototype
///   ::= id '(' id* ')'
///   ::= binary LETTER number? (id, id)
///   ::= unary LETTER (id)
static std::unique_ptr<PrototypeAST> ParsePrototype(std::string ClassName="") {
  std::string FnName = ClassName;

  unsigned Kind = 0; // 0 = identifier, 1 = unary, 2 = binary.
  unsigned BinaryPrecedence = 30;

  switch (CurTok) {
  default:
    return LogErrorP("Esperado nome da função no protótipo");
  case tok_identifier:
    FnName += IdentifierStr;
    Kind = 0;
    getNextToken();
    break;
  case tok_unary:
    getNextToken();
    if (!isascii(CurTok))
      return LogErrorP("Esperado operador unário");
    FnName += "unary";
    FnName += (char)CurTok;
    Kind = 1;
    getNextToken();
    break;
  case tok_binary:
    getNextToken();
    if (!isascii(CurTok))
      return LogErrorP("Esperado operador binário");
    FnName += "binary";
    FnName += (char)CurTok;
    Kind = 2;
    getNextToken();

    // Read the precedence if present.
    if (CurTok == tok_number) {
      if (NumVal < 1 || NumVal > 100)
        return LogErrorP("Precedência inválida: deve ser entre 1 e 100");
      BinaryPrecedence = (unsigned)NumVal;
      getNextToken();
    }
    break;
  }

  if (CurTok != '(')
    return LogErrorP("Esperado '(' no protótipo");

  getNextToken();

  bool is_tensor=false;
  std::vector<std::string> ArgNames, ArgTypes;
  while (CurTok != ')')
  {
    ArgTypes.push_back(IdentifierStr);
    if (IdentifierStr=="t")
      is_tensor=true;
    if (IdentifierStr!="t" && IdentifierStr!="f")
      LogErrorP_to_comma("Tipo da variável no protótipo precisa ser t ou f.");
    else {
      getNextToken();

      ArgNames.push_back(IdentifierStr);
      if (is_tensor)
        tensorVars.push_back(IdentifierStr);
      
      getNextToken();
    }
    is_tensor=false;


    if (CurTok == ')')
        break;
      
    if (CurTok != ',')
    {
      std::cout << "comma Cur Tok " << IdentifierStr << "\n";
      return LogErrorP("Esperado ')' ou ',' na lista de argumentos do protótipo.");
    }
    getNextToken();
  }

  // success.
  getNextToken(); // eat ')'.

  // Verify right number of names for operator.
  if (Kind && ArgNames.size() != Kind)
    return LogErrorP("Número inválido de operandos para o operador");

  return std::make_unique<PrototypeAST>(FnName, ArgNames, ArgTypes, Kind != 0,
                                         BinaryPrecedence);
}


/// definition ::= 'def' prototype expression
static std::unique_ptr<FunctionAST> ParseDefinition(std::string ClassName="") {
  getNextToken(); // eat def.
  auto Proto = ParsePrototype(ClassName);
  if (!Proto)
    return nullptr;

  if (auto E = ParseExpression())
    return std::make_unique<FunctionAST>(std::move(Proto), std::move(E));
  return nullptr;
}

/// toplevelexpr ::= expression
static std::unique_ptr<FunctionAST> ParseTopLevelExpr() {
  //std::cout << "Top Level Expression\n";
  if (auto E = ParseExpression()) {
    // Make an anonymous proto.
    auto Proto = std::make_unique<PrototypeAST>("__anon_expr",
                                                std::vector<std::string>(),
                                                std::vector<std::string>());
    return std::make_unique<FunctionAST>(std::move(Proto), std::move(E));
  }
  return nullptr;
}

/// external ::= 'extern' prototype
static std::unique_ptr<PrototypeAST> ParseExtern() {
  getNextToken(); // eat extern.
  return ParsePrototype();
}







//===----------------------------------------------------------------------===//
// Code Generation
//===----------------------------------------------------------------------===//

static std::unique_ptr<KaleidoscopeJIT> TheJIT;
static std::unique_ptr<LLVMContext> TheContext;
static std::unique_ptr<LLVMContext> GlobalContext = std::make_unique<LLVMContext>();


static std::unique_ptr<IRBuilder<>> Builder;
static std::unique_ptr<Module> TheModule;


static std::map<std::string, std::unique_ptr<PrototypeAST>> FunctionProtos;
static ExitOnError ExitOnErr;


// Vars
static std::map<std::string, AllocaInst *> NamedValues;
static std::map<std::string, Value *> NamedClassValues;
static std::map<std::string, float> StoredValues;


// Tensors
static std::map<std::string, float *> NamedTensors;
static std::map<std::string, std::vector<float>> NamedDims;


// Optimizer
static std::map<std::string, float *> NamedParamGrads;
static std::map<std::string, float *> NamedV;
static std::map<std::string, float *> NamedM;
int optimizer_timestep = 0;



// Handle Class self with phantom argument
std::string FirstArg;


bool ends_with(std::string str_input, std::string str_end)
{
  return str_input.size() >= str_end.size() && str_input.compare(str_input.size() - str_end.size(), str_end.size(), str_end) == 0;
}

static std::unique_ptr<ExprAST> ParseClass() {
  getNextToken(); // eat class.

  if (CurTok != tok_identifier)
    return LogError("Esperado o nome da Classe.");
  std::string Name = IdentifierStr;

  Classes.push_back(Name);

  getNextToken();

  while(CurTok==tok_tab || CurTok==tok_space)
    getNextToken();
  

  if (CurTok!=tok_def)
    return LogError("Definição de uma Classe requer suas respectivas funções.");

  int i=0;
  while(CurTok==tok_def)
  {
    
    auto Func = ParseDefinition(Name);
    if (!Func)
      return nullptr;
      //return LogError("Falha no parsing da função da Classe.");
    if (!ends_with(Func->getProto().getName(),"__init__") && i==0)
      return LogError("Classe requer método init");
    
    //std::cout << "THE FUNCTION WAS CREATED AS: " << Func->getProto().getName() << "\n";

    FunctionProtos[Func->getProto().getName()] =
      std::make_unique<PrototypeAST>(Func->getProto());
    ExitOnErr(TheJIT->addAST(std::move(Func)));
    if(CurTok==';')
      getNextToken();
    while(CurTok==tok_space || CurTok==tok_tab)
      getNextToken();

    i+=1;
  }
  //if (auto E = ParseExpression())
  //  return std::make_unique<ClassAST>(std::move(Proto), std::move(E));
  return nullptr;
}






int dimsProd(std::vector<float> dims)
{
  float aux=1;
  for (int i = 0; i < dims.size(); i++)
    aux = aux*dims[i];
  return (int)aux;
}


int resultingDimsProdOnMult(std::vector<float> Ldims, std::vector<float> Rdims)
{
  float aux=1;
  for (int i = 0; i < Ldims.size()-1; i++)
    aux = aux * Ldims[i];
  aux = aux * Rdims[0];
  return (int)aux;
}

std::vector<float> format_LinearLayer_Dims(std::vector<float> dims)
{
  std::vector<float> new_dims;
  int aux=1;
  for (int i = 0; i < dims.size()-1; i++)
    aux *= dims[i];
  new_dims.push_back(aux);
  new_dims.push_back(dims[dims.size()-1]);
  return new_dims;
}

std::vector<float> newDimsOnMult(std::vector<float> Ldims, std::vector<float> Rdims)
{
  std::vector<float> new_dims;
  if (Ldims[Ldims.size()-1]!=Rdims[Rdims.size()-1])
  {
    LogError("A última dimensão dos tensors multiplicados precisa ser igual.");
    return new_dims; 
  }
  for (int i = 0; i < Ldims.size()-1; i++)
    new_dims.push_back(Ldims[i]);
  new_dims.push_back(Rdims[0]);
  
  return new_dims;
}

void PrintDims(std::vector<float> dims)
{
  std::cout << "dims: [";
  for (int i=0; i<dims.size();i++)
  {
    std::cout << (int)dims[i];
    if (i==dims.size()-1)
      std::cout << "]";
    else
      std::cout << ", ";
  }
  std::cout  << "\n";
}

extern "C" float PrintTensor(char* tensorName){
  

  std::vector<float> dims = NamedDims[tensorName];
  int arr_size = dimsProd(dims);


  float *tensor_cuda = NamedTensors[tensorName];
  float *tensor = new float[arr_size];
  //std::cout << "Printing Tensor " << arr_size << "\n";
  
  cudaDeviceSynchronize();
  cudaCheck(cudaMemcpy(tensor, tensor_cuda, arr_size*sizeof(float), cudaMemcpyDeviceToHost));


  std::cout << "\nTensor \033[95m" << tensorName << "\033[0m:\n";
  PrintDims(dims);
  std::cout << "\n";
  std::vector<float> ends;


  for (int i = 0; i < dims.size(); i++) {
    int prod=1;
    for (int j = 0; j <= i; j++)
      prod = prod*dims[dims.size()-1-j];
    ends.push_back(prod);
  }


  int line = 1;
  bool line_changed = true;
  for (int i = 0; i < arr_size; i++) {

    int to_prints = 0;

    for (int e = 0; e < ends.size(); e++)
    {
      if (fmod((arr_size-i),(int)ends[e]) == 0.0f)
        to_prints+=1;
    }

    if(to_prints>0)
    {
      for (int j=0; j<(dims.size()-to_prints); j++)
        std::cout << " ";
        
      for (int j=0; j<to_prints; j++)
        std::cout << "[";
    }
    

    //std::cout << "LAST SIZE " << dims[dims.size()-1] << " Mod: " << fmod(i, 1+dims[dims.size()-1]) << "\n";
    int precision;
    if (tensor[i]>=0)
      precision=4;
    else
      precision=3;
    std::cout << std::fixed  << std::setprecision(precision) << tensor[i];


    for (int e = 0; e < ends.size(); e++)
      if (fmod((i+1),(int)ends[e]) == 0.0f)
        std::cout << "]";
    

    if (i!=(arr_size-1))
    {
      if (fmod(i+1, dims[dims.size()-1]) == 0.0f)
      {
        line+=1;
        line_changed=true;
        std::cout << "\n";
      }
      else
        std::cout << "  ";
    }

    if(fmod(i+1, ends[1]) == 0.0f)
      std::cout << "\n";


  }
  std::cout << "\n";

  return 0;
}

float PrintTensorF(float *cuda_tensor, int d1, int d2){
  

  std::vector<float> dims;
  dims.push_back(d1);
  dims.push_back(d2);

  int arr_size = dimsProd(dims);


  float *tensor = new float[arr_size];
  //std::cout << "Printing Tensor " << arr_size << "\n";
  
  cudaDeviceSynchronize();
  cudaCheck(cudaMemcpy(tensor, cuda_tensor, arr_size*sizeof(float), cudaMemcpyDeviceToHost));


  
  PrintDims(dims);
  std::cout << "\n";
  std::vector<float> ends;


  for (int i = 0; i < dims.size(); i++) {
    int prod=1;
    for (int j = 0; j <= i; j++)
      prod = prod*dims[dims.size()-1-j];
    ends.push_back(prod);
  }


  int line = 1;
  bool line_changed = true;
  for (int i = 0; i < arr_size; i++) {

    int to_prints = 0;

    for (int e = 0; e < ends.size(); e++)
    {
      if (fmod((arr_size-i),(int)ends[e]) == 0.0f)
        to_prints+=1;
    }

    if(to_prints>0)
    {
      for (int j=0; j<(dims.size()-to_prints); j++)
        std::cout << " ";
        
      for (int j=0; j<to_prints; j++)
        std::cout << "[";
    }
    

    //std::cout << "LAST SIZE " << dims[dims.size()-1] << " Mod: " << fmod(i, 1+dims[dims.size()-1]) << "\n";
    int precision;
    if (tensor[i]>=0)
      precision=4;
    else
      precision=3;
    std::cout << std::fixed  << std::setprecision(precision) << tensor[i];


    for (int e = 0; e < ends.size(); e++)
      if (fmod((i+1),(int)ends[e]) == 0.0f)
        std::cout << "]";
    

    if (i!=(arr_size-1))
    {
      if (fmod(i+1, dims[dims.size()-1]) == 0.0f)
      {
        line+=1;
        line_changed=true;
        std::cout << "\n";
      }
      else
        std::cout << "  ";
    }

    if(fmod(i+1, ends[1]) == 0.0f)
      std::cout << "\n";


  }
  std::cout << "\n";

  return 0;
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

/// CreateEntryBlockAlloca - Create an alloca instruction in the entry block of
/// the function.  This is used for mutable variables etc.
static AllocaInst *CreateEntryBlockAlloca(Function *TheFunction,
                                          StringRef VarName) {
  IRBuilder<> TmpB(&TheFunction->getEntryBlock(),
                   TheFunction->getEntryBlock().begin());
  return TmpB.CreateAlloca(Type::getFloatTy(*TheContext), nullptr, VarName);
}

Value *NumberExprAST::codegen() {
  //std::cout << "Codegen for Number: " << Val << "\n";
  return ConstantFP::get(*TheContext, APFloat(Val));
}

Value *StringExprAST::codegen() {
  SetName(Val);
  return Builder->CreateGlobalString(Val);
}


//===----------------------------------------------------------------------===//
// Tensor -- Scalar   Operations
//===----------------------------------------------------------------------===//


__global__ void vec_mult(float a, float* x, float* y) {
  y[threadIdx.x] = x[threadIdx.x] * a;
}
__global__ void vec_div(float a, float* x, float* y) {
  y[threadIdx.x] = x[threadIdx.x] / a;
}
__global__ void vec_add(float a, float* x, float* y) {
  y[threadIdx.x] = x[threadIdx.x] + a;
}
__global__ void vec_sub(float a, float* x, float* y) {
  y[threadIdx.x] = x[threadIdx.x] - a;
}
__global__ void vec_log(float* x, float* y) {
  y[threadIdx.x] = logf(x[threadIdx.x]);
}

//float currentCudaResult[100];
float *currentCudaResult;
std::vector<float> currentDims;
int used_cuda = 0;


extern "C" float CudaScalarMult(char *tensorName, float R, int _used_cuda) {
  
  float * device_x;
  

  if (_used_cuda==1)
    device_x = currentCudaResult;
  else
  {
    device_x = NamedTensors[tensorName];
    currentDims = NamedDims[tensorName];
  }

  int kDataLen = dimsProd(currentDims);


  float* device_y;
  cudaCheck(cudaMalloc(&device_y, kDataLen * sizeof(float)));


  // Launch the kernel.
  vec_mult<<<1, kDataLen>>>(R, device_x, device_y);

  currentCudaResult = device_y;

  return 0;
}


extern "C" float CudaScalarDiv(char *tensorName, float R, int _used_cuda) {
  
  float * device_x;
  

  if (_used_cuda==1)
    device_x = currentCudaResult;
  else
  {
    device_x = NamedTensors[tensorName];
    currentDims = NamedDims[tensorName];
  }

  int kDataLen = dimsProd(currentDims);


  float* device_y;
  cudaCheck(cudaMalloc(&device_y, kDataLen * sizeof(float)));


  // Launch the kernel.
  vec_div<<<1, kDataLen>>>(R, device_x, device_y);

  currentCudaResult = device_y;

  return 0;
}

extern "C" float CudaScalarAdd(char *tensorName, float R, int _used_cuda) {
  float * device_x;

  

  if (_used_cuda==1)
    device_x = currentCudaResult;
  else
  {
    device_x = NamedTensors[tensorName];
    currentDims = NamedDims[tensorName];
  }

  int kDataLen = dimsProd(currentDims);


  float* device_y;
  cudaCheck(cudaMalloc(&device_y, kDataLen * sizeof(float)));
  
  // Launch the kernel.
  vec_add<<<1, kDataLen>>>(R, device_x, device_y);
  
  currentCudaResult = device_y;


  return 0;
}

extern "C" float CudaScalarSub(char *tensorName, float R, int _used_cuda) {
  
  float * device_x;

  

  if (_used_cuda==1)
    device_x = currentCudaResult;
  else
  {
    device_x = NamedTensors[tensorName];
    currentDims = NamedDims[tensorName];
  }

  int kDataLen = dimsProd(currentDims);


  float* device_y;
  cudaCheck(cudaMalloc(&device_y, kDataLen * sizeof(float)));


  // Launch the kernel.
  vec_sub<<<1, kDataLen>>>(R, device_x, device_y);

  currentCudaResult = device_y;

  return 0;
}


extern "C" float logE(char *tensorName, int _used_cuda) {
  
  float * device_x;

  if (_used_cuda==1)
    device_x = currentCudaResult;
  else
  {
    device_x = NamedTensors[tensorName];
    currentDims = NamedDims[tensorName];
  }

  int kDataLen = dimsProd(currentDims);


  float* device_y;
  cudaCheck(cudaMalloc(&device_y, kDataLen * sizeof(float)));


  // Launch the kernel.
  vec_log<<<1, kDataLen>>>(device_x, device_y);

  currentCudaResult = device_y;

  return 0;
}


extern "C" float FirstArgOnDemand(char *pre_dot, int nested_function)
{
  if (nested_function)
    FirstArg = FirstArg+pre_dot;
  else
    FirstArg = pre_dot;
  return 0;
}

extern "C" float DimnishFirstArgOnDemand(char *pre_dot, int nested_function)
{
  if (nested_function)
    if(ends_with(FirstArg, pre_dot))
    {
      size_t pos = FirstArg.find(pre_dot);

      FirstArg.erase(pos, std::strlen(pre_dot));
    }
    
  
  return 0;
}


extern "C" char * ConcatStr(char *lc, char *rc)
{
  std::string l = lc;
  std::string r = rc;

  std::string result_str = l + r;
  char* result_cstr = new char[result_str.length() + 1]; // +1 for null terminator
  std::strcpy(result_cstr, result_str.c_str());
  
  return result_cstr;
}

extern "C" char * ConcatFirstArgToVarName(char *var_name)
{
  //std::cout << "\nConcatFirstArgToVarName: " << FirstArg << "\nVar name: " << var_name <<"\n\n";
  
  std::string l = var_name;

  std::string result_str = FirstArg + l;
  char* result_cstr = new char[result_str.length() + 1]; // +1 for null terminator
  std::strcpy(result_cstr, result_str.c_str());


  return result_cstr;
}


extern "C" float StoreOnDemand(char *object_var_name, float value){
  
//  std::cout << "StoreOnDemand: " << FirstArg << "." << object_var_name << " " << value << "\n";

  NamedClassValues[FirstArg + object_var_name] = ConstantFP::get(*GlobalContext, APFloat(value));
  return 0;
}


extern "C" float LoadOnDemand(char *object_var_name) {
  //std::cout << "LoadOnDemand var to load: " << object_var_name << "\n";
    
  Value * class_val = NamedClassValues[object_var_name];

  if (class_val) 
    return (float) cast<ConstantFP>(class_val)->getValueAPF().convertToFloat();
  else
    return 0;
}


bool seen_var_attr = false;
Value *VariableExprAST::codegen() {
  // Look this variable up in the function.

  //std::cout << "Now Loading Var "<< Name <<" to Context" << "  \n";


  Function *TheFunction = Builder->GetInsertBlock()->getParent();
  std::string functionName = TheFunction->getName().str();
  
  

  Value *var_name, *object_name, *object_var_name;
  Value * ret = ConstantFP::get(*TheContext, APFloat(0.0f));
  var_name = Builder->CreateGlobalString(Name);
  
  
      
  std::string pre_dot = GetSelf();
  if (pre_dot!="false")
  {
    // Gets from FirstArg if it is self
    if (pre_dot=="true")
      var_name = Builder->CreateCall(TheModule->getFunction("ConcatFirstArgToVarName"),
                                                      {var_name});
    // Gets from pre_dot if it is a class attribute
    else {
      object_name = Builder->CreateGlobalString(pre_dot);
      var_name = Builder->CreateGlobalString(Name);

      var_name = Builder->CreateCall(TheModule->getFunction("ConcatStr"),
                                                      {object_name, var_name});
    }
      for (const auto &entry : NamedClassValues)
        if (ends_with(entry.first, Name))
          return Builder->CreateCall(TheModule->getFunction("LoadOnDemand"),
                                                      {var_name});        
  }

  if (NamedValues.count(Name)>0) 
  {

    Value *V = NamedValues[Name];
    
    

    return Builder->CreateLoad(Type::getFloatTy(*GlobalContext), V, Name.c_str());

  } else {
    //std::cout << "Load Tensor " << Name << " Codegen.\n";
  

    if (!seen_var_attr)
    {

      Builder->CreateCall(TheModule->getFunction("PrintTensor"), {var_name});
    }
    
    // float_to_value
    return ret;
  }
}



extern "C" float toStoredValues(float Val, char * name_to_store)
{
  StoredValues[name_to_store] = Val;
  return 0;
}


extern "C" float temporaryCudaResult_Attr(char *tensorName)
{
  std::cout << "Attributing to tensor: " << tensorName << "\n";
  cudaCheck(cudaFree(NamedTensors[tensorName]));

  float * tensor = new float[4];
  cudaMemcpy(tensor, currentCudaResult, 4, cudaMemcpyDeviceToHost);

  NamedTensors[tensorName] = currentCudaResult;
  NamedDims[tensorName] = currentDims;

  return 0;
}





Value *BinaryTensorScalarExprAST::codegen() {

  Value *tensorName = Builder->CreateGlobalString(LHS->GetName());

  std::string pre_dot = LHS->GetSelf();
  if (pre_dot=="true")
    tensorName = Builder->CreateCall(TheModule->getFunction("ConcatFirstArgToVarName"),
                                                      {tensorName});
    // Gets from pre_dot if it is a class attribute
  else if (pre_dot!="false") {
    Value * object_name = Builder->CreateGlobalString(pre_dot);

    tensorName = Builder->CreateCall(TheModule->getFunction("ConcatStr"),
                                                      {object_name, tensorName});
  }



  // Special case '=' because we don't want to emit the LHS as an expression.
  if (Op == '=') {
    seen_var_attr=true;
    // Assignment requires the LHS to be an identifier.
    // This assume we're building without RTTI because LLVM builds that way by
    // default.  If you build LLVM with RTTI this can be changed to a
    // dynamic_cast for automatic error checking.
    VariableExprAST *LHSE = static_cast<VariableExprAST *>(LHS.get());
    if (!LHSE)
      return LogErrorV("Destino do '=' deve ser uma variável.");
    // Codegen the RHS.
    
    Value *Val = RHS->codegen();
    if (!Val)
      return nullptr;

    
    
    std::cout << "1 0 attr\n";
    /*
    float *Variable = NamedTensors[LHSE->getName()];
    if (!Variable)
      return LogErrorV("O nome do tensor/variável é desconhecido.");
    */
    
    if (used_cuda)
    {
      Function *temporaryCudaResult_AttrFn = TheModule->getFunction("temporaryCudaResult_Attr");
      Builder->CreateCall(temporaryCudaResult_AttrFn, {tensorName});        
      
      used_cuda=0;
    }
      
    
    seen_var_attr=false;
    return Val;
  }


  Value *L = LHS->codegen();
  Value *R = RHS->codegen();
  
  if (!L || !R)
    return nullptr;


  Function *CudaFn;

  std::cout << "\nTensorScalar, LHS is self: " << LHS->GetSelf() << "\n";
  Function *TheFunction = Builder->GetInsertBlock()->getParent();
  std::string functionName = TheFunction->getName().str();
  std::cout << "Fname: " << functionName << "\n\n";
  
  

  Value *used_cuda_aux = ConstantInt::get(Type::getInt32Ty(*TheContext), used_cuda);
  used_cuda = 1;

  switch (Op)
  {
  case '*':
    CudaFn = TheModule->getFunction("CudaScalarMult");
    return Builder->CreateCall(CudaFn, {tensorName, R, used_cuda_aux}, "cudascalarmult");
  case '/':
    CudaFn = TheModule->getFunction("CudaScalarDiv");
    return Builder->CreateCall(CudaFn, {tensorName, R, used_cuda_aux}, "cudascalardiv");
  case '+':
    CudaFn = TheModule->getFunction("CudaScalarAdd");
    return Builder->CreateCall(CudaFn, {tensorName, R, used_cuda_aux}, "cudascalaradd");
  case '-':
    CudaFn = TheModule->getFunction("CudaScalarSub");
    return Builder->CreateCall(CudaFn, {tensorName, R, used_cuda_aux}, "cudascalarsub");
  case ':':
    return L;
  case tok_space:
    return R;
  default:
    break;
  }
  

  // If it wasn't a builtin binary operator, it must be a user defined one. Emit
  // a call to it.
  Function *F = getFunction(std::string("binary") + Op);
  assert(F && "Operator not found.");

  Value *Ops[] = {L, R};
  return Builder->CreateCall(F, Ops, "binop");
}




/*
void matmul_forward2(float* out,
                     const float* inp, const float* weight, const float* bias,
                     int B, int T, int C, int OC,
                     const int sqrt_block_size) {*/
void matmul_forward2(float* out,
                     const float* inp, const float* weight,
                     int B, int C, int OC) {
                     //const int sqrt_block_size
                     
    
    const float alpha = 1.0f;
    const float beta = 0.0f;
    
    std::cout << "matmul_forward. B: " << B << " C: " << C << " OC: " << OC << "\n";
    
    cublasCheck(cublasSgemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, OC, B, C, &alpha, weight, C, inp, C, &beta, out, OC));


    /* //bias
    if (bias != NULL) {
        int block_size = sqrt_block_size * sqrt_block_size;
        int grid_size = ceil_div(OC * B * T, block_size);
        add_bias<<<grid_size, block_size>>>(out, bias, B, T, OC);
        cudaCheck(cudaGetLastError());
    }
    */
}

void matmul_backward(float *inp,  float *weight,
                     int B, int C, int OC,
                     float *dinp, float *dw,
                     float *dout)
{
  std::cout << "matmul_backward. B: " << B << " C: " << C << " OC: " << OC << "\n";

  float one = 1.0f, zero = 0.0f;
  // backward to input
  cublasCheck(cublasGemmEx(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, C, B, OC, &one,
                             weight, CUBLAS_LOWP, C, dout, CUBLAS_LOWP, OC, &zero,
                             dinp, CUBLAS_LOWP, C, cublas_compute, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
  // backward to weight, uses += in the backward pass (accumulate the gradient) by setting alpha=one
  cublasCheck(cublasGemmEx(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T, C, OC, B, &one,
                             inp, CUBLAS_LOWP, C, dout, CUBLAS_LOWP, OC, &one,
                             dw, CUBLAS_LOWP, C, cublas_compute, CUBLAS_GEMM_DEFAULT_TENSOR_OP));

  cudaCheck(cudaGetLastError());
}



using backward_tuple = std::tuple<int, int, int, float *, float *, float *, std::string, std::string>;
std::vector<backward_tuple> todo_backwards;



extern "C" float CudaMult(char *LtensorName, char *RtensorName, int _used_cuda, int is_forward_func) {
  
  float * device_x;
  float * device_w;
  

  if (_used_cuda==1)
    device_x = currentCudaResult;
  else
  {
    device_x = NamedTensors[LtensorName];
    currentDims = NamedDims[LtensorName];
  }

  std::cout << "Left tensor dims:\n";
  PrintDims(currentDims);

  device_w = NamedTensors[RtensorName];

  std::vector<float> Rdims = NamedDims[RtensorName];
  


  std::vector<float> linear_layer_dims = format_LinearLayer_Dims(currentDims);
  int input_dims_prod = dimsProd(linear_layer_dims);
  //int resultingDimsProd = (int)linear_layer_dims[0]*Rdims[0];
  int resultingDimsProd = resultingDimsProdOnMult(linear_layer_dims, Rdims);


  float* device_y;
  cudaCheck(cudaMalloc(&device_y, resultingDimsProd * sizeof(float)));

  if (currentDims.size()<2)
    LogErrorS("Tensor de entrada da multiplicação de tensors precisa ter ao menos 2 dimensões.");


  matmul_forward2(device_y, device_x, device_w,
                  linear_layer_dims[0], linear_layer_dims[1],
                  Rdims[0]);
                  //64
                  //);


  currentCudaResult = device_y;
  currentDims = newDimsOnMult(currentDims, Rdims);

  //std::cout << "is_forward_func: " << is_forward_func << "\n";
  if (is_forward_func)
  {
    float *inp, *out;

    cudaCheck(cudaMalloc(&inp, input_dims_prod * sizeof(float)));
    cudaCheck(cudaMalloc(&out, resultingDimsProd * sizeof(float)));
    cudaMemcpy(inp, device_x, input_dims_prod * sizeof(float), cudaMemcpyDeviceToDevice);
    cudaMemcpy(out, device_y, resultingDimsProd * sizeof(float), cudaMemcpyDeviceToDevice);

    todo_backwards.push_back(std::make_tuple(linear_layer_dims[0], linear_layer_dims[1],
                                           Rdims[0], inp, device_w, out,
                                           "matmul", RtensorName));
  }
  return 0;
}

int num_classes=5;

float eps = 1e-8;

// warp-level reduction for finding the maximum value
__device__ float warpReduceMax(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
    }
    return val;
}

// warp-level reduction for summing values
__device__ float warpReduceSum(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

__global__ void softmax_forward_kernel4(const float* inp, float* out, int N, int C) {
    // out is (N, C) just like inp. Each row of inp will get softmaxed.
    // same as kernel3, but can handle any block size (multiple of 32)
    // each row of C elements is handled by block_size threads
    // furthermore, each block_size threads get executed in warps of 32 threads

    // special reduction operations warpReduceMax/warpReduceSum are used for intra-warp reductions
    // shared memory is used for inter-warp reduction
    extern __shared__ float shared[];
    int idx = blockIdx.x;
    int tid = threadIdx.x;
    int warpId = threadIdx.x / 32; // warp index within a block
    int laneId = threadIdx.x % 32; // thread index within a warp

    // the number of warps per block. recall that blockDim.x is block_size
    int warpsPerBlock = blockDim.x / 32;

    // shared[] must be allocated to have 2 * warpsPerBlock elements
    // first half for max values, the second half for sum values
    float* maxvals = shared;
    float* sumvals = &shared[warpsPerBlock];

    // one row of inp, i.e. inp[idx, :] of shape (C,)
    const float* x = inp + idx * C;

    // first, thread coarsening by directly accessing global memory in series
    float maxval = -INFINITY;
    for (int i = tid; i < C; i += blockDim.x) {
        maxval = fmaxf(maxval, x[i]);
    }
    // now within-warp reductions for maxval
    maxval = warpReduceMax(maxval);

    // the 0th thread of each warp writes the maxval of that warp to shared memory
    if (laneId == 0) maxvals[warpId] = maxval;
    __syncthreads();

    // now the 0th thread reduces the maxvals in shared memory, i.e. across warps
    if (tid == 0) {
        float val = maxvals[tid];
        for (int i = 1; i < warpsPerBlock; i++) {
            val = fmaxf(val, maxvals[i]);
        }
        // store the final max in the first position
        maxvals[0] = val;
    }
    __syncthreads();
    // broadcast the max to all threads
    float offset = maxvals[0];

    // compute expf and write the result to global memory
    for (int i = tid; i < C; i += blockDim.x) {
        out[idx * C + i] = expf(x[i] - offset);
    }

    // okay now we calculated exp(x - max(x))
    // step 2: sum all the values and divide by the sum

    // thread coarsening for sum
    x = out + idx * C;
    float sumval = 0.0f;
    for (int i = tid; i < C; i += blockDim.x) {
        sumval += x[i];
    }
    // within-warp reduction for sumval
    sumval = warpReduceSum(sumval);

    // write sumval to shared memory
    if (laneId == 0) sumvals[warpId] = sumval;
    __syncthreads();

    // inter-thread reduction of sum
    if (tid == 0) {
        float val = sumvals[tid];
        for (int i = 1; i < warpsPerBlock; ++i) {
            val += sumvals[i];
        }
        sumvals[0] = val;
    }
    __syncthreads();
    // broadcast the sum to all threads
    float sum = sumvals[0];

    // divide the whole row by the sum
    for (int i = tid; i < C; i += blockDim.x) {
        out[idx * C + i] = x[i] / sum;
    }
}



__global__ void crossentropy_softmax_backward_kernel1(float* dlogits,
                           const float* probs, const float* targets,
                           int B, int T, int C) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    //int i = threadIdx.x;
    
    // Parallelizes over block_size blocks, each of size B*C
    if (i < B * T * C) {
        int b = i / (C);
        int v = i % C;

        float* dlogits_bt = dlogits + b * C;
        const float* probs_bt = probs + b * C;

        int ix = targets[b];

        float p = probs_bt[v];

        float indicator = (v==ix) ? 1.0f : 0.0f; // TODO : change this to the line below
        //float indicator = 0.0f;

        dlogits_bt[v] += (p - indicator) / B;
        
    }
}

extern "C" float softmax(char * tensor_name)
{
  std::cout << "Called softmax for: " << tensor_name << "\n";

  float * tensor = NamedTensors[tensor_name];
  std::vector<float> dims = NamedDims[tensor_name];
  dims =  format_LinearLayer_Dims(dims);
  int B = dims[0];
  int C = dims[1];

  int grid_size = B;
  int block_size = 32;
  size_t shared_mem_size = 2 * block_size / 32 * sizeof(float);

  //float *cpu_out = make_zeros_float(B*C);
  float *probs;
  cudaMalloc(&probs, B*C*sizeof(float));

  softmax_forward_kernel4<<<grid_size, block_size, shared_mem_size>>>(tensor, probs, B, C);

  std::cout << "\n\nPROBS ARE:\n\n";
  PrintTensorF(probs, B, C);

  return 0;
}

void CrossEntropyBackward(float *y_hat,
                          float *y,
                          int B, int C, int OC,
                          float *dloss)
{
  std::cout << "Cross entropy backward\n";

  int grid_size = B;
  int block_size = 32;
  size_t shared_mem_size = 2 * block_size / 32 * sizeof(float);

  //float *cpu_out = make_zeros_float(B*C);
  float *probs;
  cudaMalloc(&probs, B*C*sizeof(float));

  std::cout << "\n\nINPUT TO SOFTMAX:\n";
  PrintTensorF(y_hat, B, C);

  softmax_forward_kernel4<<<grid_size, block_size, shared_mem_size>>>(y_hat, probs, B, C);

  std::cout << "\n\nPROBS ARE:\n\n";
  PrintTensorF(probs, B, C);

  grid_size = ceil_div(B*C, block_size);
  crossentropy_softmax_backward_kernel1<<<grid_size, block_size>>>(dloss, probs, y, B, 1, C);

  cudaCheck(cudaGetLastError());
}



extern "C" float CrossEntropy(char *y_hat, char *y)
{
  std::cout << "\n Cross Entropy call\n";

  float *device_y_hat = NamedTensors[y_hat];
  float *device_y = NamedTensors[y];
  std::vector<float> y_hat_dims = NamedDims[y_hat];
  std::vector<float> y_dims = NamedDims[y];
  std::cout << "y_hat: " << y_hat << "\n";


  PrintDims(y_hat_dims);

  std::cout << "y: " << y << "\n";
  PrintDims(y_dims);

  std::vector<float> linear_layer_dims = format_LinearLayer_Dims(y_hat_dims);
  

  todo_backwards.push_back(std::make_tuple(linear_layer_dims[0], linear_layer_dims[1],
                                           y_dims[0], device_y_hat, nullptr, device_y,
                                           "cross_entropy", "none"));


  return 0;
}



extern "C" float Backpropagation()
{
  std::cout << "\n Called Loss Backward \n\n";
  std::cout << "Todo Backwards size: " << todo_backwards.size() << "\n";

  //float * loss_gradient = ;
  
  int B, C, OC;
  float *inp, *w, *out;
  float *dinp, *device_dinp, *dw, *device_dw, *dout, *device_dout;

  std::string op, param_name;

  while(todo_backwards.size()>0)
  {
    backward_tuple bt = std::move(todo_backwards.back());
    todo_backwards.pop_back();


    
    B = std::get<0>(bt);
    C = std::get<1>(bt);
    OC = std::get<2>(bt);
    inp = std::get<3>(bt);
    w = std::get<4>(bt);
    out = std::get<5>(bt);
    op = std::get<6>(bt);
    param_name = std::get<7>(bt);

    dinp = make_zeros_float(B*C);
    dw = make_zeros_float(OC*C);
    

    cudaMalloc(&device_dinp, B*C*sizeof(float));
    cudaMalloc(&device_dw, OC*C*sizeof(float));
    

    cudaMemcpy(device_dinp, dinp, B*C*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(device_dw, dw, OC*C*sizeof(float), cudaMemcpyHostToDevice);
    

    std::cout << "B: " << B << "\n";
    std::cout << "C: " << C << "\n";
    std::cout << "OC: " << OC << "\n";
    std::cout << "Op: " << op << "\n";


    // No switch case for std::string
    if (op=="matmul")
      matmul_backward(inp, w, B, C, OC, device_dinp, device_dw, device_dout);
    else if (op=="cross_entropy")
      CrossEntropyBackward(inp, out, B, C, OC, device_dinp);
    else
      LogErrorS("A operação não possui implementação do backward.");

    std::cout << "\nd inp:\n";
    PrintTensorF(device_dinp, B, C);
    std::cout << "\n";

    std::cout << "d w:\n";
    PrintTensorF(device_dw, OC, C);
    std::cout << "\n\n";

    device_dout = device_dinp;

    std::cout << "Added grad for parameter: " << param_name << "\n";
    NamedParamGrads[param_name] = std::move(device_dw);
  }

  return 0;
}


class Optimizer {
public:
  virtual ~Optimizer() = default;

  int timestep = 1;
  float lr = 0.0f;
  float eps = 1.5e-4;
    
  virtual void init_states(std::string param_name, float params_count) {}
  virtual void step(float *param, float *grad, std::vector<float> dims, std::string param_name) {}
  virtual void count_step() {
    timestep+=1;
  }
};

class AdamW_optim : public Optimizer {
  std::map<std::string, float *> NamedV, NamedM;
  float lr, beta1, beta2, weight_decay;

  public:
    AdamW_optim(float lr, float beta1, float beta2, float weight_decay)
      : lr(lr), beta1(beta1), beta2(beta2), weight_decay(weight_decay) {}
    
  void init_states(std::string param_name, float params_count) override;
  void step(float *param, float *grad, std::vector<float> dims, std::string param_name) override;
};



__device__ inline float lerp(float start, float end, float weight) {
    return fma(weight, end, fma(-weight, start, start));
}

__global__ void adamw_kernel(float* params_memory, const float* grads_memory, float* m_memory, float* v_memory, long num_parameters,
                              float learning_rate, float beta1, float beta2, float beta1_correction, float beta2_correction,
                              float eps, float weight_decay) {

   int i = blockIdx.x * blockDim.x + threadIdx.x;
   if (i >= num_parameters) return;  // guard
   float grad = grads_memory[i];
   float m = m_memory[i];
   float v = v_memory[i];
   // update the first moment (momentum)
   m = lerp(grad, m, beta1);
   m_memory[i] = m;
   // update the second moment (RMSprop)
   v = lerp(grad * grad, v, beta2);
   v_memory[i] = v;
   m /= beta1_correction;  // m_hat
   v /= beta2_correction;  // v_hat
   params_memory[i] -= learning_rate * (m / (sqrtf(v) + eps) + weight_decay * params_memory[i]);
}


void AdamW_optim::init_states(std::string param_name, float params_count)
{
  

  if (NamedV[param_name]==nullptr)
  {
    std::cout << "init_states for param " << param_name << " with params count: " << params_count << "\n";

    float *v, *m, *device_v, *device_m;
    v = new float[params_count];
    m = new float[params_count];

    v = make_zeros_float(params_count);
    m = make_zeros_float(params_count);


    cudaMalloc(&device_v, params_count*sizeof(float));
    cudaMalloc(&device_m, params_count*sizeof(float));
    cudaMemcpy(device_v, v, params_count*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(device_m, m, params_count*sizeof(float), cudaMemcpyHostToDevice);

    NamedV[param_name] = device_v; 
    NamedM[param_name] = device_m;
  }
}

void AdamW_optim::step(float *param, float *grad, std::vector<float> dims, std::string param_name)
{
  std::cout  << "Optimizer step called\n";
  
  std::cout << "learning rate: " <<  lr << "\n";
  std::cout << "beta 1: " <<  beta1 << "\n";
  std::cout << "beta 2: " <<  beta2 << "\n";
  std::cout << "weight decay: " <<  weight_decay << "\n";
  std::cout << "eps: " <<  eps << "\n";

  float *v = NamedV[param_name];
  float *m = NamedM[param_name];

  float beta1_correction = 1.0f - powf(beta1, timestep);
  float beta2_correction = 1.0f - powf(beta2, timestep);

  
  std::cout << "Timestep: " << timestep << "\n\n";

  std::cout << "param pre: \n";
  PrintTensorF(param, dims[0], dims[1]);

  std::cout << "\n\ngrad: \n";
  PrintTensorF(grad, dims[0], dims[1]);

  int params_count = dims[0]*dims[1];
  int block_size = 512;
  int num_blocks = ceil_div(params_count, block_size);

  adamw_kernel<<<num_blocks, block_size>>>(param, grad, m, v, params_count,
                                           lr, beta1, beta2, beta1_correction, beta2_correction,
                                           eps, weight_decay);


  std::cout << "\n\nparam post: \n";
  PrintTensorF(param, dims[0], dims[1]);

  std::cout << "\n\n";

}






std::unique_ptr<Optimizer> optimizer = nullptr;

extern "C" float AdamW(float lr, float beta1, float beta2, float weight_decay)
{
  std::cout << "\n\nCalled AdamW\n";

  if (optimizer==nullptr)
    optimizer = std::make_unique<AdamW_optim>(lr, beta1, beta2, weight_decay);


  for (auto& pair : NamedParamGrads)
  {
    std::string param_name = pair.first;

    if (param_name!="none")
    {
      optimizer->init_states(param_name, dimsProd(NamedDims[param_name]));
      optimizer->step(NamedTensors[param_name], pair.second, NamedDims[param_name], param_name);
    }
  }
  optimizer->count_step();

  used_cuda=0;

  return 0;
}




Value *BinaryTensorTensorExprAST::codegen() {
  Value *LtensorName = Builder->CreateGlobalString(LHS->GetName());
  Value *RtensorName = Builder->CreateGlobalString(RHS->GetName());
  Value * object_name;

  std::string pre_dot = LHS->GetSelf();
  if (pre_dot=="true")
    LtensorName = Builder->CreateCall(TheModule->getFunction("ConcatFirstArgToVarName"),
                                                      {LtensorName});
    // Gets from pre_dot if it is a class attribute
  else if (pre_dot!="false") {
    object_name = Builder->CreateGlobalString(pre_dot);

    LtensorName = Builder->CreateCall(TheModule->getFunction("ConcatStr"),
                                                      {object_name, LtensorName});
  }
  pre_dot = RHS->GetSelf();
  if (pre_dot=="true")
    RtensorName = Builder->CreateCall(TheModule->getFunction("ConcatFirstArgToVarName"),
                                                      {RtensorName});
    // Gets from pre_dot if it is a class attribute
  else if (pre_dot!="false") {
    object_name = Builder->CreateGlobalString(pre_dot);

    RtensorName = Builder->CreateCall(TheModule->getFunction("ConcatStr"),
                                                      {object_name, RtensorName});
  }


  if (Op == '=') {
    seen_var_attr=true;

    VariableExprAST *LHSE = static_cast<VariableExprAST *>(LHS.get());
    if (!LHSE)
      return LogErrorV("Destino do '=' deve ser uma variável.");
    
    
    Value *Val = RHS->codegen();
    std::cout << "1 1 attr\n";
    if (!Val)
      return nullptr;

    //float *Variable = NamedTensors[LHSE->getName()];
    //if (!Variable)
    //  return LogErrorV("O nome do tensor/variável é desconhecido.");

    
    Function *temporaryCudaResult_AttrFn = TheModule->getFunction("temporaryCudaResult_Attr");
    Builder->CreateCall(temporaryCudaResult_AttrFn, {LtensorName});


    used_cuda=0;
      
      
    
    seen_var_attr=false;
    return Val;
  }


  Value *L = LHS->codegen();
  Value *R = RHS->codegen();
  

  std::string functionName = Builder->GetInsertBlock()->getParent()->getName().str();
  std::cout << "Tensor Tensor for function: " << functionName << "\n";
  int forward_func = 0;
  if(ends_with(functionName, "forward"))
    forward_func = 1;
  forward_func = 1; // TODO: Remove this line


  
  if (!L || !R)
    return nullptr;

    Function *CudaFn;

    std::cout << "Tensor tensor: " << LHS->GetName() << " " << RHS->GetName() << "\n";
    

    Value *used_cuda_aux = ConstantInt::get(Type::getInt32Ty(*TheContext), used_cuda);
    Value *is_forward_func = ConstantInt::get(Type::getInt32Ty(*TheContext), forward_func);
    used_cuda = 1;

  switch (Op)
  {
  case '@':
    CudaFn = TheModule->getFunction("CudaMult");
    return Builder->CreateCall(CudaFn,{LtensorName, RtensorName, used_cuda_aux, is_forward_func},
                               "cudamult");
  case '*':
    CudaFn = TheModule->getFunction("CudaMult");
    return Builder->CreateCall(CudaFn,{LtensorName, RtensorName, used_cuda_aux, is_forward_func},
                               "cudamult");
  case '/':
    CudaFn = TheModule->getFunction("CudaDiv");
    return Builder->CreateCall(CudaFn, {LtensorName, RtensorName, used_cuda_aux},
                               "cudadiv");
  case '+':
    CudaFn = TheModule->getFunction("CudaAdd");
    return Builder->CreateCall(CudaFn, {LtensorName, RtensorName, used_cuda_aux},
                               "cudaadd");
  case '-':
    CudaFn = TheModule->getFunction("CudaSub");
    return Builder->CreateCall(CudaFn, {LtensorName, RtensorName, used_cuda_aux},
                               "cudasub");
  case ':':
    return L;
  case tok_space:
    return R;
  default:
    break;
  }
  

  
  Function *F = getFunction(std::string("binary") + Op);
  assert(F && "Operator not found.");

  Value *Ops[] = {L, R};
  return Builder->CreateCall(F, Ops, "binop");
}


Value *LossBackwardExprAST::codegen()
{
  return Builder->CreateCall(TheModule->getFunction("Backpropagation"),
                             {}, "backprop");
}

Value *LogExprAST::codegen() {
  
  Value *used_cuda_aux = ConstantInt::get(Type::getInt32Ty(*TheContext), used_cuda);
  used_cuda=1;

  return Builder->CreateCall(TheModule->getFunction("logE"),
                             {Builder->CreateGlobalString(Name), used_cuda_aux}, "cudalog");
}



Value *BinaryExprAST::codegen() {
  // Special case '=' because we don't want to emit the LHS as an expression.
  if (Op == '=') {
    seen_var_attr=true;
    // Assignment requires the LHS to be an identifier.
    // This assume we're building without RTTI because LLVM builds that way by
    // default.  If you build LLVM with RTTI this can be changed to a
    // dynamic_cast for automatic error checking.
    VariableExprAST *LHSE = static_cast<VariableExprAST *>(LHS.get());
    if (!LHSE)
      return LogErrorV("Destino do '=' deve ser uma variável.");
    // Codegen the RHS.
    
    Value *Val = RHS->codegen();
    if (!Val)
      return nullptr;

    // Look up the name.
    if (NamedValues.count(LHSE->getName()) != 0) {
      
      Value *Variable = NamedValues[LHSE->getName()];
      
      if (!Variable)
        return LogErrorV("O nome da variável é desconhecido.");


      if(LHS->GetSelf()=="true")
        Builder->CreateCall(TheModule->getFunction("StoreOnDemand"),
                                                  {Builder->CreateGlobalString(LHSE->getName()),
                                                   Val});
      else
        Builder->CreateStore(Val, Variable);
      
    
    } else if (NamedTensors.count(LHSE->getName()) != 0 ) {
      /*
      float *Variable = NamedTensors[LHSE->getName()];
      if (!Variable)
        return LogErrorV("O nome do tensor/variável é desconhecido.");
      */
      std::cout << "Atribuíndo em 0 0\n";
      
      Value *valStr = Builder->CreateGlobalString(LHSE->getName());
      Function *temporaryCudaResult_AttrFn = TheModule->getFunction("temporaryCudaResult_Attr");
      Builder->CreateCall(temporaryCudaResult_AttrFn, {valStr});
        
      used_cuda=0;
      
    }
    seen_var_attr=false;
    return Val;
  }


  

  Value *L = LHS->codegen();
  Value *R = RHS->codegen();
  
  if (!L || !R)
    return nullptr;


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
    default:
      break;
    }
  

  // If it wasn't a builtin binary operator, it must be a user defined one. Emit
  // a call to it.
  Function *F = getFunction(std::string("binary") + Op);
  assert(F && "Operator not found.");

  Value *Ops[] = {L, R};
  return Builder->CreateCall(F, Ops, "binop");
}


Value *UnaryExprAST::codegen() {
  Value *OperandV = Operand->codegen();
  if (!OperandV)
    return nullptr;

  
  // todo: Adjust this for tensors
  std::cout << "unary: " << used_cuda << "\n";
  std::cout << "Operand type" << Operand->GetType();
  if (Opcode=='-')
  {
    if (Operand->GetType()=="tensor")
    {
      Value *tensorName = Builder->CreateGlobalString(Operand->GetName());
      Value *used_cuda_aux = ConstantInt::get(Type::getInt32Ty(*TheContext), used_cuda);
      Value *R = ConstantFP::get(Type::getFloatTy(*TheContext), -1);
      used_cuda=1;
      return Builder->CreateCall(TheModule->getFunction("CudaScalarMult"),
                                {tensorName, R, used_cuda_aux}, "cudascalarmult");
    }
    return Builder->CreateFMul(ConstantFP::get(Type::getFloatTy(*TheContext), -1),
                              OperandV, "multmp");
  }

  Function *F = getFunction(std::string("unary") + Opcode);
  if (!F)
    return LogErrorV("Operador unário desconhecido.");

  return Builder->CreateCall(F, OperandV, "unop");
}


Value *IfExprAST::codegen() {
  Value *CondV = Cond->codegen();
  if (!CondV)
    return nullptr;

  // Convert condition to a bool by comparing equal to 0.0.
  CondV = Builder->CreateFCmpONE(
      CondV, ConstantFP::get(*TheContext, APFloat(0.0)), "ifcond");

  Function *TheFunction = Builder->GetInsertBlock()->getParent();

  // Create blocks for the then and else cases.  Insert the 'then' block at the
  // end of the function.
  BasicBlock *ThenBB = BasicBlock::Create(*TheContext, "then", TheFunction);
  BasicBlock *ElseBB = BasicBlock::Create(*TheContext, "else");
  BasicBlock *MergeBB = BasicBlock::Create(*TheContext, "ifcont");

  Builder->CreateCondBr(CondV, ThenBB, ElseBB);

  // Emit then value.
  Builder->SetInsertPoint(ThenBB);

  Value *ThenV = Then->codegen();
  if (!ThenV)
    return nullptr;

  Builder->CreateBr(MergeBB);
  // Codegen of 'Then' can change the current block, update ThenBB for the PHI.
  ThenBB = Builder->GetInsertBlock();

  // Emit else block.
  TheFunction->insert(TheFunction->end(), ElseBB);
  Builder->SetInsertPoint(ElseBB);

  Value *ElseV = Else->codegen();

  if (!ElseV)
  {
    return nullptr;
  }
    

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
Value *WhileExprAST::codegen() {
	Function* TheFunction = Builder->GetInsertBlock()->getParent();

	BasicBlock *entryBB = BasicBlock::Create(*TheContext, "entry_while", TheFunction);
	BasicBlock *LoopBB = BasicBlock::Create(*TheContext, "loop_while", TheFunction);
	BasicBlock *AfterBB = BasicBlock::Create(*TheContext, "end_while", TheFunction);

	
	Builder->CreateBr(entryBB);

	// Handle Cond

	Builder->SetInsertPoint(entryBB);
	Value* condVal = Cond->codegen();
	if (! condVal)
    return nullptr;

	condVal = Builder->CreateFCmpONE(condVal, ConstantFP::get(*TheContext, APFloat(0.0)), "loopcond");
	Builder->CreateCondBr(condVal, LoopBB, AfterBB);
	entryBB = Builder->GetInsertBlock();


	// Handle Loop Body
	
  Builder->SetInsertPoint(LoopBB);
	Value* bodyVal = Body->codegen();
	if (! bodyVal)
    return nullptr;
	Builder->CreateBr(entryBB);


	// Handle Loop End
	
	Builder->SetInsertPoint(AfterBB);

	return Constant::getNullValue(Type::getFloatTy(*TheContext));
}


Value *ForExprAST::codegen() {
  Function *TheFunction = Builder->GetInsertBlock()->getParent();

  // Create an alloca for the variable in the entry block.
  AllocaInst *Alloca = CreateEntryBlockAlloca(TheFunction, VarName);

  // Emit the start code first, without 'variable' in scope.
  Value *StartVal = Start->codegen();
  if (!StartVal)
    return nullptr;

  // Store the value into the alloca.
  Builder->CreateStore(StartVal, Alloca);

  // Make the new basic block for the loop header, inserting after current
  // block.
  BasicBlock *LoopBB = BasicBlock::Create(*TheContext, "loop", TheFunction);

  // Insert an explicit fall through from the current block to the LoopBB.
  Builder->CreateBr(LoopBB);

  
  Builder->SetInsertPoint(LoopBB);

  // Within the loop, the variable is defined equal to the PHI node.  If it
  // shadows an existing variable, we have to restore it outside this scope
  AllocaInst *OldVal = NamedValues[VarName];
  NamedValues[VarName] = Alloca;

  // Emit the body of the loop.  This, like any other expr, can change the
  // current BB.  Note that we ignore the value computed by the body, but don't
  // allow an error.
  if (!Body->codegen())
    return nullptr;

  // Emit the step value.
  Value *StepVal = nullptr;
  if (Step) {
    StepVal = Step->codegen();
    if (!StepVal)
      return nullptr;
  } 

  // Compute the end condition.
  Value *EndCond = End->codegen();
  if (!EndCond)
    return nullptr;

  // Reload, increment, and restore the alloca.  This handles the case where
  // the body of the loop mutates the variable.
  Value *CurVar = Builder->CreateLoad(Type::getFloatTy(*TheContext), Alloca,
                                      VarName.c_str());
  Value *NextVar = Builder->CreateFAdd(CurVar, StepVal, "nextvar"); // Increment
  Builder->CreateStore(NextVar, Alloca);

  // Convert condition to a bool by comparing equal to 0.0.
  EndCond = Builder->CreateFCmpONE(
      EndCond, ConstantFP::get(*TheContext, APFloat(0.0)), "loopcond");

  // Create the "after loop" block and insert it.
  BasicBlock *AfterBB =
      BasicBlock::Create(*TheContext, "afterloop", TheFunction);

  // goto branch
  Builder->CreateCondBr(EndCond, LoopBB, AfterBB);

  // Any new code will be inserted in AfterBB.
  Builder->SetInsertPoint(AfterBB);

  // Restore the unshadowed variable.
  if (OldVal)
    NamedValues[VarName] = OldVal;
  else
    NamedValues.erase(VarName);

  // for expr always returns 0.0.
  return Constant::getNullValue(Type::getFloatTy(*TheContext));
}


// Create Var
Value *VarExprAST::codegen() {
  std::vector<AllocaInst *> OldBindings;

  Function *TheFunction = Builder->GetInsertBlock()->getParent();


  // Register all variables and emit their initializer.
  for (unsigned i = 0, e = VarNames.size(); i != e; ++i) {
    const std::string &VarName = VarNames[i].first;
    ExprAST *Init = VarNames[i].second.get();

    // Emit the initializer before adding the variable to scope, this prevents
    // the initializer from referencing the variable itself, and permits stuff
    // like this:
    //  var a = 1 in
    //    var a = a in ...   # refers to outer 'a'.
    Value *InitVal;
    if (Init) {
      InitVal = Init->codegen();
      if (!InitVal)
        return nullptr;
    } else { // If not specified, use 0.0.
      InitVal = ConstantFP::get(*TheContext, APFloat(0.0));
    }


    AllocaInst *Alloca = CreateEntryBlockAlloca(TheFunction, VarName);
    Builder->CreateStore(InitVal, Alloca);
      
    // Remember the old variable binding so that we can restore the binding when
    // we unrecurse.
    OldBindings.push_back(NamedValues[VarName]);

    // Remember this binding.
    NamedValues[VarName] = Alloca;
    
    
    
  }

  // Codegen the body that is contained by the in expression
  Value *BodyVal = Body->codegen();
  if (!BodyVal)
    return nullptr;

  // Pop all our variables from scope.
  for (unsigned i = 0, e = VarNames.size(); i != e; ++i)
    NamedValues[VarNames[i].first] = OldBindings[i];

  // Return the body computation.
  return BodyVal;
}


std::vector<float> cur_dim;

extern "C" float StoreDimsOnDemand(float d)
{
  cur_dim.push_back(d);
  return 0;
}

extern "C" float CreateTensorOnDemand(char *tensorName, int is_obj_attr_or_self, char *init)
{
  
  std::string objectTensorName = tensorName;
  if (is_obj_attr_or_self)
    objectTensorName = FirstArg + tensorName;

  char * cObjectTensorName = new char[objectTensorName.length() + 1];
  std::strcpy(cObjectTensorName, objectTensorName.c_str());


  //float * d = (float *) dims;
  int product = dimsProd(cur_dim);
  float * tensor;
  float * tensor_cpu;


  if (std::strcmp(init, "randu") == 0)
    tensor_cpu = make_random_float(product);
  else if (std::strcmp(init, "zeros") == 0)
    tensor_cpu = make_zeros_float(product);
  else if (std::strcmp(init, "ones") == 0)
    tensor_cpu = make_ones_float(product);

  

  cudaMalloc(&tensor, product*sizeof(float));
  cudaCheck(cudaMemcpy(tensor, tensor_cpu, product*sizeof(float), cudaMemcpyHostToDevice));
  

  NamedTensors[cObjectTensorName] = tensor;
  NamedDims[cObjectTensorName] = cur_dim;



  PrintTensor(cObjectTensorName);

  cur_dim.clear();

  return 0;
}

Value *TensorExprAST::codegen() {
  std::vector<AllocaInst *> OldBindings;


  Function *TheFunction = Builder->GetInsertBlock()->getParent();

  // Register all variables and emit their initializer.
  for (unsigned i = 0, e = VarNames.size(); i != e; ++i) {
    const std::string &VarName = VarNames[i].first;
    ExprAST *Init = VarNames[i].second.get();

    // Emit the initializer before adding the variable to scope, this prevents
    // the initializer from referencing the variable itself, and permits stuff
    // like this:
    //  var a = 1 in
    //    var a = a in ...   # refers to outer 'a'.
    Value *InitVal;
    if (Init) {
      InitVal = Init->codegen();
      if (!InitVal)
        return nullptr;
    } else { // If not specified, use 0.0.
      InitVal = ConstantFP::get(*TheContext, APFloat(0.0));
    }


    std::vector<float> dims;
    Value *aux;
    std::vector<Value *> dim_values;


    for (int j=0; j<V_Dims.size(); j++)
    {
      aux = V_Dims[j]->codegen();
      Builder->CreateCall(TheModule->getFunction("StoreDimsOnDemand"),
                                                  {aux});
      //dims.push_back(cast<ConstantFP>(aux)->getValueAPF().convertToFloat());
      //std::cout << "Dim: " << cast<ConstantFP>(aux)->getValueAPF().convertToFloat() << "\n";
    }
    //void * v_dims_ptr = &V_Dims;

    
    int is_obj_attr_or_self = 0;
    if (GetSelf()!="false")
      is_obj_attr_or_self=1;
    
    Builder->CreateCall(TheModule->getFunction("CreateTensorOnDemand"),
                                              {Builder->CreateGlobalString(VarName),
                                               ConstantInt::get(Type::getInt32Ty(*GlobalContext), is_obj_attr_or_self),
                                               Builder->CreateGlobalString(TensorInit)});

    /*
    //SetDims(dims);

    int product = dimsProd(Dims);
    float * tensor_cpu = make_random_float(product);
    float * tensor;

    cudaMalloc(&tensor, product*sizeof(float));
    cudaCheck(cudaMemcpy(tensor, tensor_cpu, product*sizeof(float), cudaMemcpyHostToDevice));

    NamedTensors[VarName] = tensor;
    NamedDims[VarName] = Dims;



    
    
    Builder->CreateCall(TheModule->getFunction("PrintTensor"),
                        {Builder->CreateGlobalString(VarName)});
    */
     
  }

  // Codegen the body that is contained by the in expression

  Value *BodyVal = Body->codegen();
  if (!BodyVal)
    return nullptr;



  // Return the body computation.
  return BodyVal;
}




Value *CallExprAST::codegen() {
  // Look up the name in the global module table.
  std::string tgt_function = Callee;
  

  Function *TheFunction = Builder->GetInsertBlock()->getParent();
  std::string functionName = TheFunction->getName().str();
  std::string tgt_function_name;

  int nested_function;
  if (functionName=="__anon_expr")
    nested_function=0;
  else
    nested_function=1;

  int args_removal = 0;
  if(Class!="None")
  {
    tgt_function = Class+tgt_function;
    Builder->CreateCall(TheModule->getFunction("FirstArgOnDemand"),
                                                  {Builder->CreateGlobalString(Pre_dot),
                                                   ConstantInt::get(Type::getInt32Ty(*TheContext), nested_function)});
    //args_removal=1;
  }

  Function *CalleeF = getFunction(tgt_function);
  if (!CalleeF)
    return LogErrorV("Função referenciada desconhecida");
  tgt_function_name = CalleeF->getName().str();

  // If argument mismatch error.
  if ((CalleeF->arg_size()-args_removal) != Args.size())
    return LogErrorV("Parâmetros passados incorretos.");

  std::vector<Value *> ArgsV;
  
  //if(Class!="None")
  //  ArgsV.push_back(ConstantFP::get(Type::getFloatTy(*TheContext), APFloat(0.0f)));

  
  for (unsigned i = 0, e = Args.size(); i != e; ++i) {
    //std::cout << "\n\nCallExprAST::codegen for argument n°: " << i << ".\n";
    Value * arg;
    if (Args[i]->GetType()=="tensor")
      arg = Builder->CreateGlobalString(Args[i]->GetName());
    else
      arg = Args[i]->codegen();

    //std::cout << "Args[i]: " << Args[i]->GetName() << "\n";


    ArgsV.push_back(arg);


    if (!ArgsV.back())
      return nullptr;
  }
  
  //std::cout << "\n\n";

  Value * ret = Builder->CreateCall(CalleeF, ArgsV, "calltmp");
  if(Class!="None")
    Builder->CreateCall(TheModule->getFunction("DimnishFirstArgOnDemand"),
                                                  {Builder->CreateGlobalString(Pre_dot),
                                                   ConstantInt::get(Type::getInt32Ty(*TheContext), nested_function)});
  return ret;
}



Function *PrototypeAST::codegen() {
  // Make the function type:  float(float,float) etc.

  std::vector<Type *> Floats(Args.size(), Type::getFloatTy(*TheContext));
  
  /*
  if (Args.size()>0)
    if (Args[0]=="self")
      Floats[0] = PointerType::get(Type::getInt8Ty(*TheContext), 0);
  */

  FunctionType *FT = FunctionType::get(Type::getFloatTy(*TheContext), Floats, false);
  

  Function *F =
      Function::Create(FT, Function::ExternalLinkage, Name, TheModule.get());

  // Set names for all arguments.
  unsigned Idx = 0;
  for (auto &Arg : F->args())
    Arg.setName(Args[Idx++]);
  

  return F;
}

const PrototypeAST& FunctionAST::getProto() const {
  return *Proto;
}

const std::string& FunctionAST::getName() const {
  return Proto->getName();
}

Function *FunctionAST::codegen() {
  
  // Transfer ownership of the prototype to the FunctionProtos map, but keep a
  // reference to it for use below.
  auto &P = *Proto;
  FunctionProtos[Proto->getName()] = std::move(Proto);
  Function *TheFunction = getFunction(P.getName());
  if (!TheFunction)
    return nullptr;

  // If this is an operator, install it.
  if (P.isBinaryOp())
    BinopPrecedence[P.getOperatorName()] = P.getBinaryPrecedence();

  // Create a new basic block to start insertion into.
  BasicBlock *BB = BasicBlock::Create(*TheContext, "entry", TheFunction);
  Builder->SetInsertPoint(BB);



  // Record the function arguments in the NamedValues map.


  //std::cout << "\n\n";

  NamedValues.clear();

  float val;
  int i = 0;
  for (auto &Arg : TheFunction->args()) {
    // Create an alloca for this variable.

    
    //std::cout << "Create Function alloca for: " << Arg.getName().str() << "\n";
    if (Arg.getName()!="self")
    {
      AllocaInst *Alloca = CreateEntryBlockAlloca(TheFunction, Arg.getName());

      // Store the initial value into the alloca.
      Builder->CreateStore(&Arg, Alloca);

      // Add arguments to variable symbol table.
      NamedValues[std::string(Arg.getName())] = Alloca;
    }
  }
  //std::cout << "\n\n";


  if (Value *RetVal = Body->codegen()) {
    // Finish off the function.
    Builder->CreateRet(RetVal);

    // Validate the generated code, checking for consistency.
    verifyFunction(*TheFunction);

    return TheFunction;
  }


  // Error reading body, remove function.
  TheFunction->eraseFromParent();

  if (P.isBinaryOp())
    BinopPrecedence.erase(P.getOperatorName());
  return nullptr;
}





const PrototypeAST& ClassAST::getProto(int i) const {
  return Functions[i]->getProto(); 
}

const std::string& ClassAST::getName(int i) const {
  return Functions[i]->getProto().getName();
}

Value *ClassAST::codegen() {
  /*
  // Transfer ownership of the prototype to the FunctionProtos map, but keep a
  // reference to it for use below.
  auto &P = *Proto;
  FunctionProtos[Proto->getName()] = std::move(Proto);
  Function *TheFunction = getFunction(P.getName());
  if (!TheFunction)
    return nullptr;

  // If this is an operator, install it.
  if (P.isBinaryOp())
    BinopPrecedence[P.getOperatorName()] = P.getBinaryPrecedence();

  // Create a new basic block to start insertion into.
  BasicBlock *BB = BasicBlock::Create(*TheContext, "entry", TheFunction);
  Builder->SetInsertPoint(BB);



  // Record the function arguments in the NamedValues map.
  NamedValues.clear();
  float val;
  int i = 0;
  for (auto &Arg : TheFunction->args()) {
    // Create an alloca for this variable.
    AllocaInst *Alloca = CreateEntryBlockAlloca(TheFunction, Arg.getName());

    // Store the initial value into the alloca.
    Builder->CreateStore(&Arg, Alloca);

    // Add arguments to variable symbol table.
    NamedValues[std::string(Arg.getName())] = Alloca;

    
  }


  if (Value *RetVal = Body->codegen()) {
    // Finish off the function.
    Builder->CreateRet(RetVal);

    // Validate the generated code, checking for consistency.
    verifyFunction(*TheFunction);

    return TheFunction;
  }


  // Error reading body, remove function.
  TheFunction->eraseFromParent();

  
  if (P.isBinaryOp())
    BinopPrecedence.erase(P.getOperatorName());
  */
  return nullptr;
}





//===----------------------------------------------------------------------===//
// Top-Level parsing and JIT Driver
//===----------------------------------------------------------------------===//

static void InitializeModule() {
  // Open a new context and module.
  TheContext = std::make_unique<LLVMContext>();
  TheModule = std::make_unique<Module>("my cool jit", *TheContext);
  TheModule->setDataLayout(TheJIT->getDataLayout());

  //std::cout << "Initialize Module\n";
  // todo: It's creating one initialize for each ";" (top level expression).

  // Create a new builder for the module.
  Builder = std::make_unique<IRBuilder<>>(*TheContext);

  Type *floatPtrType = PointerType::get(Type::getFloatTy(*TheContext), 0);

  //===----------------------------------------------------------------------===//
  // Tensor -- Scalar   Operations
  //===----------------------------------------------------------------------===//

  // char *, float, int
  FunctionType *CudaScalarMultTy = FunctionType::get(
      Type::getFloatTy(*TheContext),
      {PointerType::get(Type::getInt8Ty(*TheContext), 0), Type::getFloatTy(*TheContext), Type::getInt32Ty(*TheContext)}, 
      false // Not vararg
  );

  Function::Create(
    CudaScalarMultTy,
    Function::ExternalLinkage, // Linkage (e.g., external for linking with other modules)
    "CudaScalarMult", // Function name
    TheModule.get() // Module to which the function belongs
  );



  // char *, float, int
  FunctionType *CudaScalarDivTy = FunctionType::get(
      Type::getFloatTy(*TheContext),
      {PointerType::get(Type::getInt8Ty(*TheContext), 0), Type::getFloatTy(*TheContext), Type::getInt32Ty(*TheContext)}, 
      false // Not vararg
  );

  Function::Create(
    CudaScalarDivTy,
    Function::ExternalLinkage, // Linkage (e.g., external for linking with other modules)
    "CudaScalarDiv", // Function name
    TheModule.get() // Module to which the function belongs
  );



  // char *, float, int
  FunctionType *CudaScalarAddTy = FunctionType::get(
      Type::getFloatTy(*TheContext),
      {PointerType::get(Type::getInt8Ty(*TheContext), 0), Type::getFloatTy(*TheContext), Type::getInt32Ty(*TheContext)}, 
      false // Not vararg
  );

  Function::Create(
    CudaScalarAddTy,
    Function::ExternalLinkage, 
    "CudaScalarAdd", 
    TheModule.get() 
  );



  // char *, float, int
  FunctionType *CudaScalarSubTy = FunctionType::get(
      Type::getFloatTy(*TheContext),
      {PointerType::get(Type::getInt8Ty(*TheContext), 0), Type::getFloatTy(*TheContext), Type::getInt32Ty(*TheContext)}, 
      false // Not vararg
  );

  Function::Create(
    CudaScalarSubTy,
    Function::ExternalLinkage, 
    "CudaScalarSub", 
    TheModule.get()
  );


  //===----------------------------------------------------------------------===//
  // Tensor Tensor CUDA Ops
  //===----------------------------------------------------------------------===//


  // char *, char *, int
  FunctionType *CudaMultTy = FunctionType::get(
      Type::getFloatTy(*TheContext),
      {PointerType::get(Type::getInt8Ty(*TheContext), 0), PointerType::get(Type::getInt8Ty(*TheContext), 0), Type::getInt32Ty(*TheContext), Type::getInt32Ty(*TheContext)}, 
      false // Not vararg
  );

  Function::Create(
    CudaMultTy,
    Function::ExternalLinkage, // Linkage (e.g., external for linking with other modules)
    "CudaMult", // Function name
    TheModule.get() // Module to which the function belongs
  );



  //===----------------------------------------------------------------------===//
  // Backward and Optimizers CUDA Ops
  //===----------------------------------------------------------------------===//

  //
  FunctionType *BackpropagationTy = FunctionType::get(
      Type::getFloatTy(*TheContext),
      {}, 
      false // Not vararg
  );
  Function::Create(
    BackpropagationTy,
    Function::ExternalLinkage,
    "Backpropagation",
    TheModule.get()
  );

  //
  FunctionType *AdamWTy = FunctionType::get(
      Type::getFloatTy(*TheContext),
      {Type::getFloatTy(*TheContext),
       Type::getFloatTy(*TheContext),
       Type::getFloatTy(*TheContext),
       Type::getFloatTy(*TheContext)}, 
      false // Not vararg
  );
  Function::Create(
    AdamWTy,
    Function::ExternalLinkage,
    "AdamW",
    TheModule.get()
  );

  //===----------------------------------------------------------------------===//
  // Unary CUDA Ops
  //===----------------------------------------------------------------------===//

  // char *, int
  FunctionType *CudaLogTy = FunctionType::get(
      Type::getFloatTy(*TheContext),
      {PointerType::get(Type::getInt8Ty(*TheContext), 0), Type::getInt32Ty(*TheContext)},
      false // Not vararg
  );
  Function::Create(
    CudaLogTy,
    Function::ExternalLinkage,
    "logE",
    TheModule.get()
  );


  // char *
  FunctionType *softmaxTy = FunctionType::get(
      Type::getFloatTy(*TheContext),
      {PointerType::get(Type::getInt8Ty(*TheContext), 0)},
      false // Not vararg
  );
  Function::Create(
    softmaxTy,
    Function::ExternalLinkage,
    "softmax",
    TheModule.get()
  );


  //===----------------------------------------------------------------------===//
  // Loss CUDA Ops
  //===----------------------------------------------------------------------===//


  // char *, char *
  FunctionType *CrossEntropyTy = FunctionType::get(
      Type::getFloatTy(*TheContext),
      {PointerType::get(Type::getInt8Ty(*TheContext), 0), PointerType::get(Type::getInt8Ty(*TheContext), 0)}, 
      false // Not vararg
  );
  Function::Create(
    CrossEntropyTy,
    Function::ExternalLinkage, // Linkage (e.g., external for linking with other modules)
    "CrossEntropy", // Function name
    TheModule.get() // Module to which the function belongs
  );


  //===----------------------------------------------------------------------===//
  // Other Ops
  //===----------------------------------------------------------------------===//


  // char *, int
  FunctionType *FirstArgOnDemandTy = FunctionType::get(
      //PointerType::get(Type::getVoidTy(*TheContext), 0),
      Type::getFloatTy(*TheContext),
      {PointerType::get(Type::getInt8Ty(*TheContext), 0), Type::getInt32Ty(*TheContext)},
      false // Not vararg
  );
  Function::Create(
    FirstArgOnDemandTy,
    Function::ExternalLinkage,
    "FirstArgOnDemand",
    TheModule.get()
  );
  



  // char *, int
  FunctionType *DimnishFirstArgOnDemandTy = FunctionType::get(
      //PointerType::get(Type::getVoidTy(*TheContext), 0),
      Type::getFloatTy(*TheContext),
      {PointerType::get(Type::getInt8Ty(*TheContext), 0), Type::getInt32Ty(*TheContext)},
      false // Not vararg
  );
  Function::Create(
    DimnishFirstArgOnDemandTy,
    Function::ExternalLinkage,
    "DimnishFirstArgOnDemand",
    TheModule.get()
  );
  

  // char *, char *
  FunctionType * ConcatStrTy = FunctionType::get(
      //PointerType::get(Type::getVoidTy(*TheContext), 0),
      PointerType::get(Type::getInt8Ty(*TheContext), 0),
      {PointerType::get(Type::getInt8Ty(*TheContext), 0), PointerType::get(Type::getInt8Ty(*TheContext), 0)},
      false // Not vararg
  );
  Function::Create(
    ConcatStrTy,
    Function::ExternalLinkage,
    "ConcatStr",
    TheModule.get()
  );


  // char *, char *
  FunctionType *ConcatFirstArgToVarNameTy = FunctionType::get(
      PointerType::get(Type::getInt8Ty(*TheContext), 0),
      //{PointerType::get(Type::getInt8Ty(*TheContext), 0), PointerType::get(Type::getInt8Ty(*TheContext), 0)},
      {PointerType::get(Type::getInt8Ty(*TheContext), 0)},
      false // Not vararg
  );
  Function::Create(
    ConcatFirstArgToVarNameTy,
    Function::ExternalLinkage,
    "ConcatFirstArgToVarName",
    TheModule.get()
  );


  // char *, float
  FunctionType *StoreOnDemandTy = FunctionType::get(
      Type::getFloatTy(*TheContext),
      {PointerType::get(Type::getInt8Ty(*TheContext), 0), Type::getFloatTy(*TheContext)},
      false // Not vararg
  );
  Function::Create(
    StoreOnDemandTy,
    Function::ExternalLinkage,
    "StoreOnDemand",
    TheModule.get()
  );

  // char *
  FunctionType *LoadOnDemandTy = FunctionType::get(
      //PointerType::get(Type::getVoidTy(*TheContext), 0),
      Type::getFloatTy(*TheContext),
      {PointerType::get(Type::getInt8Ty(*TheContext), 0)},
      false // Not vararg
  );
  Function::Create(
    LoadOnDemandTy,
    Function::ExternalLinkage,
    "LoadOnDemand",
    TheModule.get()
  );



  // char *
  FunctionType *StoreDimsOnDemandTy = FunctionType::get(
      //PointerType::get(Type::getVoidTy(*TheContext), 0),
      Type::getFloatTy(*TheContext),
      {Type::getFloatTy(*TheContext)},
      //{PointerType::get(Type::getInt8Ty(*TheContext), 0)},
      false // Not vararg
  );
  Function::Create(
    StoreDimsOnDemandTy,
    Function::ExternalLinkage,
    "StoreDimsOnDemand",
    TheModule.get()
  );


  // char *, int
  FunctionType *CreateTensorOnDemandTy = FunctionType::get(
      //PointerType::get(Type::getVoidTy(*TheContext), 0),
      Type::getFloatTy(*TheContext),
      {PointerType::get(Type::getInt8Ty(*TheContext), 0),
       Type::getInt32Ty(*TheContext),
       PointerType::get(Type::getInt8Ty(*TheContext), 0)},
      false // Not vararg
  );
  Function::Create(
    CreateTensorOnDemandTy,
    Function::ExternalLinkage,
    "CreateTensorOnDemand",
    TheModule.get()
  );




  // float, char *
  FunctionType *CallToStoredValuesTy = FunctionType::get(
      PointerType::get(Type::getFloatTy(*TheContext), 0),
      {Type::getFloatTy(*TheContext), PointerType::get(Type::getInt8Ty(*TheContext), 0)}, 
      false 
  );
  Function::Create(
    CallToStoredValuesTy,
    Function::ExternalLinkage, 
    "toStoredValues", 
    TheModule.get() 
  );


  // char *
  FunctionType *temporaryCudaResult_AttrTy = FunctionType::get(
      Type::getFloatTy(*TheContext),
      {PointerType::get(Type::getInt8Ty(*TheContext), 0)}, 
      false 
  );
  Function::Create(
    temporaryCudaResult_AttrTy,
    Function::ExternalLinkage, 
    "temporaryCudaResult_Attr", 
    TheModule.get() 
  );


  // char *
  FunctionType *printTTy = FunctionType::get(
      Type::getFloatTy(*TheContext),
      {PointerType::get(Type::getInt8Ty(*TheContext), 0)}, 
      false 
  );
  Function::Create(
    printTTy,
    Function::ExternalLinkage, 
    "PrintTensor", 
    TheModule.get() 
  );
  

}

ThreadSafeModule irgenAndTakeOwnership(FunctionAST &FnAST,
                                       const std::string &Suffix) {
  if (auto *F = FnAST.codegen()) {
    F->setName(F->getName() + Suffix);
    auto TSM = ThreadSafeModule(std::move(TheModule), std::move(TheContext));
    // Start a new module.
    InitializeModule();
    return TSM;
  } else
    report_fatal_error("Não foi possível compilar a função JIT de forma lazy");
}

ThreadSafeModule irgenAndTakeOwnershipClass(ClassAST &FnAST,
                                       const std::string &Suffix) {
  if (auto *F = FnAST.codegen()) {
    F->setName(F->getName() + Suffix);
    auto TSM = ThreadSafeModule(std::move(TheModule), std::move(TheContext));
    // Start a new module.
    InitializeModule();
    return TSM;
  } else
    report_fatal_error("Não foi possível compilar a função JIT de forma lazy");
}



static void HandleClass() {
  

  ParseClass();

}

static void HandleDefinition() {
  
  if (auto FnAST = ParseDefinition()) {
    FunctionProtos[FnAST->getProto().getName()] =
      std::make_unique<PrototypeAST>(FnAST->getProto());
    ExitOnErr(TheJIT->addAST(std::move(FnAST)));
  } else {
    // Skip token for error recovery.
    getNextToken();
  }
}

static void HandleExtern() {
  if (auto ProtoAST = ParseExtern()) {
    if (auto *FnIR = ProtoAST->codegen()) {
      fprintf(stderr, "Ler extern: ");
      FnIR->print(errs());
      fprintf(stderr, "\n");
      FunctionProtos[ProtoAST->getName()] = std::move(ProtoAST);
    }
  } else {
    // Skip token for error recovery.
    getNextToken();
  }
}

static void HandleTopLevelExpression() {
  // Evaluate a top-level expression into an anonymous function.
  if (auto FnAST = ParseTopLevelExpr()) {
    if (FnAST->codegen()) {
      // Create a ResourceTracker for memory managment
      // anonymous expression -- that way we can free it after executing.
      auto RT = TheJIT->getMainJITDylib().createResourceTracker();

      auto TSM = ThreadSafeModule(std::move(TheModule), std::move(TheContext));
      ExitOnErr(TheJIT->addModule(std::move(TSM), RT));
      // Add IR module

      InitializeModule();

      // Points __anon_expr
      auto Sym = ExitOnErr(TheJIT->lookup("__anon_expr"));

      // Get the symbol's address and cast it to the right type (takes no
      // arguments, returns a float) so we can call it as a native function.
      auto *FP = Sym.getAddress().toPtr<float (*)()>();
      auto fp = FP();
      
      //std::cout << "\nResult times 5 is " << fp*5 << "\n";
      fprintf(stderr, "%.2f\n", fp);

      // Delete the anonymous expression module from the JIT.
      ExitOnErr(RT->remove());
    }
  } else {
    // Skip token for error recovery.
    getNextToken();
  }
}

/// top ::= definition | external | expression | ';'
static void MainLoop() {
  while (true) {
    //if (CurTok!=tok_space)
    //  std::cout << "MAIN LOOP, reading token: " << CurTok << "\n";
    switch (CurTok) {
    case tok_eof:
      return;
    case ';': // ignore top-level semicolons.
      getNextToken();
      break;
    case tok_space:
      getNextToken();
      break;
    case tok_tab:
      LogError("Tab inesperado encontrado\n");
      break;
    case tok_def:
      HandleDefinition();
      break;
    case tok_class:
      HandleClass();
      break;
    case tok_extern:
      HandleExtern();
      break;
    //case (tok_space || 59):
    default:
      HandleTopLevelExpression();
      break;
    }
  }
}


//===----------------------------------------------------------------------===//
// "Library" functions that can be "extern'd" from user code.
//===----------------------------------------------------------------------===//

/// putchard - putchar that takes a float and returns 0.
extern "C" float putchard(float X) {
  fputc((char)X, stderr);
  return 0;
}

/// printd - printf that takes a float prints it as "%f\n", returning 0.
extern "C" float printd(float X) {
  fprintf(stderr, "%f\n", X);
  return 0;
}

//===----------------------------------------------------------------------===//
// Main driver code.
//===----------------------------------------------------------------------===//

int main() {

  int deviceIdx = 0;
  cudaCheck(cudaSetDevice(deviceIdx));
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, deviceIdx);
  printf("Device %d: %s\n", deviceIdx, deviceProp.name);


  cublasCheck(cublasCreate(&cublas_handle));
  cublasCheck(cublasLtCreate(&cublaslt_handle));


  int enable_tf32 = deviceProp.major >= 8 ? 1 : 0;
  //printf("enable_tf32: %d\n", enable_tf32);
  cublas_compute_type = enable_tf32 ? CUBLAS_COMPUTE_32F_FAST_TF32 : CUBLAS_COMPUTE_32F;
  cublasMath_t cublas_math_mode = enable_tf32 ? CUBLAS_TF32_TENSOR_OP_MATH : CUBLAS_DEFAULT_MATH;
  cublasCheck(cublasSetMathMode(cublas_handle, cublas_math_mode));
  // setup the (global) cuBLASLt workspace
  cudaCheck(cudaMalloc(&cublaslt_workspace, cublaslt_workspace_size));

  InitializeNativeTarget();
  InitializeNativeTargetAsmPrinter(); // Prepare for target hardware
  InitializeNativeTargetAsmParser();

  // Install standard binary operators.
  // 1 is lowest precedence.
  BinopPrecedence[tok_space] = 1;
  BinopPrecedence[':'] = 9;
  BinopPrecedence['='] = 4;
  BinopPrecedence['>'] = 10;
  BinopPrecedence['<'] = 10;
  BinopPrecedence['+'] = 20;
  BinopPrecedence['-'] = 20;
  BinopPrecedence['/'] = 39;
  BinopPrecedence['*'] = 40;  // highest.
  BinopPrecedence['^'] = 50;
  BinopPrecedence['@'] = 60;

  // Prime the first token.
  //fprintf(stderr, "ready> ");
  getNextToken();

  TheJIT = ExitOnErr(KaleidoscopeJIT::Create());
  InitializeModule();

  // Run the main "interpreter loop" now.
  MainLoop();

  return 0;
}


/*
//forward
cublasSgemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, OC, B, C, &alpha, weight, C, inp,  C,  &beta, out,     OC);

//backward to input
cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, C, B, OC, &alpha, weight, C, dout, OC, &beta, dinp,     C)
//backward to weight
cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T, C, OC, , &alpha, inp,    C, dout, OC, &beta, dweight,  C)
*/

