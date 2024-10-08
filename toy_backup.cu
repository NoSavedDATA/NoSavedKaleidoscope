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
#include <string>
#include <utility>
#include <vector>

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cublasLt.h>
#include <omp.h>

#include "include/KaleidoscopeJIT.h"
#include "include/cu_commons.h"

using namespace llvm;
using namespace llvm::orc;

//===----------------------------------------------------------------------===//
// Lexer
//===----------------------------------------------------------------------===//

// The lexer returns tokens [0-255] if it is an unknown character, otherwise one
// of these for known things.
enum Token {
  tok_eof = -1,

  // commands
  tok_def = -2,
  tok_extern = -3,

  // primary
  tok_identifier = -4,
  tok_number = -5,

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
  tok_tensor = -16
};

static std::string IdentifierStr; // Filled in if tok_identifier
static float NumVal;             // Filled in if tok_number

/// get_token - Return the next token from standard input.
static int get_token() {
  static int LastChar = ' ';

  // Skip any whitespace and backspace.
  //while (LastChar==32 || LastChar==tok_tab)
  while (LastChar==32 || LastChar==tok_tab)
    LastChar = getchar();
  //while (isspace(LastChar))
    
    

  if (isalpha(LastChar)) { // identifier: [a-zA-Z][a-zA-Z0-9]*
    IdentifierStr = LastChar;
    while (isalnum((LastChar = getchar())))
      IdentifierStr += LastChar;

    if (IdentifierStr == "def")
      return tok_def;
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
    if (IdentifierStr == "tensor")
      return tok_tensor;
    return tok_identifier;
  }

  if (isdigit(LastChar) || LastChar == '.') { // Number: [0-9.]+
    std::string NumStr;
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
    return tok_space;

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
  std::string Type = "None";
  std::string Name = "Unnamed";

  virtual Value *codegen() = 0;
  
  /*virtual void setType(std::string Type) {
    this->Type=Type;
  }*/
  virtual std::string GetName() {
    return Name;
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


class CudaNumExprAST : public ExprAST {
  std::unique_ptr<ExprAST> LHS, RHS;

  public:
    CudaNumExprAST(std::unique_ptr<ExprAST> LHS, std::unique_ptr<ExprAST> RHS)
        : LHS(std::move(LHS)), RHS(std::move(RHS)) {}

  Value *codegen() override;
};


/// VariableExprAST - Expression class for referencing a variable, like "a".
class VariableExprAST : public ExprAST {
  std::string Name;

  public:
    VariableExprAST(const std::string &Name) : Name(Name) {}

    const std::string &getName() const { return Name; }
    
    Value *codegen() override;
};


/// VarExprAST - Expression class for var
class VarExprAST : public ExprAST {
  std::vector<std::pair<std::string, std::unique_ptr<ExprAST>>> VarNames;
  std::unique_ptr<ExprAST> Body;
  std::string Type;

  public:
    VarExprAST(
        std::vector<std::pair<std::string, std::unique_ptr<ExprAST>>> VarNames,
        std::unique_ptr<ExprAST> Body,
        std::string Type)
        : VarNames(std::move(VarNames)), Body(std::move(Body)), Type(Type) {}

  Value *codegen() override;
};

class TensorExprAST : public VarExprAST {
  std::vector<int> Dims;

  public:
    TensorExprAST(
        std::vector<std::pair<std::string, std::unique_ptr<ExprAST>>> VarNames,
        std::unique_ptr<ExprAST> Body,
        std::string Type)
        : VarNames(std::move(VarNames)), Body(std::move(Body)), Type(Type) {}

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

/// CallExprAST - Expression class for function calls.
class CallExprAST : public ExprAST {
  std::string Callee;
  std::vector<std::unique_ptr<ExprAST>> Args;

public:
  CallExprAST(const std::string &Callee,
              std::vector<std::unique_ptr<ExprAST>> Args)
      : Callee(Callee), Args(std::move(Args)) {}

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
  bool IsOperator;
  unsigned Precedence; // Precedence if a binary op.

  public:
    PrototypeAST(const std::string &Name, std::vector<std::string> Args,
                bool IsOperator = false, unsigned Prec = 0)
        : Name(Name), Args(std::move(Args)), IsOperator(IsOperator),
          Precedence(Prec) {}

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
std::unique_ptr<ExprAST> LogError(const char *Str) {
  fprintf(stderr, "Erro: %s\n", Str);
  return nullptr;
}

// Modified LogError function with token parameter
std::unique_ptr<ExprAST> LogErrorT(int CurTok) {
  char buf[100];
  snprintf(buf, sizeof(buf), "token %d inesperado.", CurTok);
  fprintf(stderr, "Erro: %s\n", buf);
  return nullptr;
}

std::unique_ptr<PrototypeAST> LogErrorP(const char *Str) {
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

/// identifierexpr
///   ::= identifier
///   ::= identifier '(' expression* ')'
static std::unique_ptr<ExprAST> ParseIdentifierExpr(int tabcount=0) {
  std::string IdName = IdentifierStr;

  getNextToken(); // eat identifier.
  
  if (CurTok != '(') // Simple variable ref.
    return std::make_unique<VariableExprAST>(IdName);

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

  return std::make_unique<CallExprAST>(IdName, std::move(Args));
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
static std::unique_ptr<ExprAST> ParseVarExpr(std::string Type) {
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


static std::unique_ptr<ExprAST> ParseTensorExpr(std::string Type) {
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

  return std::make_unique<TensorExprAST>(std::move(VarNames), std::move(Body), "tensor");
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
  case tok_number:
    return ParseNumberExpr(tabcount);
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
  case tok_space:
    getNextToken();
    return ParsePrimary();
  }
}

/// unary
///   ::= primary
///   ::= '!' unary
static std::unique_ptr<ExprAST> ParseUnary(int tabcount=0) {
  //std::cout << "Parse Unary";
  
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
static std::unique_ptr<ExprAST> ParseBinOpRHS(int ExprPrec,
                                              std::unique_ptr<ExprAST> LHS,
                                              int tabcount=0) {
  
  // If this is a binop, find its precedence.
  int RhsTok = 0;
  int LhsTok = 0;
  std::string LName, RName;
  while (true) {
    int TokPrec = get_tokenPrecedence();

    // If this is a binop that binds at least as tightly as the current binop,
    // consume it, otherwise we are done.
    

    if (TokPrec==BinopPrecedence[':'])
    {
      getNextToken();
      return LHS;
    }
    if (TokPrec < ExprPrec)
      return LHS;

    
    int BinOp = CurTok;

    if(CurTok==':')
    {
      getNextToken();
      return LHS;
    }

    if (CurTok==')')
      return LHS;

    
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

    
    if((BinOp==tok_space) && (!( (CurTok==tok_identifier) || (CurTok==tok_number) )))
    {
      //std::cout << "SPACE WITHOUT NUMBER OR VAR " << CurTok << "\n";
      return LHS;
    }
    
    


    auto RHS = ParseUnary(); // Returns an identifier, number or expression result
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
      return nullptr;
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
      return RHS;
    } else {


      // If BinOp binds less tightly with RHS than the operator after RHS, let
      // the pending operator take RHS as its LHS.
      int NextPrec = get_tokenPrecedence();
        
      // || ((seen_tabs<tabcount)&&(seen_tabs>0))
      if (TokPrec < NextPrec){
        //std::cout << NextPrec << " Next Prec\n";
        RHS = ParseBinOpRHS(TokPrec + 1, std::move(RHS), tabcount);
        //std::cout << "Error after RHS parse \n";
        if (!RHS)
        {
          //std::cout << "RETURNING NULL Recursive Bin Op \n";
          return nullptr;
        }
      }

        
      //std::cout << LhsTok << " " << BinOp << " " << RhsTok << "\n" << CurTok <<  " " << RName << "\n\n";
      if(BinOp==64) // @
        LHS = std::make_unique<CudaNumExprAST>(std::move(LHS), std::move(RHS));
        //LHS = std::make_unique<CudaNumExprAST>(LHS, RHS);
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

  return ParseBinOpRHS(0, std::move(LHS), tabcount);
}

/// prototype
///   ::= id '(' id* ')'
///   ::= binary LETTER number? (id, id)
///   ::= unary LETTER (id)
static std::unique_ptr<PrototypeAST> ParsePrototype() {
  std::string FnName;

  unsigned Kind = 0; // 0 = identifier, 1 = unary, 2 = binary.
  unsigned BinaryPrecedence = 30;

  switch (CurTok) {
  default:
    return LogErrorP("Esperado nome da função no protótipo");
  case tok_identifier:
    FnName = IdentifierStr;
    Kind = 0;
    getNextToken();
    break;
  case tok_unary:
    getNextToken();
    if (!isascii(CurTok))
      return LogErrorP("Esperado operador unário");
    FnName = "unary";
    FnName += (char)CurTok;
    Kind = 1;
    getNextToken();
    break;
  case tok_binary:
    getNextToken();
    if (!isascii(CurTok))
      return LogErrorP("Esperado operador binário");
    FnName = "binary";
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

  std::vector<std::string> ArgNames;
  while (getNextToken() == tok_identifier)
    ArgNames.push_back(IdentifierStr);
  if (CurTok != ')')
    return LogErrorP("Esperado ')' no protótipo");

  // success.
  getNextToken(); // eat ')'.

  // Verify right number of names for operator.
  if (Kind && ArgNames.size() != Kind)
    return LogErrorP("Número inválido de operandos para o operador");

  return std::make_unique<PrototypeAST>(FnName, ArgNames, Kind != 0,
                                         BinaryPrecedence);
}

/// definition ::= 'def' prototype expression
static std::unique_ptr<FunctionAST> ParseDefinition() {
  getNextToken(); // eat def.
  auto Proto = ParsePrototype();
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
static std::unique_ptr<IRBuilder<>> Builder;
static std::unique_ptr<Module> TheModule;
static std::map<std::string, AllocaInst *> NamedValues;
static std::map<std::string, float> StoredValues;
static std::map<std::string, float *> NamedTensors;
static std::map<std::string, std::unique_ptr<PrototypeAST>> FunctionProtos;
static ExitOnError ExitOnErr;

Value *LogErrorV(const char *Str) {
  LogError(Str);
  return nullptr;
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
  return ConstantFP::get(*TheContext, APFloat(Val));
}

__global__ void vec_mult(float a, float* x, float* y) {
  //y[threadIdx.x] = a * x[threadIdx.x];
  *y = a * *x;
}



float currentCudaResult[100];
int used_cuda = 0;



extern "C" float CudaMult(char *tensorName, float R, int _used_cuda) {
  std::cout << "Cuda Mult Here! " << tensorName << " used cuda " << _used_cuda << " R " << R << "\n";
  //float R = cast<ConstantFP>(r)->getValueAPF().convertToFloat();
  float *L;

  
  if (_used_cuda==1)
    L = new float(currentCudaResult[0]);
  else
    L = NamedTensors[tensorName];


  std::cout << "Cuda in is: " << *L << "\n";
  int kDataLen = 1;


  float* device_x;
  cudaMalloc(&device_x, kDataLen * sizeof(float));
  cudaMemcpy(device_x, L, kDataLen * sizeof(float), cudaMemcpyHostToDevice);
  

  float* device_y;
  cudaMalloc(&device_y, kDataLen * sizeof(float));
  cudaMemcpy(device_x, L, kDataLen * sizeof(float),
             cudaMemcpyHostToDevice);


  // Launch the kernel.
  vec_mult<<<1, kDataLen>>>(R, device_x, device_y);
  //vec_mult<<<1, kDataLen>>>(R, device_x, device_y);

  cudaDeviceSynchronize();
  cudaMemcpy(currentCudaResult, device_y, kDataLen * sizeof(float),
             cudaMemcpyDeviceToHost);

  // Print the results.
  for (int i = 0; i < kDataLen; ++i) {
    std::cout << "y[" << i << "] = " << currentCudaResult[i] << "\n";
  }
  std::cout << "\n";


  //return ConstantFP::get(*TheContext, APFloat(currentCudaResult[0]));
  return 0;
}



Value *VariableExprAST::codegen() {
  // Look this variable up in the function.

  //std::cout << "Var Code Gen \n";


  if (NamedValues.count(Name) != 0) 
  {
    Type="var";
    Value *V = NamedValues[Name];
    std::cout << "Now Loading Var "<< Name <<" to Context" << "  \n";
    

    return Builder->CreateLoad(Type::getFloatTy(*TheContext), V, Name.c_str());

  } else {
    Type="tensor";
    float *v = NamedTensors[Name];
    if (!v)
    {
      std::cout << "Erro: O nome do tensor/variável " << Name << " é desconhecido\n";
      return LogErrorV(" ");
      //return LogErrorV("O nome do tensor/variável é desconhecido");
    }
      

    std::cout << "Now Loading Tensor " << Name << ": " << *v << " to Context \n";
    
    // float_to_value
    Value *V = ConstantFP::get(*TheContext, APFloat(0.0f));
    return V;
  }
}



extern "C" float toStoredValues(float Val, char * name_to_store)
{
  std::cout << "ULULULULULULULLULULULULULULULULU " << name_to_store << " \n";
  std::cout << typeid(Val).name() << std::endl;

  StoredValues[name_to_store] = Val;
  
  std::cout << Val << "stored\n";
  return 0;
}


extern "C" float temporaryCudaResult_Attr(char *name_to_store)
{
  NamedTensors[name_to_store] = new float(currentCudaResult[0]);
  return 0;
}

//Function *callCudaMult, *CallToStoredValues;

Value *BinaryExprAST::codegen() {
  // Special case '=' because we don't want to emit the LHS as an expression.
  if (Op == '=') {
    
    // Assignment requires the LHS to be an identifier.
    // This assume we're building without RTTI because LLVM builds that way by
    // default.  If you build LLVM with RTTI this can be changed to a
    // dynamic_cast for automatic error checking.
    VariableExprAST *LHSE = static_cast<VariableExprAST *>(LHS.get());
    if (!LHSE)
      return LogErrorV("Destino do '=' deve ser uma variável ou operação.");
    // Codegen the RHS.
    
    Value *Val = RHS->codegen();
    if (!Val)
      return nullptr;

    // Look up the name.
    if (NamedValues.count(LHSE->getName()) != 0) {
      
      Value *Variable = NamedValues[LHSE->getName()];
      
      if (!Variable)
        return LogErrorV("O nome da variável é desconhecido.");

      Builder->CreateStore(Val, Variable);
    

      /*
      Function *CallToStoredValuesFn = TheModule->getFunction("toStoredValues");

      Value *valStr = Builder->CreateGlobalString(LHSE->getName());
      Builder->CreateCall(CallToStoredValuesFn, {Val, valStr});
      */
      
      
    } else {
      float Variable = *NamedTensors[LHSE->getName()];
      if (!Variable)
        return LogErrorV("O nome do tensor/variável é desconhecido.");
      
      if (used_cuda)
      {
        
        Value *valStr = Builder->CreateGlobalString(LHSE->getName());
        Function *temporaryCudaResult_AttrFn = TheModule->getFunction("temporaryCudaResult_Attr");
        Builder->CreateCall(temporaryCudaResult_AttrFn, {valStr});
        
        used_cuda=0;
      }
      
    }
    return Val;
  }


  

  Value *L = LHS->codegen();
  Value *R = RHS->codegen();

  //std::cout << "LHS type " << LHS->Type << "\nRHS type " << RHS->Type << "\n\n";

  if ((RHS->Type=="tensor") && (LHS->Type!="tensor"))
  {
    std::unique_ptr<ExprAST> aux = std::move(LHS);
    LHS = std::move(RHS);
    RHS = std::move(aux);

    Value *aux_ = std::move(L);
    L = std::move(R);
    R = std::move(aux_);
  }

  

  if (!L || !R)
    return nullptr;

  if ((LHS->Type=="tensor") || (used_cuda))
  {  

    std::cout << "Deciding CUDA Operation for: " << LHS->GetName() << " \n";


    //std::cout << "L value: " << val1 << " | R value: " << r  << ".\n";

    Function *CudaFn;

  
    Value *tensorName = Builder->CreateGlobalString(LHS->GetName());

    Value *used_cuda_aux = ConstantInt::get(Type::getInt32Ty(*TheContext), used_cuda);
    used_cuda = 1;

    switch (Op)
    {
    case ':':
      return L;
    case tok_space:
      return R;
    case '+':
      return Builder->CreateFAdd(L, R, "addtmp");
    case '*':
      CudaFn = TheModule->getFunction("CudaMult");
      return Builder->CreateCall(CudaFn, {tensorName, R, used_cuda_aux}, "cudamult");
    default:
      break;
    }
  } else {

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
  if (Type=="var")
    for (unsigned i = 0, e = VarNames.size(); i != e; ++i)
      NamedValues[VarNames[i].first] = OldBindings[i];

  // Return the body computation.
  return BodyVal;
}


Value *TensorExprAST::codegen() {
  std::vector<AllocaInst *> OldBindings;

  Function *TheFunction = Builder->GetInsertBlock()->getParent();

  // Register all variables and emit their initializer.
  for (unsigned i = 0, e = VarNames.size(); i != e; ++i) {
    const std::string &VarName = VarNames[i].first;
    ExprAST *Init = VarNames[i].second.get();


    Value *InitVal;
    if (Init) {
      InitVal = Init->codegen();
      if (!InitVal)
        return nullptr;
    } else { // If not specified, use 0.0.
      InitVal = ConstantFP::get(*TheContext, APFloat(0.0));
    }

    NamedTensors[VarName] = make_random_float(1);
  }

  // Codegen the body that is contained by the in expression
  Value *BodyVal = Body->codegen();
  if (!BodyVal)
    return nullptr;

  // Pop all our variables from scope.
  if (Type=="var")
    for (unsigned i = 0, e = VarNames.size(); i != e; ++i)
      NamedValues[VarNames[i].first] = OldBindings[i];


  return BodyVal;
}



Value *CallExprAST::codegen() {
  // Look up the name in the global module table.
  Function *CalleeF = getFunction(Callee);
  if (!CalleeF)
    return LogErrorV("Função referenciada desconhecida");

  // If argument mismatch error.
  if (CalleeF->arg_size() != Args.size())
    return LogErrorV("Incorrect # arguments passed");

  std::vector<Value *> ArgsV;
  float val;
  for (unsigned i = 0, e = Args.size(); i != e; ++i) {
    
    ArgsV.push_back(Args[i]->codegen());


    if (!ArgsV.back())
      return nullptr;
  }

  return Builder->CreateCall(CalleeF, ArgsV, "calltmp");
}

Function *PrototypeAST::codegen() {
  // Make the function type:  float(float,float) etc.
  std::vector<Type *> Floats(Args.size(), Type::getFloatTy(*TheContext));
  FunctionType *FT =
      FunctionType::get(Type::getFloatTy(*TheContext), Floats, false);

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
  // TODO: It's creating one initialize for each ";" (top level expression).

  // Create a new builder for the module.
  Builder = std::make_unique<IRBuilder<>>(*TheContext);

  Type *floatPtrType = PointerType::get(Type::getFloatTy(*TheContext), 0);

  
  // Function takes two float pointers and returns a float pointer
  FunctionType *CudaMultTy = FunctionType::get(
      Type::getFloatTy(*TheContext), // Return type: pointer to float
      {PointerType::get(Type::getInt8Ty(*TheContext), 0), Type::getFloatTy(*TheContext), Type::getInt32Ty(*TheContext)}, // char *, float, int
      false // Not vararg
  );

  Function::Create(
    CudaMultTy,
    Function::ExternalLinkage, // Linkage (e.g., external for linking with other modules)
    "CudaMult", // Function name
    TheModule.get() // Module to which the function belongs
  );



  FunctionType *CallToStoredValuesTy = FunctionType::get(
      Type::getFloatTy(*TheContext),
      {Type::getFloatTy(*TheContext), PointerType::get(Type::getInt8Ty(*TheContext), 0)}, // float, char *
      false 
  );
  
  Function::Create(
    CallToStoredValuesTy,
    Function::ExternalLinkage, 
    "toStoredValues", 
    TheModule.get() 
  );



  FunctionType *temporaryCudaResult_AttrTy = FunctionType::get(
      Type::getFloatTy(*TheContext),
      {PointerType::get(Type::getInt8Ty(*TheContext), 0)}, // char *
      false 
  );
  
  Function::Create(
    temporaryCudaResult_AttrTy,
    Function::ExternalLinkage, 
    "temporaryCudaResult_Attr", 
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
  InitializeNativeTarget();
  InitializeNativeTargetAsmPrinter(); // Prepare for target hardware
  InitializeNativeTargetAsmParser();

  // Install standard binary operators.
  // 1 is lowest precedence.
  BinopPrecedence[tok_space] = 1;
  BinopPrecedence[':'] = 9;
  BinopPrecedence['='] = 2;
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
