// JIT
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



#include "include/KaleidoscopeJIT.h"



#include <cudnn.h>
#include <algorithm>
#include <cstdarg>
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
#include <utility>
#include <vector>
#include <iomanip>
#include <math.h>
#include <fenv.h>
#include <tuple>
#include <glob.h>
#include <chrono>
#include <thread>
#include <random>

// Cuda
#include <stdio.h>
#include <stdlib.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cublasLt.h>
#include <omp.h>

#include "include/cu_commons.h"

#define GELU_SCALING_FACTOR sqrtf(2.0f / M_PI)


// Files
#define STB_IMAGE_IMPLEMENTATION
#include "include/stb/stb_image.h"


static cublasHandle_t cublas_handle;
static cublasLtHandle_t cublaslt_handle;
cudnnHandle_t cudnn;

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

#define checkCUDNN(expression)                               \
  {                                                          \
    cudnnStatus_t status = (expression);                     \
    if (status != CUDNN_STATUS_SUCCESS) {                    \
      std::cerr << "Error on line " << __LINE__ << ": "      \
                << cudnnGetErrorString(status) << std::endl; \
      std::exit(EXIT_FAILURE);                               \
    }                                                        \
  }

// Error Colors

// \033[0m default
// \033[31m red
// \033[33m yellow
// \033[95m purple


bool ends_with(std::string str_input, std::string str_end)
{
  return str_input.size() >= str_end.size() && str_input.compare(str_input.size() - str_end.size(), str_end.size(), str_end) == 0;
}

bool starts_with(const char* str, const char* sub) {
  return strncmp(str, sub, strlen(sub)) == 0;
}

std::vector<std::string> split_str(const std::string& str, char delimiter) {
  std::vector<std::string> tokens;
  std::string token;
  std::istringstream stream(str);

  while (std::getline(stream, token, delimiter)) {
    tokens.push_back(token);
  }

  return tokens;
}

std::vector<std::string> split(const char* input, const std::string& delimiter) {
    std::vector<std::string> tokens;
    size_t start = 0, end = 0;
    while ((end = std::string(input + start).find(delimiter)) != std::string::npos) {
        tokens.push_back(std::string(input + start, end));
        start += end + delimiter.length();
    }
    tokens.push_back(std::string(input + start));
    return tokens;
}


bool in_char(char ch, const std::vector<char>& list) {
  // Use std::find to efficiently search the list for the character
  return std::find(list.begin(), list.end(), ch) != list.end();
}

bool in_str(std::string str, std::vector<std::string> list) {
    return std::find(list.begin(), list.end(), str) != list.end();
}


std::vector<char> ops = {'+', '-', '*', '/', '@', '=', '>', '<', 10, -14, ',', '(', ')', ';'};

std::vector<std::string> tensor_methods = {"view","permute", "onehot", "mean", "sum", "max", "min"};
std::vector<std::string> vararg_methods = {"view", "Datasetyield"};
std::vector<std::string> tensor_resulting_methods = {"gelu", "relu", "softmax"};
std::vector<std::string> activation_functions = {"gelu", "relu", "softmax"};

std::vector<std::string> preprocessing_names = {"load_img", "split_str_to_float"};
std::vector<std::string> tensor_inits = {"randint", "randu", "zeros", "ones", "xavu", "xavu_relu", "xavn"};




PointerType *floatPtrTy, *int8PtrTy;

bool ShallCodegen = true;

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
  tok_async = -22,
  tok_async_finish = -23,
  tok_tab = 9,

  // operators
  tok_binary = -11,
  tok_unary = -12,


  tok_space = -14,

  
  // var definition
  tok_var = -15,
  tok_tensor = -16,
  tok_var_str = -17,
  tok_attr_var = -18,
  tok_attr_tensor = -19,
  tok_preprocessing = -20,
  tok_conv2d = -21,

  // function ops
  tok_log = -30
};

std::map<int, std::string> token_to_string = {
  { tok_eof, "eof" },

  // functions/classes
  { tok_def, "def" },
  { tok_class, "class" },
  { tok_self, "self" },
  { tok_class_attr, "class attr" },
  { tok_extern, "extern" },

  // primary
  { tok_identifier, "tok identifier" },
  { tok_number, "tok number" },
  { tok_str, "tok str" },

  // control
  { tok_if, "if" },
  { tok_then, "then" },
  { tok_else, "else" },
  { tok_for, "for" },
  { tok_while, "while" },
  { tok_async, "async" },
  { tok_async_finish, "finish" },
  { tok_tab, "tok tab" },

  // operators
  { tok_binary, "tok binary" },
  { tok_unary,"tok unary" },


  { tok_space, "tok_space" },

  
  // var definition
  { tok_var, "var" },
  { tok_tensor, "tensor" },
  { tok_var_str, "var str" },
  { tok_attr_var, "tok attr var" },
  { tok_attr_tensor, "tok attr tensor" },
  { tok_preprocessing, "tok preprocessing" },
  { tok_conv2d, "Conv2d" },

  { 10, "tok space"},

  { 42, "*" },
  { 43, "+" },
  { 44, "," },
  { 45, "-" },
  { 47, "/" },

  { 48, "0" },
  { 49, "1" },
  { 50, "2" },
  { 51, "3" },
  { 52, "4" },
  { 53, "5" },
  { 54, "6" },
  { 55, "7" },
  { 56, "8" },
  { 57, "9" },
  { 58, ":" },
  { 59, ";" },
  { 60, "<" },
  { 61, "=" },
  { 62, ">" },
  { 64, "@" },


  { static_cast<int>('a'), "a" },
  { static_cast<int>('b'), "b" },
  { static_cast<int>('c'), "c" },
  { static_cast<int>('d'), "d" },
  { static_cast<int>('e'), "e" },
  { static_cast<int>('f'), "f" },
  { static_cast<int>('g'), "g" },
  { static_cast<int>('h'), "h" },
  { static_cast<int>('i'), "i" },
  { static_cast<int>('j'), "j" },
  { static_cast<int>('k'), "k" },
  { static_cast<int>('l'), "l" },
  { static_cast<int>('m'), "m" },
  { static_cast<int>('n'), "n" },
  { static_cast<int>('o'), "o" },
  { static_cast<int>('p'), "p" },
  { static_cast<int>('q'), "q" },
  { static_cast<int>('r'), "r" },
  { static_cast<int>('s'), "s" },
  { static_cast<int>('t'), "t" },
  { static_cast<int>('u'), "u" },
  { static_cast<int>('v'), "v" },
  { static_cast<int>('w'), "w" },
  { static_cast<int>('x'), "x" },
  { static_cast<int>('y'), "y" },
  { static_cast<int>('z'), "z" },

};



static std::string IdentifierStr; // Filled in if tok_identifier
static float NumVal;             // Filled in if tok_number

std::string ReverseToken(int _char)
{
  /*
  if (_char>=48 && _char<=57) // Handle number
    return std::to_string(NumVal);
  */
  if (_char==tok_identifier)
    return IdentifierStr;

  return token_to_string[_char];
}

int LineCounter;

int SeenTabs = 0;
int LastSeenTabs = 0;

/// get_token - Return the next token from standard input.
static int get_token() {
  static int LastChar = ' ';

  

  /*
  if (LastChar!=32)
    std::cout << "Pre last char: " << ReverseToken(LastChar) << "\n";
  */

  // Skip any whitespace and backspace.
  
  
  while (LastChar==32 || LastChar==tok_tab)
    LastChar = getchar();
    
    

  //std::cout << "Last char: " << LastChar << "\n";
    
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
      if (IdentifierStr == "Conv2d" && LastChar=='[')
      {
        LastChar = getchar();
        return tok_conv2d;
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
    if (IdentifierStr == "async")
      return tok_async;
    if (IdentifierStr == "finish")
      return tok_async_finish;
    if (IdentifierStr == "binary")
      return tok_binary;
    if (IdentifierStr == "unary")
      return tok_unary;
    if (IdentifierStr == "var")
      return tok_var;
    if (IdentifierStr == "log")
      return tok_log;
    if (IdentifierStr == "glob")
      IdentifierStr = "_glob_b_";
    if (IdentifierStr == "str")
      return tok_var_str;
    if (in_str(IdentifierStr,preprocessing_names))
      return tok_preprocessing;
    
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
    while (LastChar != EOF && LastChar != '\n' && LastChar != tok_space && LastChar != '\r');

    if (LastChar != EOF)
      return get_token();
  }

  // Check for end of file.  Don't eat the EOF.
  if (LastChar == EOF)
    return tok_eof;

  // Otherwise, just return the character as its ascii value.
  int ThisChar = LastChar;


  
  if (ThisChar==10 || LastChar==tok_tab)
  {
    //std::cout << "Gotcha\n";
    //std::cout << "ThisChar: " << ThisChar << " LastChar " << LastChar << "\n";
    
    while(LastChar==10 || LastChar==tok_tab) {
      if(ThisChar==10)
      {
        LineCounter += 1;
        LastSeenTabs = SeenTabs;
        SeenTabs = 0;
      }
      if (LastChar==tok_tab)
        SeenTabs+=1;

      ThisChar = (int)LastChar;
      LastChar = getchar(); 
    } 

    //std::cout << "New seen tabs: " << SeenTabs << "\n";
    return tok_space;
  }


  LastChar = getchar();
  int otherChar = LastChar;


  if((ThisChar==47)&&(otherChar == 47)){
    LastChar = getchar();
    return 77; //
  }

  //std::cout << "Post char: " << ReverseToken(ThisChar) << "\n";

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

  Value *TensorPtr, *DimsPtr;


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
  virtual Value *GetTensorPtr() {
    return TensorPtr;
  }
  virtual Value *GetDimsPtr() {
    return DimsPtr;
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
    
    std::string Type;
    VarExprAST(
        std::vector<std::pair<std::string, std::unique_ptr<ExprAST>>> VarNames,
        std::string Type)
        : VarNames(std::move(VarNames)), Type(Type) {}

  Value *codegen() override;
};

class StrExprAST : public ExprAST {

  public:
    std::vector<std::pair<std::string, std::unique_ptr<ExprAST>>> VarNames;
    std::unique_ptr<ExprAST> Body;
    std::string Type;
    StrExprAST(
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
      std::string Type,
      std::vector<std::unique_ptr<ExprAST>> V_Dims,
      const std::string &TensorInit)
      : VarExprAST(std::move(VarNames), std::move(Type)),
                   V_Dims(std::move(V_Dims)), TensorInit(TensorInit) {}

  Value *codegen() override;
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
      const std::string &TensorInit)
      : VarExprAST(std::move(VarNames), std::move(Type)),
                   C(std::move(C)), OC(std::move(OC)), Ks(std::move(Ks)),
                   Stride(std::move(Stride)), Padding(std::move(Padding)),
                   TensorInit(TensorInit) {}

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
  std::string PreDot;
  bool IsVarForward;
  std::string CalleeOverride;

  public:
    CallExprAST(const std::string &Callee,
                std::vector<std::unique_ptr<ExprAST>> Args,
                const std::string &Class,
                const std::string &PreDot,
                bool IsVarForward,
                const std::string &CalleeOverride)
        : Callee(Callee), Args(std::move(Args)), Class(Class), PreDot(PreDot), IsVarForward(IsVarForward), CalleeOverride(CalleeOverride) {}

  Value *codegen() override;
};



/// IfExprAST - Expression class for if/then/else.
class IfExprAST : public ExprAST {
  std::unique_ptr<ExprAST> Cond;
  std::vector<std::unique_ptr<ExprAST>> Then, Else;

  public:
    IfExprAST(std::unique_ptr<ExprAST> Cond,
              std::vector<std::unique_ptr<ExprAST>> Then,
              std::vector<std::unique_ptr<ExprAST>> Else)
        : Cond(std::move(Cond)), Then(std::move(Then)), Else(std::move(Else)) {}

  Value *codegen() override;
};

/// ForExprAST - Expression class for for.
class ForExprAST : public ExprAST {
  std::string VarName;
  std::unique_ptr<ExprAST> Start, End, Step;
  std::vector<std::unique_ptr<ExprAST>> Body;

  public:
    ForExprAST(const std::string &VarName, std::unique_ptr<ExprAST> Start,
              std::unique_ptr<ExprAST> End, std::unique_ptr<ExprAST> Step,
              std::vector<std::unique_ptr<ExprAST>> Body)
        : VarName(VarName), Start(std::move(Start)), End(std::move(End)),
          Step(std::move(Step)), Body(std::move(Body)) {}

  Value *codegen() override;
};

/// WhileExprAST - Expression class for while.
class WhileExprAST : public ExprAST {
	std::unique_ptr<ExprAST> Cond;
  std::vector<std::unique_ptr<ExprAST>> Body;

  public:
    WhileExprAST(std::unique_ptr<ExprAST> Cond, std::vector<std::unique_ptr<ExprAST>> Body)
      : Cond(std::move(Cond)), Body(std::move(Body)) {}

	Value* codegen() override;
};


/// AsyncExprAST - Expression class for async.
class AsyncExprAST : public ExprAST {
	std::vector<std::unique_ptr<ExprAST>> Body;

  public:
    AsyncExprAST(std::vector<std::unique_ptr<ExprAST>> Body)
      : Body(std::move(Body)) {}

	Value* codegen() override;
};


/// FinishExprAST - Expression class for finish/async.
class FinishExprAST : public ExprAST {
  std::vector<std::unique_ptr<ExprAST>> Bodies;
  std::vector<bool> IsAsync;

  public:
    FinishExprAST(std::vector<std::unique_ptr<ExprAST>> Bodies,
                  std::vector<bool> IsAsync)
            : Bodies(std::move(Bodies)), IsAsync(std::move(IsAsync)) {}


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



//===----------------------------------------------------------------------===//
// Parser
//===----------------------------------------------------------------------===//
//global

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
  ShallCodegen = false;
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


std::unique_ptr<ExprAST> LogError_toNextToken(std::string Str) {
  //fprintf(stderr, "\033[31m Erro: \033[0m%s\n", Str);
  LogErrorS(Str);

  getNextToken();
  
  return nullptr;
}


std::unique_ptr<ExprAST> LogErrorBreakLine(std::string Str) {
  //fprintf(stderr, "\033[31m Erro: \033[0m%s\n", Str);
  LogErrorS(Str);

  while(CurTok!=tok_space && CurTok!=';')
    getNextToken();

  if (CurTok==tok_space)
    getNextToken();
  
  return nullptr;
}


void LogWarning(const char *Str) {
  std::cout << "\nLinha: " << LineCounter << "\n   \033[33m Aviso: \033[0m " << Str << "\n\n";
}

// Modified LogError function with token parameter
std::unique_ptr<ExprAST> LogErrorT(int CurTok) {
  ShallCodegen = false;
  //char buf[100];
  //snprintf(buf, sizeof(buf), "token %d inesperado.", CurTok);
  //fprintf(stderr, "\033[31mErro: \033[0m%s\n", buf);
  std::cout << "\nLinha: " << LineCounter << "\n   \033[31m Erro: \033[0mtoken " << ReverseToken(CurTok) << " inesperado. Esperava-se uma expressão.\n\n";
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

Value *LogErrorV(std::string Str) {
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
  getNextToken(); // consume the "
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

//global
std::vector<std::string> tensorVars;
std::map<std::string, std::string> functionVars;


//global
static std::vector<std::string> Classes;
static std::map<std::string, std::string> Object_toClass;


/// identifierexpr
///   ::= identifier
///   ::= identifier '(' expression* ')'
static std::unique_ptr<ExprAST> ParseIdentifierExpr(int tabcount=0) {
  
  for(int i=0; i<Classes.size(); i++)
    if(IdentifierStr==Classes[i])  // Object object
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
    //std::cout << "call arg identifier type: " << aux->GetType() <<  "\n";
    
  if (CurTok==tok_space)
    getNextToken();
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

  bool is_var_forward = false;
  std::string callee_override = "none";
  if (functionVars.find(IdName) != functionVars.end())
  {
    is_var_forward = true;
    callee_override = functionVars[IdName];
  }
  
  auto aux = std::make_unique<CallExprAST>(IdName, std::move(Args), "None", "None", is_var_forward, callee_override);

  
  if (in_str(IdName, tensor_resulting_methods) || is_var_forward)
    aux->SetType("tensor");
  
  return aux;
}



/// ifexpr ::= 'if' expression 'then' expression 'else' expression
static std::unique_ptr<ExprAST> ParseIfExpr(int tabcount=1) {
  
  int cur_level_tabs = SeenTabs;

  getNextToken(); // eat the if.

  
  //std::cout << "If tabs level: " << cur_level_tabs <<  "\n";
  

  // condition.
  auto Cond = ParseExpression();
  if (!Cond)
    return nullptr;

  if(CurTok==tok_space)
    getNextToken();

  
  std::vector<std::unique_ptr<ExprAST>> Then, Else;
  
  while(true)
  {
    if (SeenTabs <= cur_level_tabs && CurTok != tok_space)
      break;

    while (CurTok == tok_space)
      getNextToken();
    
    if (SeenTabs <= cur_level_tabs)
      break;
    
    auto body = ParseExpression();
    if (!body)
      return nullptr;
    Then.push_back(std::move(body));
    
  }
  
  if (Then.size()==0)
  {
    LogError("Then is null");
    return nullptr;
  }
  
  
  if(CurTok == tok_space)
    getNextToken();

  //std::cout << "\n\nIf else token: " << ReverseToken(CurTok) <<  "\n\n\n";

  if (CurTok != tok_else) {
    Else.push_back(std::make_unique<NumberExprAST>(0));

    return std::make_unique<IfExprAST>(std::move(Cond), std::move(Then),
                                      std::move(Else));
  }
  else {
    getNextToken(); //eat else
    if(CurTok != tok_space)
      LogError("else requer barra de espaço.");
    getNextToken();

    while(true)
    {

      if (SeenTabs <= cur_level_tabs && CurTok != tok_space && CurTok != tok_tab)
        break;

      while (CurTok == tok_space)
        getNextToken();

      if (SeenTabs <= cur_level_tabs)
        break;
      
      auto body = ParseExpression();
      if (!body)
        return nullptr;
      Else.push_back(std::move(body));
    }

  
    if (CurTok==tok_space)
      getNextToken();

    return std::make_unique<IfExprAST>(std::move(Cond), std::move(Then),
                                      std::move(Else));
  }
}


/// forexpr ::= 'for' identifier '=' expr ',' expr (',' expr)? 'in' expression
static std::unique_ptr<ExprAST> ParseForExpr() {

  int cur_level_tabs = SeenTabs;

  //std::cout << "\nSeen tabs on for: " << SeenTabs << "\n\n";

  getNextToken(); // eat the for.


  if (CurTok != tok_identifier)
    return LogError("identificador da variável de controle esperado depois do for.");

  std::string IdName = IdentifierStr;
  getNextToken(); // eat identifier.

  std::unique_ptr<ExprAST> Start, End;
  Start = nullptr;

  if (CurTok != '=')
    LogError("Esperada atribuição do valor inicial do for.");
  else 
  {
    getNextToken(); // eat '='.

    auto Start = ParseExpression();
  }
  //if (!Start)
  //  return nullptr;
  
  if (CurTok != ',')
    return LogError("Esperado ',' depois de atribuir valor inicial do for.");
  getNextToken();



  End = ParseExpression();
  //if (!End)
  //  return nullptr;

  


  std::unique_ptr<ExprAST> Step = std::make_unique<NumberExprAST>(1.0);
  if (CurTok == ',') { // The step value is optional.
    getNextToken();
    auto aux = ParseExpression();
    if (aux)
      Step = std::move(aux);
  }
  
  std::vector<std::unique_ptr<ExprAST>> Body;

  //std::cout << "\nSeen tabs on for body POST: " << SeenTabs << "\n\n";
  if (CurTok==tok_space)
    getNextToken();

  while(true)
  {
    //std::cout << "\n\nParsing new expression with tabs: " << SeenTabs << " tok: " << ReverseToken(CurTok) << "\n";
    if (SeenTabs <= cur_level_tabs && CurTok != tok_space)
    {
      //std::cout << "Breaking for with cur tok: " << ReverseToken(CurTok) << " Seen Tabs:" << SeenTabs <<  "\n";
      break;
    } 
    //std::cout << "\nSeen tabs on for body: " << SeenTabs << "\nCur tok: " << ReverseToken(CurTok) << "\n\n";

    while (CurTok == tok_space)
    {
      //std::cout << "\nJumping tok space\n\n";
      getNextToken();
    }

    //std::cout << "Post space has " << SeenTabs << " tabs.\n";
    if (SeenTabs <= cur_level_tabs)
      break;


    auto body = ParseExpression();
    if (!body)
      return nullptr;
    Body.push_back(std::move(body));
    //getNextToken();
  }

  if (CurTok==tok_space)
    getNextToken();

  return std::make_unique<ForExprAST>(IdName, std::move(Start), std::move(End),
                                       std::move(Step), std::move(Body));
}



/// whileexpr ::= 'while' identifier '=' expr ',' expr (',' expr)? 'in' expression
static std::unique_ptr<ExprAST> ParseWhileExpr() {

  int cur_level_tabs = SeenTabs;

  getNextToken(); // eat the while.


  if (CurTok != tok_identifier)
    return LogError("Identificador da variável de controle esperado depois do while.");


  auto Cond = ParseExpression(0);
  if (!Cond)
    return nullptr;
  
  std::vector<std::unique_ptr<ExprAST>> Body;

  //std::cout << "\nSeen tabs on for body: " << SeenTabs << "\n\n";

  if (CurTok==tok_space)
    getNextToken();

  while(true)
  {
    if (SeenTabs <= cur_level_tabs && CurTok != tok_space && CurTok != tok_tab)
    {
      //std::cout << "Breaking for with cur tok: " << CurTok << "\n";
      break;
    } 
    //std::cout << "\nSeen tabs on for body: " << SeenTabs << "\nCur tok: " << CurTok << "\n\n";

    while (CurTok == tok_space)
    {
      //std::cout << "\nJumping tok space\n\n";
      getNextToken();
    }

    //std::cout << "Post space has " << SeenTabs << " tabs.\n";
    if (SeenTabs <= cur_level_tabs)
      break;

    //std::cout << "\nParse new for expression" <<  "\n\n";
    auto body = ParseExpression();
    if (!body)
      return nullptr;
    Body.push_back(std::move(body));
    //getNextToken();
  }

  if (CurTok==tok_space)
    getNextToken();

  return std::make_unique<WhileExprAST>(std::move(Cond), std::move(Body));
}

static std::unique_ptr<ExprAST> ParseAsyncExpr() {

  int cur_level_tabs = SeenTabs;

  getNextToken(); // eat the async.

  
  std::vector<std::unique_ptr<ExprAST>> Bodies;
  std::cout << "async tabs level: " << cur_level_tabs <<  "\n";
  std::cout << "Pre expression token: " << ReverseToken(CurTok) << "\n";

  if (CurTok != tok_space)
    Bodies.push_back(std::move(ParseExpression()));
  else 
  {
    getNextToken(); // eat \n
    while(CurTok != ';')
    {

      if (SeenTabs <= cur_level_tabs && CurTok != tok_space && CurTok != tok_tab)
        break;
      
      //std::cout << "\nSeen tabs on finish body: " << SeenTabs << "\nCur tok: " << CurTok << "\n\n";


      while (CurTok == tok_space)
        getNextToken();
      

      //std::cout << "Post space has " << SeenTabs << " tabs.\n";

      if (SeenTabs <= cur_level_tabs)
        break;

      if (CurTok==tok_tab)
        getNextToken();

      //std::cout << "async expression current token: " << ReverseToken(CurTok) << "\n";


      Bodies.push_back(std::move(ParseExpression()));
        
    }
  }
  
  
  //std::cout << "Post async: " << ReverseToken(CurTok) << "\n";

  return std::make_unique<AsyncExprAST>(std::move(Bodies));
}


static std::unique_ptr<ExprAST> ParseFinishExpr() {

  int cur_level_tabs = SeenTabs;
  //std::cout << "Finish tabs level: " << cur_level_tabs <<  "\n";

  getNextToken(); // eat the finish.


  std::vector<std::unique_ptr<ExprAST>> Bodies;
  std::vector<bool> IsAsync;
  

  if (CurTok!=tok_space)
    LogError("Finish requer quebra de linha.");
  getNextToken(); 


  while(CurTok != ';')
  {

    if (SeenTabs <= cur_level_tabs && CurTok != tok_space)
    {
      //std::cout << "Breaking finish with cur tok: " << ReverseToken(CurTok) << "\n";
      //std::cout << "Current tabs: " << SeenTabs << "\n";
      break;
    }
    //std::cout << "\nSeen tabs on finish body: " << SeenTabs << "\nCur tok: " << CurTok << "\n\n";




    if (CurTok==tok_space)
      getNextToken();
    

    //std::cout << "finish expression current token: " << ReverseToken(CurTok) << "\n";

    if (CurTok == tok_async)
    {
      Bodies.push_back(std::move(ParseAsyncExpr()));
      IsAsync.push_back(true);
    }
    else
    {
      Bodies.push_back(std::move(ParseExpression()));
      IsAsync.push_back(false);
    }
  }


  return std::make_unique<FinishExprAST>(std::move(Bodies),
                                         std::move(IsAsync));
}


/// varexpr ::= 'var' identifier ('=' expression)?
//                    (',' identifier ('=' expression)?)* 'in' expression
static std::unique_ptr<ExprAST> ParseVarExpr() {
  getNextToken(); // eat the var.
  std::cout << "Parsing var expr\n";

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

  if (CurTok==tok_space)
    getNextToken();


  return std::make_unique<VarExprAST>(std::move(VarNames), "var");
}


static std::unique_ptr<ExprAST> ParseStrExpr() {
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

      Init = ParseStringExpr();
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

  return std::make_unique<StrExprAST>(std::move(VarNames), std::move(Body), "str");
}



unsigned char* current_data_attr;
std::vector<float> current_data_attr_dims;


extern "C" float * load_img(char *img_name)
{
  int width, height, channels;
  
  unsigned char* image_data = stbi_load(img_name, &width, &height, &channels, 0);

  if (image_data) {
    
    current_data_attr_dims.clear();
    current_data_attr_dims.push_back((float)width);
    current_data_attr_dims.push_back((float)height);
    current_data_attr_dims.push_back((float)channels);

    /*
    std::cout << "Width: " << width << " pixels\n";
    std::cout << "Height: " << height << " pixels\n";
    std::cout << "Channels: " << channels << "\n";
    */

    //stbi_image_free(image_data);

    float *image_data_float = new float[width * height * channels];
  
    // Loop through each pixel and convert to float between 0.0 and 1.0
    for (int y = 0; y < height; ++y) {
      for (int x = 0; x < width; ++x) {
        for (int c = 0; c < channels; ++c) {
          // Assuming unsigned char has 8 bits, scale by 1/255.0 to get a float value between 0.0 and 1.0
          image_data_float[(y * width + x) * channels + c] = (float)image_data[(y * width + x) * channels + c] / 255.0f;
        }
      }
    }

    return image_data_float;
    
  } else {
    std::string img_n = img_name;
    std::string _error = "Falha ao abrir a imagem: " + img_n + ".";
    LogErrorS(_error);
  }

  return nullptr;
}


extern "C" float * split_str_to_float(char *in_string, int gather_position)
{
  std::vector<std::string> splitted = split_str(in_string,'/');

  float * ret = new float[1];

  if(gather_position<0)
    gather_position = splitted.size()+gather_position;

  ret[0] = std::stof(splitted[gather_position]);

  return ret;
}


class PreprocessingsInterface {
public:
  virtual ~PreprocessingsInterface() = default;
  std::vector<float> Dims = {-1.0f};
  std::string Type = "None";
  std::string Name = "Unnamed";
  std::string isSelf = "false";

  virtual float * Preprocess(char *, ...) {}
  virtual void SetType(std::string Type) {
    this->Type=Type;
  }
};


class LoadImg : public PreprocessingsInterface
{

  public:
   LoadImg() {}

    virtual float * Preprocess(char *, ...) override;
};


class SplitStrToFloat : public PreprocessingsInterface
{
  std::vector<int> Args;

  public:
   SplitStrToFloat(std::vector<int> Args)
    : Args(std::move(Args)) {}

    virtual float * Preprocess(char *, ...) override;
};


float * LoadImg::Preprocess(char *file_name, ...)
{
  va_list args;
  va_end(args);

  return load_img(file_name);
}

float * SplitStrToFloat::Preprocess(char *file_name, ...)
{
  va_list args;
  va_end(args);

  int gather_position = Args[0];


  return split_str_to_float(file_name, gather_position);
}


//global
std::map<std::string, std::unique_ptr<PreprocessingsInterface>> preprocessings;



static std::unique_ptr<ExprAST> ParsePreprocessing(std::string preprocess_var_name) {

  if (CurTok!=tok_preprocessing)
    LogError("Pré-processamento desconhecido.");

  while (CurTok==tok_preprocessing)
  {
    
    std::string preprocess_name = IdentifierStr;
    
    
    getNextToken();

    std::vector<int> Args;
    if (CurTok=='(')
    {
      getNextToken();
      
      while(CurTok!=')')
      {
        int is_minus=1;

        if (CurTok=='-')
        {
          is_minus=-1;
          getNextToken();
        }

        
        if (auto Arg = ParseNumberExpr())
          Args.push_back(is_minus*NumVal);
        else
          return nullptr;

        if (CurTok == ')')
        {
          std::cout << "Broke\n";
          break;
        }
        if (CurTok != ',')
          return LogError("Esperado ')' ou ',' na lista de argumentos");
        getNextToken();
      }
      getNextToken();
    } 

    if (preprocess_name=="load_img")
      preprocessings[preprocess_var_name] = std::make_unique<LoadImg>();

    if (preprocess_name=="split_str_to_float")
      preprocessings[preprocess_var_name] = std::make_unique<SplitStrToFloat>(std::move(Args));


    if (CurTok==tok_space)
      break;
    if (CurTok!=',')
      LogError("Esperava-se ',' ou quebra de linha após pré-processamento.");
  }
  return nullptr;
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
  
  
  //std::cout << "\n\nPre-dot: " << pre_dot << " Post-dot: " << IdentifierStr  << "\n\n\n";


  std::string IdName = IdentifierStr;

  getNextToken(); // eat identifier.

  

  if (CurTok != '(') // Simple variable ref.
  {
    auto aux = std::make_unique<VariableExprAST>(IdName);
    if (std::find(tensorVars.begin(), tensorVars.end(), IdentifierStr) != tensorVars.end())
      aux->SetType("tensor");
    if (functionVars.find(IdName) != functionVars.end())
      aux->SetType("tensor");
    if (is_class_attr)
      aux->SetSelf(pre_dot);
    if (pre_dot=="self")
      aux->SetSelf("true");
    
    if (starts_with(IdName.c_str(), "preprocess_") && pre_dot=="self")
    {
      getNextToken(); // eat =

      ParsePreprocessing(IdName);

    }
    
    if (CurTok==tok_space)
      getNextToken();

    return aux;
  }


  // ParseCall.

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

  if (IdName=="view")
    Args.push_back(std::make_unique<NumberExprAST>(-2.0f));
  if (ends_with(IdName, "yield"))
    Args.push_back(std::make_unique<StringExprAST>("-2"));

  // Eat the ')'.
  getNextToken();


  bool is_var_forward = false;
  std::string callee_override = "none";
  if (functionVars.find(IdName) != functionVars.end())
  {
    is_var_forward = true;
    callee_override = functionVars[IdName];
  }

  
  auto aux = std::make_unique<CallExprAST>(IdName, std::move(Args), object_class, pre_dot, is_var_forward, callee_override);

  if (in_str(IdName, tensor_resulting_methods) || is_var_forward)
    aux->SetType("tensor");

  
  if (CurTok==tok_space)
    getNextToken();
  
  return aux;
}


static std::unique_ptr<ExprAST> ParseTensorExpr() {
  
  getNextToken(); // eat the tensor.
  
  std::vector<std::unique_ptr<ExprAST>> dims;
  std::vector<std::pair<std::string, std::unique_ptr<ExprAST>>> VarNames;
  std::string init = "xavu_relu";
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
      if (in_str(IdentifierStr, tensor_inits))
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

    
    std::unique_ptr<ExprAST> Init = nullptr;
    VarNames.push_back(std::make_pair(Name, std::move(Init)));

    // End of var list, exit loop.
    if (CurTok != ',')
      break;
    getNextToken(); // eat the ','.

    if (CurTok != tok_identifier)
      return LogError("Esperado um ou mais identificadores após var.");
  }



  auto aux = std::make_unique<TensorExprAST>(std::move(VarNames), "tensor",
                                             std::move(dims), init);
  aux->SetSelf(pre_dot);

  
  if (CurTok==tok_space)
    getNextToken();
  
  return aux;
}


//
static std::unique_ptr<ExprAST> ParseConv2dExpr() {
  
  getNextToken(); // eat the Conv2d.
  
  std::vector<std::unique_ptr<ExprAST>> dims;
  std::vector<std::pair<std::string, std::unique_ptr<ExprAST>>> VarNames;
  std::string init = "xavu_relu";
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
      if (in_str(IdentifierStr, tensor_inits))
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

  if (dims.size()<5)
    return LogError("Convolução requer argumentos canais, canais de saída, kernel size, stride e padding.");


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
    
    getNextToken(); // eat identifier.

    
    std::unique_ptr<ExprAST> Init = nullptr;
    VarNames.push_back(std::make_pair(Name, std::move(Init)));
    functionVars[Name] = "Conv2d";

    // End of var list, exit loop.
    if (CurTok != ',')
      break;
    getNextToken(); // eat the ','.

    if (CurTok != tok_identifier)
      return LogError("Esperado um ou mais identificadores após var.");
  }



  auto aux = std::make_unique<Conv2dExprAST>(std::move(VarNames), "conv2d",
                                             std::move(dims[0]), std::move(dims[1]), std::move(dims[2]),
                                             std::move(dims[3]), std::move(dims[4]),
                                             init);
  aux->SetSelf(pre_dot);

  
  if (CurTok==tok_space)
    getNextToken();
  
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
    getNextToken();
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
  case tok_var_str:
    return ParseStrExpr();
  case '(':
    return ParseParenExpr();
  case tok_if:
    return ParseIfExpr();
  case tok_for:
    return ParseForExpr();
  case tok_while:
    return ParseWhileExpr();
  case tok_async_finish:
    return ParseFinishExpr();
  case tok_var:
    return ParseVarExpr();
  case tok_tensor:
    return ParseTensorExpr();
  case tok_conv2d:
    return ParseConv2dExpr();
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
  //std::cout <<"Parse unary\n";
  if(CurTok==tok_space)
    getNextToken();
  // If the current token is not an operator, it must be a primary expr.
  
  //std::cout << "Unary current token " << CurTok << "\n";
  if (!isascii(CurTok) || CurTok == '(' || CurTok == ',')
  {
    //std::cout << "Returning, non-ascii found.\n";
    return ParsePrimary();
  }
  
  
  // If this is a unary operator, read it.
  int Opc = CurTok;
  
  //std::cout << "Unary expr\n";
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
    
    // check if it is a valid op


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
      


    if (CurTok == tok_space)
    {
      //std::cout << "Returning tok space with " << SeenTabs << " tabs. \n\n\n";
      getNextToken();
      return std::make_tuple(std::move(LHS), L_cuda);
    }

    int BinOp = CurTok;


    /*
    //todo: it somehow jumps wrong op placements
    std::cout << "\n\nCur tok: " << CurTok << "\n";
    std::cout << in_char(CurTok, ops) <<  "\n";

    if (not in_char(CurTok, ops))
    {
      LogErrorBreakLine("Operador desconhecido.");
      return std::make_tuple(nullptr,0);
    }
    
    std::cout << "Cur tok post error: " << CurTok << "\n";
    */



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
    

    

    
    RhsTok = CurTok;

    
    auto RHS = ParseUnary(); // Returns an identifier, number or expression result
    if (RHS->GetType()=="tensor")
      R_cuda=1;
    
    
    
    if (!RHS)
    {
      //std::cout << "RETURNING NULL Parse Unary \n";
      return std::make_tuple(nullptr,0);
    }

    
    


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
      
      

      

      if (L_cuda==1 && R_cuda==0)
      {
        LHS = std::make_unique<BinaryTensorScalarExprAST>(BinOp,
                                                      std::move(LHS), std::move(RHS));
        
      }
      else if (L_cuda==0 && R_cuda==1 )
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
        } else {
          if (BinOp!=':') // Avoid codegen reversing
            LHS = std::make_unique<BinaryTensorScalarExprAST>(BinOp,
                                                    std::move(RHS), std::move(LHS));
          else
            LHS = std::make_unique<BinaryTensorScalarExprAST>(BinOp,
                                                    std::move(LHS), std::move(RHS));
        }
          
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


/// expression
///   ::= unary binoprhs
///
static std::unique_ptr<ExprAST> ParseExpression(int tabcount) {
  //std::cout << "\nParse Expression tabcount " << tabcount << "\n\n";
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

  std::string is_tensor="no";
  std::vector<std::string> ArgNames, Types;
  while (CurTok != ')')
  {
    Types.push_back(IdentifierStr);
    if (IdentifierStr=="t")
      is_tensor="tensor";
    if (IdentifierStr=="c")
      is_tensor="function";
    if (IdentifierStr!="t" && IdentifierStr!="f" && IdentifierStr!="s" && IdentifierStr!="c")
      LogErrorP_to_comma("Tipo da variável no protótipo precisa ser t ou f.");
    else {
      getNextToken();

      ArgNames.push_back(IdentifierStr);
      if (is_tensor=="tensor")
        tensorVars.push_back(IdentifierStr);
      if (is_tensor=="function")
        functionVars[IdentifierStr] = "Conv2d";
      
      getNextToken();
    }
    is_tensor="no";


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

  if (CurTok!=tok_space)
    LogError("Protótipo requer finalização com quebra de linha.");
  getNextToken();

  return std::make_unique<PrototypeAST>(FnName, ArgNames, Types, Kind != 0,
                                         BinaryPrecedence);
}



/// definition ::= 'def' prototype expression
static std::unique_ptr<FunctionAST> ParseDefinition(std::string ClassName="") {

  int cur_level_tabs = SeenTabs;

  getNextToken(); // eat def.


  auto Proto = ParsePrototype(ClassName);
  if (!Proto)
    return nullptr;
  
  
  std::vector<std::unique_ptr<ExprAST>> Body;

  while(CurTok!=';')
  {
    if (SeenTabs <= cur_level_tabs && CurTok != tok_space)
      break;
      

    if (CurTok==tok_space)
      getNextToken();

    Body.push_back(std::move(ParseExpression()));
  }

  //std::cout << "function number of expressions: " << Body.size() << "\n";

  if (Body.size()==0)
    return nullptr;

  return std::make_unique<FunctionAST>(std::move(Proto), std::move(Body));
  
}


/// toplevelexpr ::= expression
static std::unique_ptr<FunctionAST> ParseTopLevelExpr() {
  //std::cout << "Top Level Expression\n";

  
  std::vector<std::unique_ptr<ExprAST>> Body;
  while(CurTok!=';')
  {
    Body.push_back(std::move(ParseExpression()));
    //std::cout << "\n\nTop level expr cur tok: " << ReverseToken(CurTok) <<  ".\n";
    //std::cout << "Top level expr number of expressions: " << Body.size() <<  ".\n\n\n";
  }
  

  // Make an anonymous proto.
  auto Proto = std::make_unique<PrototypeAST>("__anon_expr",
                                                std::vector<std::string>(),
                                                std::vector<std::string>());
    
  return std::make_unique<FunctionAST>(std::move(Proto), std::move(Body));
  
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

//global
static std::unique_ptr<KaleidoscopeJIT> TheJIT;
static std::unique_ptr<LLVMContext> TheContext;
static std::unique_ptr<LLVMContext> GlobalContext = std::make_unique<LLVMContext>();


static std::unique_ptr<IRBuilder<>> Builder;
static std::unique_ptr<Module> TheModule;
static std::unique_ptr<Module> GlobalModule;


static std::map<std::string, std::unique_ptr<PrototypeAST>> FunctionProtos;
static ExitOnError ExitOnErr;


// Vars
static std::map<std::string, AllocaInst *> NamedValues;
static std::map<std::string, AllocaInst *> NamedStrs;
static std::map<std::string, Value *> NamedClassValues;
static std::map<std::string, float> StoredValues;


// Tensors
static std::map<std::string, float *> NamedTensors;
static std::map<std::string, std::vector<float>> NamedDims;
static std::map<std::string, std::vector<float>> NamedDimsConv;

// Current Cuda Result

std::vector<float> currentDims;

// Cuda Parallellism
constexpr int num_parallel_streams = 2;
cudaStream_t parallel_streams[num_parallel_streams];
cudaEvent_t parallel_events[num_parallel_streams];

// Optimizer
static std::map<std::string, float *> NamedParamGrads;


// File Handling
std::vector<char *> glob_str_files;




// Handle Class self with phantom argument
std::string FirstArg, LastPreDot;


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
    return LogError("A definição de uma Classe requer suas respectivas funções.");

  int i=0;
  while(CurTok==tok_def)
  {
    
    auto Func = ParseDefinition(Name);
    if (!Func)
      return nullptr;
      //return LogError("Falha no parsing da função da Classe.");
    if (!ends_with(Func->getProto().getName(),"__init__") && i==0)
      return LogError("Classe requer método __init__");
    
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
  
  return nullptr;
}



extern "C" float sleep(float id)
{
  std::cout << "\n\nSleep " << id << " begin" << "\n";
  std::random_device rd;
  std::mt19937 gen(rd());

  std::uniform_int_distribution<> dis(3, 7); // Generate between 1 and 100
  int random_number = dis(gen);

  //std::this_thread::sleep_for(std::chrono::seconds(random_number));
  std::this_thread::sleep_for(std::chrono::seconds((int)id));

  std::cout << "Sleep " << id << " finish" << "\n";

  return id;
}






std::vector<float> BatchLessDims(std::vector<float> dims)
{
  // Removes first dim (batch dim).
  if (dims.size()<=1)
    LogError("Remover dimensão do batch requer uma entrada com mais de uma dimensão.");

  std::vector<float> new_dims;

  for (int i=0; i<dims.size()-1;i++)
    new_dims.push_back(dims[i+1]);

  return new_dims;
}

int DimsProd(std::vector<float> dims)
{
  float aux=1;
  for (int i = 0; i < dims.size(); i++)
    aux = aux*dims[i];
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


extern "C" void PrintDims(std::vector<float> dims)
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

int resultingDimsProdOnMult(std::vector<float> Ldims, std::vector<float> Rdims)
{
  float aux=1;
  for (int i = 0; i < Ldims.size()-1; i++)
    aux = aux * Ldims[i];
  aux = aux * Rdims[0];
  return (int)aux;
}

std::string RandomString(size_t length) {
  const std::string charset = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";
  std::random_device rd;
  std::mt19937 generator(rd());
  std::uniform_int_distribution<int> distribution(0, charset.size() - 1);

  std::string random_string;
  for (size_t i = 0; i < length; ++i) {
    int random_index = distribution(generator);
    random_string += charset[random_index];
  }

  return random_string;
}

extern "C" void *NewDimsOnMult(std::vector<float> Ldims, std::vector<float> Rdims)
{
  

  std::vector<float> new_dims;
  if (Ldims[Ldims.size()-1]!=Rdims[Rdims.size()-1])
  {
    LogError("A última dimensão dos tensors multiplicados precisa ser igual.");
    std::cout << "Dim LHS: ";
    PrintDims(Ldims);
    std::cout << "Dim RHS: ";
    PrintDims(Rdims);
    return nullptr; 
  }
  for (int i = 0; i < Ldims.size()-1; i++)
    new_dims.push_back(Ldims[i]);
  new_dims.push_back(Rdims[0]);


  std::string random_str = RandomString(15); 
  NamedDims[random_str] = new_dims; // Deal with new_dims being deleted after scope finished.

  //TODO: This can lead to out of memory errors.
  return &NamedDims[random_str];
}

extern "C" float Add(float value, float v2)
{
  return value + v2; 
}

extern "C" void PrintFloat(float value){
  std::cout << "Printing float.\n";
  std::cout << "Float value: " << value << "\n";
}

extern "C" void UnbugFloat(float value){
}

extern "C" float PrintStr(char* value){
  std::cout << "Str: " << value << "\n";
  return 0;
}

extern "C" char * shuffle_str(char *string_list)
{

  std::ostringstream oss;

  std::vector<std::string> splitted = split(string_list, "|||");


  std::random_shuffle(splitted.begin(), splitted.end());

  for (int i=0; i<splitted.size(); i++)
  {
    if (i>0)
      oss << "|||";
    oss << splitted[i];
  }

  std::string result = oss.str();

  char * cstr = new char [result.length()+1];
  std::strcpy (cstr, result.c_str());
    
  return cstr;
}


extern "C" float * LoadTensor(char* tensor_name){
  //std::cout << "\n\nLOAD TENSOR: " << tensor_name <<  "\n\n\n";
  return NamedTensors[tensor_name];
}

extern "C" void *LoadDims(char* tensor_name)
{
  return &NamedDims[tensor_name];
}

extern "C" void * LoadDimsConv(char *conv_namec, int is_obj_attr_or_self)
{
  std::string conv_name = conv_namec;
  if (is_obj_attr_or_self)
    conv_name = FirstArg + conv_name;
  
  return &NamedDimsConv[conv_name];
}



extern "C" float PrintTensor(char* tensorName){
  //std::cout << "Called print tensor\n";
  

  std::vector<float> dims = NamedDims[tensorName];
  int arr_size = DimsProd(dims);


  float *tensor_cuda = NamedTensors[tensorName];
  float *tensor = new float[arr_size];
  //std::cout << "Printing Tensor " << tensorName << "\n";
  
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
  PrintDims(dims);
  std::cout << "\n";

  delete[] tensor;

  return 0;
}

extern "C" float PrintTensorF(float *cuda_tensor, int d1, int d2){
  

  std::vector<float> dims;
  dims.push_back(d1);
  dims.push_back(d2);

  int arr_size = DimsProd(dims);


  float *tensor = new float[arr_size];
  //std::cout << "Printing Tensor " << arr_size << "\n";
  
  cudaDeviceSynchronize();
  cudaCheck(cudaMemcpy(tensor, cuda_tensor, arr_size*sizeof(float), cudaMemcpyDeviceToHost));


  
  std::cout << "\n";
  PrintDims(dims);
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
  
  delete[] tensor;

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
  if (not ShallCodegen)
    return ConstantFP::get(*TheContext, APFloat(0.0f));
  //std::cout << "Codegen for Number: " << Val << "\n";
  return ConstantFP::get(*TheContext, APFloat(Val));
}

Value *StringExprAST::codegen() {
  if (not ShallCodegen)
    return ConstantFP::get(*TheContext, APFloat(0.0f));
  SetName(Val);
  return Builder->CreateGlobalString(Val);
}


//===----------------------------------------------------------------------===//
// Dataset
//===----------------------------------------------------------------------===//



extern "C" char * _glob_b_(char *pattern) {
    // TODO: make var of type string vector to hold this result.

    glob_t glob_result;
    //std::vector<char *> glob_str_files;

    std::ostringstream  oss;
    oss << "";

    if (glob(pattern, GLOB_TILDE, NULL, &glob_result) == 0) {
        for (size_t i = 0; i < glob_result.gl_pathc; ++i) {
            //result = result + "|||" + glob_result.gl_pathv[i];
            if (i>0)
              oss << "|||";
            oss << glob_result.gl_pathv[i];

            glob_str_files.push_back(strdup(glob_result.gl_pathv[i]));
        }
        globfree(&glob_result);
    }

    int i=0;

    if (glob_str_files.size()<1)
      LogErrorS("Glob falhou ao encontrar arquivos.");
    
    
    std::string result = oss.str();
    //std::cout << result << "\n";

    char * cstr = new char [result.length()+1];
    std::strcpy (cstr, result.c_str());
    
    return cstr;
}



float *current_data;
float *current_labels;
extern "C" float Datasetinit_dataset(float batch_size)
{
  std::cout << "Executing init dataset\n";
  std::cout << "Fist arg: " << FirstArg << "\n";
  std::random_shuffle(glob_str_files.begin(), glob_str_files.end());
  load_img(glob_str_files[0]);

  
  int dims_prod = DimsProd(current_data_attr_dims);

  current_data = new float[batch_size*dims_prod];
  current_labels = new float[batch_size];

  // Using CUDA CPU pinned memory for faster PCI Express transfers to GPU
  // See: https://developer.nvidia.com/blog/how-optimize-data-transfers-cuda-cc/
  cudaCheck(cudaMallocHost(&current_data, batch_size*dims_prod*sizeof(float)));
  cudaCheck(cudaMallocHost(&current_labels, batch_size*sizeof(float)));
  return 0;
}

extern "C" float Datasetgetitem_1(float idx, char *tensor_name)
{
  //std::cout << "EXECUTING GETITEM\n";
  return 12321;
}

extern "C" float* Datasetgetitem_2(float idx);




int yield_pointer = 0;
bool has_dataset_started=false;
extern "C" float Datasetyield(float batch_size, char * x_name, ...)
{
  //std::cout << "Executing yield\n";
  //std::cout << "Fist arg: " << FirstArg << "\n";

  std::vector<char *> tensor_names;
  tensor_names.push_back(x_name);


  //std::cout << "X name: " << x_name << "\n";
  va_list args;
  va_start(args, x_name);

  for (int i=0; i<10; i++)
  {
    //std::cout << "Vararg for: " << i << "\n";
    char * name = va_arg(args, char *);
    //std::cout << "Name: " << name << "\n\n";
    
    if (starts_with(name, "-2"))
      break;

    tensor_names.push_back(name);
  }
  va_end(args);

  char * y_name = tensor_names[1];

  int dims_prod, y_dims_prod;


  /*
  if(!has_dataset_started)
  {
    for (char *preprocess : tensor_names)
    {
      std::cout << "Tensor name: " << preprocess << "\n";

    }
  }
  */
  
  int b=0;


  float *cur_float_img, *y_aux;

  while (b < batch_size)
  {


    //for (char *preprocess:tensor_names)
    std::string preprocessing = "preprocess_";
    preprocessing += (const char *)x_name;
    //std::cout << "Preprocessing for: " << preprocessing << "\n";
    cur_float_img = preprocessings[preprocessing]->Preprocess(glob_str_files[yield_pointer]);
    dims_prod = DimsProd(BatchLessDims(NamedDims[x_name]));
    for (int j = 0; j < dims_prod; ++j)
      current_data[b * dims_prod + j] = cur_float_img[j];
    
    preprocessing = "preprocess_";
    preprocessing += (const char *)y_name;
    y_aux = preprocessings[preprocessing]->Preprocess(glob_str_files[yield_pointer]);
    y_dims_prod=1;
    for (int j = 0; j < y_dims_prod; ++j)
      current_labels[b * y_dims_prod + j] = y_aux[j];

    delete[] cur_float_img;
    delete[] y_aux;
    
    b+=1;
    
    yield_pointer+=1;
    // Drop last batch and reset idx
    if(yield_pointer>(glob_str_files.size()-batch_size-batch_size))
    {
      std::random_shuffle(glob_str_files.begin(), glob_str_files.end());
      yield_pointer=0;
    }
  }

  float *x, *y;
  
  x = NamedTensors[x_name];
  y = NamedTensors[y_name];


  // todo - inputs is copied on default stream so this synchronises CPU/GPU for now
  /*
  cudaMemcpyAsync(x, current_data, batch_size*dims_prod*sizeof(float), cudaMemcpyHostToDevice,0);
  // memcpy targets in parallel then wait for them before fused_classifier
  cudaMemcpyAsync(y, current_labels, batch_size*sizeof(float), cudaMemcpyHostToDevice, parallel_streams[0]);
  cudaEventRecord(parallel_events[0], parallel_streams[0]);
  */
  
  cudaMemcpy(x, current_data, batch_size*dims_prod*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(y, current_labels, batch_size*sizeof(float), cudaMemcpyHostToDevice);

  std::vector<float> dims_x, dims_y;
  dims_x.push_back(batch_size);
  dims_y.push_back(batch_size);
  for(int i=0; i<current_data_attr_dims.size(); i++)
    dims_x.push_back(current_data_attr_dims[i]);


  NamedDims[x_name] = dims_x;
  NamedDims[y_name] = dims_y;

  return 0;
}



extern "C" float load_preprocess_img(char *tensor_name, char *img_name)
{
  float *img;
  img = load_img(img_name); 
  
  std::vector<float> dims = NamedDims[tensor_name];

  
  int img_dims_prod = dims[dims.size()-1]*dims[dims.size()-2]*dims[dims.size()-3];


  current_data = new float[img_dims_prod];
  cudaCheck(cudaMallocHost(&current_data, img_dims_prod*sizeof(float)));


  for (int j = 0; j < img_dims_prod; ++j)
    current_data[j] = img[j];
  delete[] img;


  float *x;
  cudaMalloc(&x, img_dims_prod*sizeof(float));
  cudaCheck(cudaMemcpy(x, current_data, img_dims_prod*sizeof(float), cudaMemcpyHostToDevice));

  NamedTensors[tensor_name] = x;
  NamedDims[tensor_name] = current_data_attr_dims;

  return 0;
}


//===----------------------------------------------------------------------===//
// Tensor Functionalities
//===----------------------------------------------------------------------===//


extern "C" float view(float first_dim, ...)
{
  
  std::string tensor_name = LastPreDot;
  std::vector<float> new_dims, new_dims_no_minus, current_dims;
  bool has_minus = false;
  current_dims = NamedDims[tensor_name];

  
  va_list args;
  va_start(args, first_dim);

  if (first_dim!=-1)
    new_dims_no_minus.push_back(first_dim);
  else
    has_minus=true;
  
    
  new_dims.push_back(first_dim);

  for (int i=0; i<10; i++)
  {
    if (i==9)
    {
      LogErrorS("A tensor with 10 dimensions???");
      return 0;
    }

    float dim = va_arg(args, float);
    if (dim==-2)
      break;
    new_dims.push_back(dim);

    if (dim!=-1)
      new_dims_no_minus.push_back(dim);
    else
      has_minus=true;
  }
  va_end(args);


  


  int current_dims_prod = DimsProd(current_dims);
  int new_dims_prod = DimsProd(new_dims);


  if (has_minus)
  {
    float hidden_dim = (float)current_dims_prod / (float)DimsProd(new_dims_no_minus);

    if ((float)((int)hidden_dim) != hidden_dim)
    {
      LogErrorS("view das dimensões não é compatível");
      return 0;
    }
    
    for (int i=0; i<new_dims.size(); i++)
      if (new_dims[i]==-1)
        new_dims[i] = hidden_dim;
    

  } else {
    if (current_dims_prod != new_dims_prod)
    {
      LogErrorS("view das dimensões não é compatível");
      PrintDims(current_dims);
      std::cout << "Produto das dimensões atuais: " << current_dims_prod  << "\n";
      PrintDims(new_dims);
      std::cout << "Produto das novas dimensões: " << new_dims_prod  << "\n";
      return 0;
    }
  }

  NamedDims[tensor_name] = new_dims;
  

  return  0;
}



//===----------------------------------------------------------------------===//
// Tensor -- Scalar   Operations
//===----------------------------------------------------------------------===//


__global__ void vec_mult(const float a, float* x, float* y) {
  y[threadIdx.x] = x[threadIdx.x] * a;
}
__global__ void vec_div(const float a, float* x, float* y) {
  y[threadIdx.x] = x[threadIdx.x] / a;
}
__global__ void vec_add(const float a, float* x, float* y) {
  y[threadIdx.x] = x[threadIdx.x] + a;
}
__global__ void vec_sub(const float a, float* x, float* y) {
  y[threadIdx.x] = x[threadIdx.x] - a;
}
__global__ void vec_log(const float* x, float* y) {
  y[threadIdx.x] = logf(x[threadIdx.x]);
}




extern "C" float *CudaScalarMult(float *tensor, std::vector<float> dims, float R) {

  int kDataLen = DimsProd(dims);


  float* device_y;
  cudaCheck(cudaMalloc(&device_y, kDataLen * sizeof(float)));
  


  int grid_size = kDataLen;
  int block_size = 32;
  size_t shared_mem_size = 2 * block_size / 32 * sizeof(float);
  vec_mult<<<grid_size, block_size, shared_mem_size>>>(R, tensor, device_y);

  
  return device_y;
}


extern "C" float *CudaScalarDiv(float *tensor, std::vector<float> dims, float R) {

  int kDataLen = DimsProd(dims);


  float* device_y;
  cudaCheck(cudaMalloc(&device_y, kDataLen * sizeof(float)));


  // Launch the kernel.
  vec_div<<<1, kDataLen>>>(R, tensor, device_y);

  
  return device_y;
}

extern "C" float *CudaScalarAdd(float *tensor, std::vector<float> dims, float R) {
  
  int kDataLen = DimsProd(dims);


  float* device_y;
  cudaCheck(cudaMalloc(&device_y, kDataLen * sizeof(float)));
  
  // Launch the kernel.
  
  vec_add<<<1, kDataLen>>>(R, tensor, device_y);

  return device_y;
}

extern "C" float *CudaScalarSub(float *tensor, std::vector<float> dims, float R) {

  int kDataLen = DimsProd(dims);


  float* device_y;
  cudaCheck(cudaMalloc(&device_y, kDataLen * sizeof(float)));


  // Launch the kernel.
  vec_sub<<<1, kDataLen>>>(R, tensor, device_y);



  return device_y;
}


extern "C" float *logE(char *tensorName) {
  
  float * device_x;

  
  device_x = NamedTensors[tensorName];
  currentDims = NamedDims[tensorName];
  

  int kDataLen = DimsProd(currentDims);


  float* device_y;
  cudaCheck(cudaMalloc(&device_y, kDataLen * sizeof(float)));


  // Launch the kernel.
  vec_log<<<1, kDataLen>>>(device_x, device_y);



  return device_y;
}


extern "C" float FirstArgOnDemand(char *pre_dotc, int nested_function)
{
  std::string pre_dot = pre_dotc;

  //std::cout << "\n\nLAST PRE DOT " << LastPreDot << " PRE DOT " << pre_dot << "\n";
  
  LastPreDot = pre_dot;
  if (pre_dot!="self")
  {
    if (nested_function)
      FirstArg = FirstArg+pre_dot;
    else
      FirstArg = pre_dot;
  }
  //std::cout << "Resulting first arg: " << FirstArg << "\n\n\n";

  return 0;
}

extern "C" float DimnishFirstArgOnDemand(char *pre_dot, int nested_function)
{
  //std::cout << "\n\nDIMNISH FIRST ARG" << "\n\n\n";
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


extern "C" float StoreStrOnDemand(char *object_var_name, char * value){
  

  
  //NamedClassValues[FirstArg + object_var_name] = ConstantFP::get(*GlobalContext, APFloat(value));
  NamedClassValues[FirstArg + object_var_name] = Builder->CreateGlobalString(value);
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
  if (not ShallCodegen)
    return ConstantFP::get(*TheContext, APFloat(0.0f));
  // Look this variable up in the function.

  //std::cout << "Now Loading Var "<< Name <<" to Context" << "  \n";


  Function *TheFunction = Builder->GetInsertBlock()->getParent();
  std::string functionName = TheFunction->getName().str();
  
  
  Value * ret = ConstantFP::get(*TheContext, APFloat(0.0f));


  Value *var_name, *object_name, *object_var_name;
  var_name = Builder->CreateGlobalString(Name);
  

  /*
  std::cout << "\nVARIABLE EXPR CODEGEN: " << Name << "\n";
  for (const auto &entry : NamedStrs)
    std::cout << "NamedStr: " << entry.first << "\n";
  for (const auto &entry : NamedValues)
    std::cout << "NamedValues: " << entry.first << "\n";
  for (const auto &entry : NamedTensors)
    std::cout << "NamedTensors: " << entry.first << "\n";
  */

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
    //std::cout << "\nVariable Float " << Name << " codegen.\n";

    Value *V = NamedValues[Name];

    V = Builder->CreateLoad(Type::getFloatTy(*TheContext), V, Name.c_str());
    
    
    if (!seen_var_attr) //TODO: Solve this bug
      Builder->CreateCall(TheModule->getFunction("UnbugFloat"), {V}, "unbugfloat");

    return V;

  } else if (NamedStrs.count(Name)>0) {

    //std::cout << "\nVariable Str " << Name << " Codegen. \nNamedStrs.count(Name): " << NamedStrs.count(Name) <<"\n\n";

    for (const auto &entry : NamedTensors)
      if (ends_with(entry.first, Name))
        return ret;

    Value *V = NamedStrs[Name];
    
    V = Builder->CreateLoad(int8PtrTy, V, Name.c_str());
    if (!seen_var_attr)
      Builder->CreateCall(TheModule->getFunction("PrintStr"), {V});

    //std::cout << "RETURNING STRING: " << Name << "\n";
    //std::cout << "NamedStrs count:" << NamedStrs.count(Name) << "\n";
    return V;
  } else if (NamedTensors.count(Name)>0) {
    //std::cout << "\nVariable Tensor " << Name << " Codegen.\n";
  

    if (!seen_var_attr)
      Builder->CreateCall(TheModule->getFunction("PrintTensor"), {var_name});
    
    
    
    Value *dims_ptr = Builder->CreateCall(TheModule->getFunction("LoadDims"), {var_name});
    DimsPtr = Builder->CreateAlloca(int8PtrTy);
    Builder->CreateStore(dims_ptr, DimsPtr);
    

    //Builder->CreateCall(TheModule->getFunction("PrintTensor"), {var_name});

    return Builder->CreateCall(TheModule->getFunction("LoadTensor"), {var_name});
  }
}



extern "C" float toStoredValues(float Val, char * name_to_store)
{
  StoredValues[name_to_store] = Val;
  return 0;
}


extern "C" float temporaryCudaResult_Attr(char *tensor_name, float *tensor, std::vector<float> new_dims)
{
  //std::cout << "Attributing to tensor: " << tensor_name << "\n";

  //PrintDims(currentDims);

  cudaCheck(cudaFree(NamedTensors[tensor_name]));
  


  cudaCheck(cudaGetLastError());
  
  NamedTensors[tensor_name] = tensor;
  NamedDims[tensor_name] = new_dims;
  

  return 0;
}





Value *BinaryTensorScalarExprAST::codegen() {
  if (not ShallCodegen)
    return ConstantFP::get(*TheContext, APFloat(0.0f));

  Value *tensor_name = Builder->CreateGlobalString(LHS->GetName());

  DimsPtr = Builder->CreateAlloca(int8PtrTy);

  std::string pre_dot = LHS->GetSelf();
  if (pre_dot=="true")
    tensor_name = Builder->CreateCall(TheModule->getFunction("ConcatFirstArgToVarName"),
                                                      {tensor_name});
    // Gets from pre_dot if it is a class attribute
  else if (pre_dot!="false") {
    Value * object_name = Builder->CreateGlobalString(pre_dot);

    tensor_name = Builder->CreateCall(TheModule->getFunction("ConcatStr"),
                                                      {object_name, tensor_name});
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
    
    
    Builder->CreateCall(TheModule->getFunction("temporaryCudaResult_Attr"),
                            {tensor_name});        
      
    
      
    
    seen_var_attr=false;
    return Val;
  }


  std::cout << "\n\n\nTensor scalar for LHS: " << LHS->GetName() << " RHS: " << RHS->GetName() << "\n\n\n";
  Value *LtensorPtr = LHS->codegen();
  Value *R = RHS->codegen();
  std::cout << "\n\n\nTensor scalar post codegen" << "\n\n\n";



  
  Value *LdimsPtr = Builder->CreateLoad(int8PtrTy, LHS->GetDimsPtr());
  Builder->CreateStore(LdimsPtr, DimsPtr);

  //Builder->CreateCall(TheModule->getFunction("PrintDims"),
  //                    {LdimsPtr});

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
                               {LtensorPtr, LdimsPtr, R}, "cudascalarmult");
  case '/':
    return Builder->CreateCall(TheModule->getFunction("CudaScalarDiv"),
                               {LtensorPtr, LdimsPtr, R}, "cudascalardiv");
  case '+':
    return Builder->CreateCall(TheModule->getFunction("CudaScalarAdd"),
                               {LtensorPtr, LdimsPtr, R}, "cudascalaradd");
  case '-':
    return Builder->CreateCall(TheModule->getFunction("CudaScalarSub"),
                               {LtensorPtr, LdimsPtr, R}, "cudascalarsub");
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
    
    //std::cout << "matmul_forward. B: " << B << " C: " << C << " OC: " << OC << "\n";
    
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
  //std::cout << "matmul_backward. B: " << B << " C: " << C << " OC: " << OC << "\n";

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


//global
using backward_tuple = std::tuple<int, int, int, int, int, float *, float *, float *, std::string, std::string>;
std::vector<backward_tuple> todo_backwards;



extern "C" float *CudaMult(char *LtensorName, char *RtensorName, int is_forward_func,
                          float *device_x, float *device_w,
                          std::vector<float> Ldims, std::vector<float> Rdims) {
  
  //std::cout << "cuda mult called\n";
  //std::cout << "L " << LtensorName << "\nR " << RtensorName << "\n";
  

  


  std::vector<float> linear_layer_dims = format_LinearLayer_Dims(Ldims);
  int input_dims_prod = DimsProd(linear_layer_dims);
  //int resultingDimsProd = (int)linear_layer_dims[0]*Rdims[0];
  int resultingDimsProd = resultingDimsProdOnMult(linear_layer_dims, Rdims);

  /*
  std::cout << "At cuda mult:\n";
  PrintTensorF(device_x, linear_layer_dims[0], linear_layer_dims[1]);

  PrintTensorF(device_w, 2, 2);
  */


  float* device_y;
  cudaCheck(cudaMalloc(&device_y, resultingDimsProd * sizeof(float)));

  if (Ldims.size()<2)
    LogErrorS("Tensor de entrada da multiplicação de tensors precisa ter ao menos 2 dimensões.");



  
  //if(currentDims[1]==784)
  //PrintTensorF(device_x, currentDims[0], currentDims[1]);
  //PrintTensorF(device_w, Rdims[0], Rdims[1]);

  matmul_forward2(device_y, device_x, device_w,
                  linear_layer_dims[0], linear_layer_dims[1],
                  Rdims[0]);
                  //64
                  //);



  //std::cout << "L tensor: " << LtensorName << " R tensor: " << RtensorName << "\n";
  

  
  if (is_forward_func)
  {
    float *inp, *out;
    float B  = linear_layer_dims[0];
    float C  = linear_layer_dims[1];
    float OC = Rdims[0];
        

    //oom
    cudaCheck(cudaMalloc(&inp, input_dims_prod * sizeof(float)));
    cudaCheck(cudaMalloc(&out, resultingDimsProd * sizeof(float)));
    cudaMemcpy(inp, device_x, input_dims_prod * sizeof(float), cudaMemcpyDeviceToDevice);
    cudaMemcpy(out, device_y, resultingDimsProd * sizeof(float), cudaMemcpyDeviceToDevice);

    todo_backwards.push_back(std::make_tuple(B, C, OC,
                                             B*C, C*OC, inp, device_w, out,
                                            "matmul", RtensorName));
  }


  //PrintTensorF(device_y, 2, 2);

  return device_y;
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


// Parallelizes over B, C
__global__ void onehot_kernel(const float* tensor,
                           float* probs,
                           int B, int C) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    //int i = threadIdx.x;
    
    
    if (i < B * C) {
        int b = i / (C);
        int v = i % C;

        float* probs_b = probs + b * C;

        int ix = tensor[b];

        //float p = probs_b[v];

        float indicator = (v==ix) ? 1.0f : 0.0f;

        probs_b[v] = indicator;
        
    }
}

extern "C" float *onehot(float num_classes)
{
  std::string tensor_name = FirstArg;

  float * tensor = NamedTensors[tensor_name];
  std::vector<float> dims = NamedDims[tensor_name];
  
  int B = DimsProd(dims);
  int C = (int)num_classes;

  float *probs;

  cudaMalloc(&probs, B*C*sizeof(float));
  


  int grid_size = B;
  int block_size = 32;
  size_t shared_mem_size = 2 * block_size / 32 * sizeof(float);


  
  onehot_kernel<<<grid_size, block_size, shared_mem_size>>>(tensor, probs, B, C);
  //grid_size = ceil_div(B*C, block_size);
  //onehot_kernel<<<grid_size, block_size>>>(tensor, probs, B, C);

  dims.push_back(C);
  NamedDims[tensor_name] = dims;

  cudaFree(NamedTensors[tensor_name]);
  NamedTensors[tensor_name] = probs;


  return probs;
}


//TODO: mean, sum, max over axis
extern "C" float mean() 
{
  std::string tensor_name = FirstArg;

  float * tensor = NamedTensors[tensor_name];
  std::vector<float> dims = NamedDims[tensor_name];
  
  int B = DimsProd(dims);


  float *meaned = new float[B];

  cudaMemcpy(meaned, tensor, B*sizeof(float), cudaMemcpyDeviceToHost);
  
  float tensor_mean=0;
  for(int i=0; i<B; i++)
    tensor_mean += meaned[i];
  tensor_mean = tensor_mean/B;

  std::cout << "Mean: " << tensor_mean << "\n";

  return 0;
}

extern "C" float sum()
{
  std::string tensor_name = FirstArg;

  float * tensor = NamedTensors[tensor_name];
  std::vector<float> dims = NamedDims[tensor_name];
  
  int B = DimsProd(dims);


  float *summed = new float[B];

  cudaMemcpy(summed, tensor, B*sizeof(float), cudaMemcpyDeviceToHost);
  
  float tensor_sum=0;
  for(int i=0; i<B; i++)
    tensor_sum += summed[i];
  tensor_sum = tensor_sum;

  std::cout << "Sum: " << tensor_sum << "\n";

  return 0;
}

extern "C" float max()
{
  std::string tensor_name = FirstArg;

  float * tensor = NamedTensors[tensor_name];
  std::vector<float> dims = NamedDims[tensor_name];
  
  int B = DimsProd(dims);



  float max=-999;
  float *summed = new float[B];

  cudaMemcpy(summed, tensor, B*sizeof(float), cudaMemcpyDeviceToHost);
  
  float tensor_sum=0;
  for(int i=0; i<B; i++)
  {
    if(summed[i]>max)
      max = summed[i];
  }

  std::cout << "Max: " << max << "\n";

  return 0;
}

__global__ void gelu_forward_kernel1(const float* inp, float* out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        float xi = inp[i];
        float cube = 0.044715f * xi * xi * xi;
        out[i] = 0.5f * xi * (1.0f + tanhf(GELU_SCALING_FACTOR * (xi + cube)));
    }
}

extern "C" float *gelu(char * tensor_name) {

  float *tensor = NamedTensors[tensor_name];
  std::vector<float> dims = NamedDims[tensor_name];
  

  float dims_prod = DimsProd(dims);
  float block_size = 32;

  std::vector<float> linear_layer_dims = format_LinearLayer_Dims(dims);
  

  float *y;
  cudaMalloc(&y, dims_prod*sizeof(float));

  const int grid_size = ceil_div(dims_prod, block_size);
  gelu_forward_kernel1<<<grid_size, block_size>>>(tensor, y, dims_prod);
  cudaCheck(cudaGetLastError());

  
  int is_forward_func=1;
  if (is_forward_func)
  {
    float *inp, *out;

    
    float B  = linear_layer_dims[0];
    float C  = linear_layer_dims[1];
    float OC = linear_layer_dims[1];

    //oom
    cudaCheck(cudaMalloc(&inp, dims_prod * sizeof(float)));
    cudaCheck(cudaMalloc(&out, dims_prod * sizeof(float)));
    cudaMemcpy(inp, tensor, dims_prod * sizeof(float), cudaMemcpyDeviceToDevice);
    cudaMemcpy(out, y, dims_prod * sizeof(float), cudaMemcpyDeviceToDevice);

    todo_backwards.push_back(std::make_tuple(B, C, OC, B*C, C*OC, inp, nullptr, out,
                                           "gelu", "none"));
  }


  

  return y;
}

__global__ void gelu_backward1(float* dinp, const float* inp, const float* dout, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        float x = (float)inp[i];
        float cube = 0.044715f * x * x * x;
        float tanh_arg = GELU_SCALING_FACTOR * (x + cube);
        float tanh_out = tanhf(tanh_arg);
        float coshf_out = coshf(tanh_arg);
        float sech_out = 1.0f / (coshf_out * coshf_out);
        float local_grad = 0.5f * (1.0f + tanh_out) + x * 0.5f * sech_out * GELU_SCALING_FACTOR * (1.0f + 3.0f * 0.044715f * x * x);
        dinp[i] = (float)(local_grad * (float)dout[i]);
    }
}

void gelu_backward(const float* inp, float B, float C, float* dinp, const float* dout) {

  float N = B * C;
  float block_size = 32;

  const int grid_size = ceil_div(N, block_size);
  gelu_backward1<<<grid_size, block_size>>>(dinp, inp, dout, N);
  cudaCheck(cudaGetLastError());
}



__global__ void relu_forward(float* Z, float* A,
                                      float N) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < N) {
        A[index] = fmaxf(Z[index], 0);
    }
}

extern "C" float *relu(char *tensor_name)
{
  float * tensor = NamedTensors[tensor_name];
  std::vector<float> dims = NamedDims[tensor_name];
  std::vector<float> linear_layer_dims = format_LinearLayer_Dims(dims);

  float N = DimsProd(dims);
  float block_size = 32;

  float *y;
  cudaMalloc(&y, N*sizeof(float));

  const int grid_size = ceil_div(N, block_size);
  relu_forward<<<grid_size, block_size>>>(tensor, y, N);

  int is_forward_func=1;
  if (is_forward_func)
  {
    float *inp, *out;

    float B  = linear_layer_dims[0];
    float C  = linear_layer_dims[1];
    float OC = linear_layer_dims[1];

    //oom
    cudaCheck(cudaMalloc(&inp, N * sizeof(float)));
    cudaCheck(cudaMalloc(&out, N * sizeof(float)));
    cudaMemcpy(inp, tensor, N * sizeof(float), cudaMemcpyDeviceToDevice);
    cudaMemcpy(out, y, N * sizeof(float), cudaMemcpyDeviceToDevice);

    todo_backwards.push_back(std::make_tuple(B, C, OC, B*C, OC*C, inp, nullptr, out,
                                           "relu", "none"));
  }


  

  return y;
}

__global__ void relu_backward1(float* Z, float* dZ, float* dA,
                                       float N) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    
    if (index < N) {
        if (Z[index] > 0) {
            dZ[index] = dA[index];
        }
        else {
            dZ[index] = 0;
        }
    }
}

void relu_backward(float* inp, float B, float C, float* dinp, float* dout) {

  float N = B * C;
  float block_size = 32;

  const int grid_size = ceil_div(N, block_size);
  relu_backward1<<<grid_size, block_size>>>(inp, dinp, dout, N);
  cudaCheck(cudaGetLastError());
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



extern "C" float *softmax(char * tensor_name)
{

  float * tensor = NamedTensors[tensor_name];
  std::vector<float> dims = NamedDims[tensor_name];
  
  dims =  format_LinearLayer_Dims(dims);

  int B = dims[0];
  int C = dims[1];

  int grid_size = B;
  int block_size = 32;
  size_t shared_mem_size = 2 * block_size / 32 * sizeof(float);


  float *probs;
  cudaMalloc(&probs, B*C*sizeof(float));

  softmax_forward_kernel4<<<grid_size, block_size, shared_mem_size>>>(tensor, probs, B, C);

  

  return probs;
}




class Conv2d
{
  // Forward
  cudnnTensorDescriptor_t input_desc;
  cudnnFilterDescriptor_t filter_desc;
  cudnnFilterDescriptor_t filter_desc_g;
  cudnnConvolutionDescriptor_t conv_desc;
  cudnnTensorDescriptor_t output_desc;

  cudnnConvolutionFwdAlgo_t fwd_algo;
  std::size_t workspace_size;
  void* d_workspace;


  // Weight backward grad
  cudnnTensorDescriptor_t dy_desc;
  cudnnConvolutionBwdFilterAlgo_t w_bwd_algo;
  std::size_t workspace_size_w_back;
  void* d_workspace_w_back;


  // Input backward grad
  cudnnConvolutionBwdDataAlgo_t y_bwd_algo;
  std::size_t workspace_size_y_back;
  void* d_workspace_y_back;


  // Weights

  

  public:
    float* d_filter=nullptr;
    float* d_filter_g=nullptr;
    int C, OC, ks, stride, padding, out_H, out_W;
    int B = 0;
    int H = 0;
    int W = 0;
    std::string Init;

    Conv2d(int C, int OC, int ks, int stride, int padding, std::string Init) 
        : C(C), OC(OC), ks(ks), stride(stride), padding(padding), Init(Init) {}

  


  void SetDescriptors(int, int, int);
  void InitFilters();
  float *Forward(float *, int, int, int);
  void Backward(float *, float *, float *, float *);

};




//global
static std::map<std::string, std::unique_ptr<Conv2d>> NamedConv2d;
void Conv2d::SetDescriptors(int H, int W, int B)
{
  this->H = H;
  this->W = W;
  this->B = B;


  //std::cout << "\nConv2d Set Descriptors\nC: " << C << " OC " << OC << " ks " << ks << " stride " << stride << " padding " << padding << " H " << H << " W " << W << "\n";


  out_H = std::floor((H - ks + 2 * padding) / stride) + 1;
  out_W = std::floor((W - ks + 2 * padding) / stride) + 1;
  //std::cout << "Out H: " << out_H << " out W: " << out_W << "\n";



  // Initialize input tensor descriptor
  cudnnTensorDescriptor_t input_desc;
  checkCUDNN(cudnnCreateTensorDescriptor(&input_desc));
  checkCUDNN(cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, B, C, H, W));
  this->input_desc = input_desc;

  // Initialize filter descriptor
  cudnnFilterDescriptor_t filter_desc;
  checkCUDNN(cudnnCreateFilterDescriptor(&filter_desc));
  checkCUDNN(cudnnSetFilter4dDescriptor(filter_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, OC, C, ks, ks));
  this->filter_desc = filter_desc;

  // Initialize convolution descriptor
  cudnnConvolutionDescriptor_t conv_desc;
  checkCUDNN(cudnnCreateConvolutionDescriptor(&conv_desc));
  checkCUDNN(cudnnSetConvolution2dDescriptor(conv_desc, padding, padding, stride, stride, 1, 1,
                                           CUDNN_CONVOLUTION, CUDNN_DATA_FLOAT));
  this->conv_desc = conv_desc;

  // Initialize output tensor descriptor
  cudnnTensorDescriptor_t output_desc;
  checkCUDNN(cudnnCreateTensorDescriptor(&output_desc));
  checkCUDNN(cudnnSetTensor4dDescriptor(output_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, B, OC, out_H, out_W));
  this->output_desc = output_desc;

  // Initialize output tensor descriptor
  cudnnTensorDescriptor_t dy_desc;
  checkCUDNN(cudnnCreateTensorDescriptor(&dy_desc));
  checkCUDNN(cudnnSetTensor4dDescriptor(dy_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, B, OC, out_H, out_W));
  this->dy_desc = dy_desc;

  
  int requested_algo_count;
  int algo_count;




  // Forward
  checkCUDNN(cudnnGetConvolutionForwardAlgorithmMaxCount(cudnn, &requested_algo_count));
  std::vector<cudnnConvolutionFwdAlgoPerf_t> perf_results(requested_algo_count);
  checkCUDNN(cudnnFindConvolutionForwardAlgorithm(
        cudnn,
        input_desc,
        filter_desc,
        conv_desc,
        output_desc,
        requested_algo_count,
        &algo_count,
        perf_results.data()
  ));

  this->fwd_algo = perf_results.front().algo;


  std::size_t workspace_size = 0;
  checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(
        cudnn,
        input_desc,
        filter_desc,
        conv_desc,
        output_desc,
        fwd_algo,
        &workspace_size
  ));

  void* d_workspace = nullptr;
  cudaCheck(cudaMalloc(&d_workspace, workspace_size));
  this->workspace_size = workspace_size;
  this->d_workspace = d_workspace;




  // Backward to input
  checkCUDNN(cudnnGetConvolutionBackwardDataAlgorithmMaxCount(cudnn, &requested_algo_count));
  std::vector<cudnnConvolutionBwdDataAlgoPerf_t> perf_results_back_y(requested_algo_count);
  checkCUDNN(cudnnFindConvolutionBackwardDataAlgorithm(
        cudnn,
        filter_desc,
        dy_desc,
        conv_desc,
        input_desc,
        requested_algo_count,
        &algo_count,
        perf_results_back_y.data()
  ));

  y_bwd_algo = perf_results_back_y.front().algo;

  std::size_t workspace_size_y_back = 0;
  checkCUDNN(cudnnGetConvolutionBackwardDataWorkspaceSize(
        cudnn,
        filter_desc,
        dy_desc,
        conv_desc,
        input_desc,
        y_bwd_algo,
        &workspace_size_y_back
  ));

  void* d_workspace_y_back = nullptr;
  cudaCheck(cudaMalloc(&d_workspace_y_back, workspace_size_y_back));
  this->workspace_size_y_back = workspace_size_y_back;
  this->d_workspace_y_back = d_workspace_y_back;




  // Backward to weight
  checkCUDNN(cudnnGetConvolutionBackwardFilterAlgorithmMaxCount(cudnn, &requested_algo_count));
  std::vector<cudnnConvolutionBwdFilterAlgoPerf_t> perf_results_back_w(requested_algo_count);
  checkCUDNN(cudnnFindConvolutionBackwardFilterAlgorithm(
        cudnn,
        input_desc,
        dy_desc,
        conv_desc,
        filter_desc,
        requested_algo_count,
        &algo_count,
        perf_results_back_w.data()
  ));

  w_bwd_algo = perf_results_back_w.front().algo;

  std::size_t workspace_size_w_back = 0;
  checkCUDNN(cudnnGetConvolutionBackwardFilterWorkspaceSize(
        cudnn,
        input_desc,
        dy_desc,
        conv_desc,
        filter_desc,
        w_bwd_algo,
        &workspace_size_w_back
  ));

  void* d_workspace_w_back = nullptr;
  cudaCheck(cudaMalloc(&d_workspace_w_back, workspace_size_w_back));
  this->workspace_size_w_back = workspace_size_w_back;
  this->d_workspace_w_back = d_workspace_w_back;





}





void Conv2d::InitFilters()
{
  std::vector<float> h_filter;
  float *filter;
  for (std::size_t idx = 0; idx < C * OC; ++idx) {

    if (Init=="xavu_relu")
      filter = make_xavier_uniform_float_relu(ks*ks, ks*ks*C, ks*ks*OC);
    if (Init=="xavu")
      filter = make_xavier_uniform_float(ks*ks, ks*ks*C, ks*ks*OC);
    if (Init=="zeros")
      filter = make_zeros_float(ks*ks);
    if (Init=="ones")
      filter = make_ones_float(ks*ks);
    if (Init=="randu")
      filter = make_random_float(ks*ks);


    for (int i=0; i < ks*ks; i++)
      h_filter.emplace_back(filter[i]);

    delete[] filter;
    //for (const auto& val : filter) 
    //  h_filter.emplace_back(val);
  }
    
  float* d_filter = nullptr;
  const std::size_t filter_size = h_filter.size();
  cudaCheck(cudaMalloc(&d_filter, filter_size * sizeof(float)));
  cudaCheck(cudaMemcpy(d_filter, h_filter.data(), filter_size * sizeof(float), cudaMemcpyDefault));
  this->d_filter = d_filter;
  
}





float *Conv2d::Forward(float *tensor, int H, int W, int B)
{
  // Initialize descriptors.
  //std::cout << "\nConv2d Forward with H: " << H << " W: " << W << "\n";


  if (H != this->H || W != this->W || B != this->B)
    this->SetDescriptors(H, W, B);

  // Initialize weights.
  if (d_filter==nullptr)
    this->InitFilters();


  
  // Forward
  float *d_output;
  cudaCheck(cudaMalloc(&d_output, B * out_H * out_W * OC * sizeof(float)));

  constexpr float one = 1.0f;
  constexpr float zero = 0.0f;

  checkCUDNN(cudnnConvolutionForward(
        cudnn,
        &one,
        input_desc,
        tensor,
        filter_desc,
        d_filter,
        conv_desc,
        fwd_algo,
        d_workspace,
        workspace_size,
        &zero,
        output_desc,
        d_output
    ));
  



  
  /*
  std::cout << "Input grad:\n";
  PrintTensorF(dx, B * C, H * W);


  std::cout << "W grad:\n";
  PrintTensorF(d_filter_g, OC * C, ks * ks);
  */

  return d_output;
}


void Conv2d::Backward(float *tensor, float *dx, float *d_filter_g, float *dy)
{
  //std::cout << "\nConv2d Backward with H: " << H << " W: " << W << "\n";


  constexpr float one = 1.0f;
  constexpr float zero = 0.0f;
  
  // Backward to input
  checkCUDNN(cudnnConvolutionBackwardData(
    cudnn,
    &one,
    filter_desc, // input tensor descriptor
    d_filter,
    dy_desc, // output grad tensor descriptor
    dy,
    conv_desc, // convolution descriptor
    y_bwd_algo, //Obtained with getConvolutionBackwardDataAlgorithm
    d_workspace_y_back, 
    workspace_size_y_back, //Obtained with getConvolutionBackwardDataWorkspaceSize
    &zero,
    input_desc, // filter descriptor
    dx
  ));


  // Backward to weight
  checkCUDNN(cudnnConvolutionBackwardFilter(
    cudnn,
    &one,
    input_desc, // input tensor descriptor
    tensor,
    dy_desc, // output grad tensor descriptor
    dy,
    conv_desc, // convolution descriptor
    w_bwd_algo, //Obtained with getConvolutionBackwardFilterAlgorithm
    d_workspace_w_back, 
    workspace_size_w_back, //Obtained with getConvolutionBackwardFilterWorkspaceSize
    &one,
    filter_desc, // filter descriptor
    d_filter_g
  ));


  /*
  std::cout << "d_w is:\n";
  PrintTensorF(d_filter_g, C*OC, ks*ks);
  std::cout << "\n";
  */

}



void conv2d_backward(float *inp,  float *weight,
                     float *dinp, float *dw,
                     float *dout, std::string conv_name)
{

  //std::cout << "conv2d_backward for " << conv_name << "\n";
  std::unique_ptr<Conv2d> conv = std::move(NamedConv2d[conv_name]);

  conv->Backward(inp, dinp, dw, dout);


  NamedConv2d[conv_name] = std::move(conv);

  cudaCheck(cudaGetLastError());
}




extern "C" float *ConvForward2d(char *tensor_name, char *conv_namec, int is_obj_attr_or_self)
{
  
  std::string conv_name = conv_namec;
  if (is_obj_attr_or_self)
    conv_name = FirstArg + conv_name;

  //std::cout << "Conv forward for tensor: " << tensor_name << " and conv: " << conv_name <<"\n";
  

  float *tensor, *output, *d_filter;
  tensor = NamedTensors[tensor_name];
  std::vector<float> dims = NamedDims[tensor_name];
  float input_dims_prod = DimsProd(dims);

  float B = dims[0];
  float C = dims[dims.size()-1];



  std::unique_ptr<Conv2d> conv = std::move(NamedConv2d[conv_name]);

  if (dims[dims.size()-1]!=conv->C)
  {
    std::string error = "O número de canais do tensor é " + std::to_string((int)dims[dims.size()-1]) + ", enquanto a entrada esperada da convolução tem canais " + std::to_string(conv->C);
    LogError(error);
    
    currentDims = dims;
    NamedConv2d[conv_name] = std::move(conv);
    return nullptr;
  }

  output = conv->Forward(tensor, dims[dims.size()-3], dims[dims.size()-2], dims[0]);

  int ks_H = conv->ks;
  int ks_W = conv->ks;


  
  
  float resultingDimsProd = B * (float)conv->OC * (float)conv->out_W * (float)conv->out_W;

  int is_forward_func = 1;
  if (is_forward_func)
  {
    float *inp, *out;
    
    
    //oom
    cudaCheck(cudaMalloc(&inp, input_dims_prod * sizeof(float)));
    cudaCheck(cudaMalloc(&out, resultingDimsProd * sizeof(float)));
    cudaMemcpy(inp, tensor, input_dims_prod * sizeof(float), cudaMemcpyDeviceToDevice);
    cudaMemcpy(out, output, resultingDimsProd * sizeof(float), cudaMemcpyDeviceToDevice);

    todo_backwards.push_back(std::make_tuple(B, C, conv->OC, input_dims_prod, C*conv->OC*ks_H*ks_W,
                                             inp, conv->d_filter, out,
                                             "conv2d", conv_name));
    
  }


  std::vector<float> new_dims = {(float)conv->B, (float)conv->out_H, (float)conv->out_W, (float)conv->OC};
  
  NamedTensors[conv_name] = conv->d_filter;

  //for backprop:
  NamedDims[conv_name] = {(float)conv->OC, (float)conv->C, (float)conv->ks, (float)conv->ks}; 

  //for forward resulting dims:
  NamedDimsConv[conv_name] = new_dims;
  

  //if (conv_name=="modelconv1")
  //  PrintTensorF(conv->d_filter, 1, conv->ks*conv->ks);

  NamedConv2d[conv_name] = std::move(conv);

  return output;
}




extern "C" float CreateConv2dOnDemand(char *tensor_name, int is_obj_attr_or_self, char *init,
                                      float C, float OC, float ks, float stride, float padding, float H, float W)
{
  
  std::string objectTensorName = tensor_name;
  if (is_obj_attr_or_self)
    objectTensorName = FirstArg + tensor_name;


  char * cObjectTensorName = new char[objectTensorName.length() + 1];
  std::strcpy(cObjectTensorName, objectTensorName.c_str());
  


  std::cout << "\nCreate conv on demand:\n   C: " << C << " OC " << OC << " ks " << ks << " stride " << stride << " padding " << padding << "\n";



  /*
  if (std::strcmp(init, "randu") == 0)
    tensor_cpu = make_random_float(product);
  else if (std::strcmp(init, "zeros") == 0)
    tensor_cpu = make_zeros_float(product);
  else if (std::strcmp(init, "ones") == 0)
    tensor_cpu = make_ones_float(product);
  else if (std::strcmp(init, "xavu") == 0)
    tensor_cpu = make_xavier_uniform_float(product, cur_dim[cur_dim.size()-1], cur_dim[cur_dim.size()-2]);
  else if (std::strcmp(init, "xavu_relu") == 0)
    tensor_cpu = make_xavier_uniform_float_relu(product, cur_dim[cur_dim.size()-1], cur_dim[cur_dim.size()-2]);
  else if (std::strcmp(init, "randint") == 0)
    tensor_cpu = make_random_int(product, 10);
  */

  auto conv = std::make_unique<Conv2d>((int)C, (int)OC, (int)ks, (int)stride, (int)padding, init);


  std::cout << "Adding " << objectTensorName << " to NamedConv2d dict\n";
  NamedConv2d[cObjectTensorName] = std::move(conv);
  


  return 0;
}





// Parallelizes over B, C
__global__ void crossentropy_softmax_backward_kernel1(float* dlogits,
                           const float* probs, const float* targets,
                           int B, int C) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    //int i = threadIdx.x;
    
    
    if (i < B * C) {
        int b = i / (C);
        int v = i % C;

        float* dlogits_b = dlogits + b * C;
        const float* probs_b = probs + b * C;

        //float ix = targets[v];
        float ix = targets[b * C + v];
        float p = probs_b[v];

        //float indicator = (v==ix) ? 1.0f : 0.0f;
        float indicator = ix;

        dlogits_b[v] += (p - indicator) / B;
        
    }
}


void CrossEntropyBackward(float *y_hat,
                          float *y,
                          int B, int C, int OC,
                          float *dloss)
{

  int grid_size = B;
  int block_size = 32;
  size_t shared_mem_size = 2 * block_size / 32 * sizeof(float);

  float *probs;
  cudaMalloc(&probs, B*C*sizeof(float));

  softmax_forward_kernel4<<<grid_size, block_size, shared_mem_size>>>(y_hat, probs, B, C);

  grid_size = ceil_div(B*C, block_size);
  crossentropy_softmax_backward_kernel1<<<grid_size, block_size>>>(dloss, probs, y, B, C);
  cudaFree(probs);

  cudaCheck(cudaGetLastError());
}



extern "C" float cross_entropy(char *y_hat, char *y)
{
  

  float *device_y_hat = NamedTensors[y_hat];
  float *device_y = NamedTensors[y];
  std::vector<float> y_hat_dims = NamedDims[y_hat];
  std::vector<float> y_dims = NamedDims[y];


  /*
  std::cout << "y_hat: " << y_hat << "\n";
  PrintDims(y_hat_dims);

  std::cout << "y: " << y << "\n";
  PrintDims(y_dims);
  */

  std::vector<float> linear_layer_dims = format_LinearLayer_Dims(y_hat_dims);
  
  float B  = linear_layer_dims[0];
  float C  = linear_layer_dims[1];
  float OC = y_dims[0];

  todo_backwards.push_back(std::make_tuple(B, C, OC, B*C, OC*C, device_y_hat, nullptr, device_y,
                                           "cross_entropy", "none"));

  return 0;
}



extern "C" float backprop()
{
  //float * loss_gradient = ;
  
  int B, C, OC;
  float x_size, w_size;
  float *inp, *w, *out, *last_inp;
  float *dinp, *device_dx, *dw, *device_dw, *dout, *device_dy;

  std::string op, param_name;
  
  bool first=true;

  while(todo_backwards.size()>0)
  {
    backward_tuple bt = std::move(todo_backwards.back());
    todo_backwards.pop_back();
    

    
    B = std::get<0>(bt);
    C = std::get<1>(bt);
    OC = std::get<2>(bt);
    x_size = std::get<3>(bt);
    w_size = std::get<4>(bt);
    inp = std::get<5>(bt);
    w = std::get<6>(bt);
    out = std::get<7>(bt);
    op = std::get<8>(bt);
    param_name = std::get<9>(bt);

    
    
    // weight gradient
    float *new_grad_ptr;
    if (w!=nullptr)
    {
      dw = make_zeros_float(w_size);
      
      if (NamedParamGrads[param_name]==nullptr)
      {
        NamedParamGrads[param_name] = new_grad_ptr;
        cudaCheck(cudaMalloc(&new_grad_ptr, w_size*sizeof(float)));
        NamedParamGrads[param_name] = new_grad_ptr;
      } 
      
      device_dw = NamedParamGrads[param_name];
      cudaCheck(cudaMemcpy(device_dw, dw, w_size*sizeof(float), cudaMemcpyHostToDevice));
      delete[] dw;
    }

    // input gradient
    dinp = make_zeros_float(x_size);
    cudaMalloc(&device_dx, x_size*sizeof(float));
    cudaMemcpy(device_dx, dinp, x_size*sizeof(float), cudaMemcpyHostToDevice);
    delete[] dinp;


    /*
    std::cout << "B: " << B << "\n";
    std::cout << "C: " << C << "\n";
    std::cout << "OC: " << OC << "\n";
    std::cout << "Op: " << op << "\n";
    */


    // No switch case for std::string
    if (op=="matmul")
      matmul_backward(inp, w, B, C, OC, device_dx, device_dw, device_dy);
    else if (op=="conv2d")
    {
      //std::cout << "\n\n\n\nSKIPPING CONVOLUTION BACKWARD\n\n\n\n";
      conv2d_backward(inp, w, device_dx, device_dw, device_dy, param_name);
    }
    else if (op=="relu")
      relu_backward(inp, B, C, device_dx, device_dy);
    else if (op=="gelu")
      gelu_backward(inp, B, C, device_dx, device_dy);
    else if (op=="cross_entropy")
      CrossEntropyBackward(inp, out, B, C, OC, device_dx);
    else
      LogErrorS("A operação não possui implementação do backward.");

    /*
    std::cout << "\nd inp:\n";
    PrintTensorF(device_dx, B, C);
    std::cout << "\n";

    std::cout << "d w:\n";
    PrintTensorF(device_dw, OC, C);
    std::cout << "\n\n";
    */
    
    if (w!=nullptr)
      NamedParamGrads[param_name] = device_dw;


    // Garbage Collector on all lines below
    cudaCheck(cudaFree(out));
    if (!first)
    {
      cudaCheck(cudaFree(last_inp));
      cudaCheck(cudaFree(device_dy));
    }
    device_dy = device_dx; // backpropagate gradient
    last_inp = inp;

    first = false;
  }
  cudaCheck(cudaFree(device_dx));
  cudaCheck(cudaFree(inp));

  return 0;
}


class Optimizer {
public:
  virtual ~Optimizer() = default;

  int timestep = 1;
  float lr = 0.0f;
  //float eps = 1.5e-4;
  float eps = 1e-8;
    
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

    delete[] v;
    delete[] m;

    NamedV[param_name] = device_v; 
    NamedM[param_name] = device_m;
  }
}

void AdamW_optim::step(float *param, float *grad, std::vector<float> dims, std::string param_name)
{
  //std::cout  << "Optimizer step called\n";
  

  float *v = NamedV[param_name];
  float *m = NamedM[param_name];

  float beta1_correction = 1.0f - powf(beta1, timestep);
  float beta2_correction = 1.0f - powf(beta2, timestep);

  

  /*
  std::cout << "param pre: \n";
  PrintTensorF(param, dims[0], dims[1]);

  std::cout << "\n\ngrad: \n";
  PrintTensorF(grad, dims[0], dims[1]);
  */


  int params_count = DimsProd(dims);
  int block_size = 512;
  int num_blocks = ceil_div(params_count, block_size);

  adamw_kernel<<<num_blocks, block_size>>>(param, grad, m, v, params_count,
                                           lr, beta1, beta2, beta1_correction, beta2_correction,
                                           eps, weight_decay);


}




std::unique_ptr<Optimizer> optimizer = nullptr;


extern "C" float AdamW(float lr, float beta1, float beta2, float weight_decay)
{

  if (optimizer==nullptr)
    optimizer = std::make_unique<AdamW_optim>(lr, beta1, beta2, weight_decay);

  for (auto& pair : NamedParamGrads)
  {
    std::string param_name = pair.first;

    if (param_name!="none")
    {
      optimizer->init_states(param_name, DimsProd(NamedDims[param_name]));
      optimizer->step(NamedTensors[param_name], pair.second, NamedDims[param_name], param_name);
    }
  }
  optimizer->count_step();



  return 0;
}




Value *BinaryTensorTensorExprAST::codegen() {
  if (not ShallCodegen)
    return ConstantFP::get(*TheContext, APFloat(0.0f));

  Value *LtensorName = Builder->CreateGlobalString(LHS->GetName());
  Value *RtensorName = Builder->CreateGlobalString(RHS->GetName());
  Value *object_name;


  //TensorPtr = Builder->CreateAlloca(floatPtrTy, nullptr);
  //Builder->CreateStore(FloatPtr_toValue(NamedTensors[Name]), TensorPtr);
  DimsPtr = Builder->CreateAlloca(int8PtrTy);


  // Concat self or obj name to tensor name
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


  


  // if is attribution
  if (Op == '=') {
  
    seen_var_attr=true;

    VariableExprAST *LHSE = static_cast<VariableExprAST *>(LHS.get());
    if (!LHSE)
      return LogErrorV("Destino do '=' deve ser uma variável.");
    
    Value *RtensorPtr = RHS->codegen();
    std::cout << "1 1 attr\n";
    

    //float *Variable = NamedTensors[LHSE->getName()];
    //if (!Variable)
    //  return LogErrorV("O nome do tensor/variável é desconhecido.");

    
    std::cout << "Pre dims\n";
    Builder->CreateLoad(int8PtrTy, RHS->GetDimsPtr());
    std::cout << "Post dims\n";

    Builder->CreateCall(TheModule->getFunction("temporaryCudaResult_Attr"),
                        {LtensorName, RtensorPtr,
                         Builder->CreateLoad(int8PtrTy, RHS->GetDimsPtr())});
    std::cout << "Post attr call\n";



    seen_var_attr=false;
    return ConstantFP::get(*TheContext, APFloat(0.0f));
  }




  


  std::string functionName = Builder->GetInsertBlock()->getParent()->getName().str();
  std::cout << "\nTensor Tensor for function: " << functionName << "\n";
  int forward_func = 0;
  if(ends_with(functionName, "forward"))
    forward_func = 1;
  forward_func = 1; // TODO: Remove this line



  
  Value *LtensorPtr = LHS->codegen();
  Value *RtensorPtr = RHS->codegen();

  std::cout << "Create load for dims\n";

  Value *LdimsPtr = Builder->CreateLoad(int8PtrTy, LHS->GetDimsPtr());
  Value *RdimsPtr = Builder->CreateLoad(int8PtrTy, RHS->GetDimsPtr());

  std::cout << "Load created\n";


  //Builder->CreateCall(TheModule->getFunction("PrintDims"),
  //                    {LdimsPtr});


  if (!LtensorPtr || !RtensorPtr)
    return nullptr;

  Function *CudaFn;

  std::cout << "Tensor tensor: " << LHS->GetName() << ", " << RHS->GetName() << "\n";
    


  Value *is_forward_func = ConstantInt::get(Type::getInt32Ty(*TheContext), forward_func);
  
  /*
  void *vec = &NamedDims[LHS->GetName()];
  Value* LLVMValue = ConstantInt::get(Type::getInt64Ty(*TheContext), reinterpret_cast<uint64_t>(vec));
  LLVMValue = Builder->CreateIntToPtr(LLVMValue, int8PtrTy);
  */

  
  Value *new_dims;

  switch (Op)
  {
  case '@':
  {
    std::cout << "Create store dims at cuda mult\n";
    new_dims = Builder->CreateCall(TheModule->getFunction("NewDimsOnMult"),
                                    {LdimsPtr, RdimsPtr});
    
    Builder->CreateStore(new_dims, DimsPtr);


    std::cout << "Create call for mult\n";
    return Builder->CreateCall(TheModule->getFunction("CudaMult"),
                                    {LtensorName, RtensorName, is_forward_func,
                                     LtensorPtr, RtensorPtr,
                                     LdimsPtr, RdimsPtr},
                                     "cudamult");
  }
  /*
  case '*':
    CudaFn = TheModule->getFunction("CudaMult");
    return Builder->CreateCall(CudaFn,{LtensorName, RtensorName, is_forward_func, LLVMValue},
                               "cudamult");
  */
  case '/':
    CudaFn = TheModule->getFunction("CudaDiv");
    return Builder->CreateCall(CudaFn, {LtensorName, RtensorName},
                               "cudadiv");
  case '+':
    CudaFn = TheModule->getFunction("CudaAdd");
    return Builder->CreateCall(CudaFn, {LtensorName, RtensorName},
                               "cudaadd");
  case '-':
    CudaFn = TheModule->getFunction("CudaSub");
    return Builder->CreateCall(CudaFn, {LtensorName, RtensorName},
                               "cudasub");
  case ':':
    return LtensorPtr;
  default:
    break;
  }
  

  
  Function *F = getFunction(std::string("binary") + Op);
  assert(F && "Operator not found.");

  Value *Ops[] = {LtensorName, RtensorName};
  return Builder->CreateCall(F, Ops, "binop");
}



Value *LogExprAST::codegen() {
  if (not ShallCodegen)
    return ConstantFP::get(*TheContext, APFloat(0.0f));
  
  
  return Builder->CreateCall(TheModule->getFunction("logE"),
                             {Builder->CreateGlobalString(Name)}, "cudalog");
}



Value *BinaryExprAST::codegen() {
  if (not ShallCodegen)
    return ConstantFP::get(*TheContext, APFloat(0.0f));
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


      if(LHS->GetSelf()=="true")
        Builder->CreateCall(TheModule->getFunction("StoreOnDemand"),
                                                  {Builder->CreateGlobalString(LHSE->getName()),
                                                   Val});
      else
        Builder->CreateStore(Val, Variable);
      
    
    } else if (NamedStrs.count(LHSE->getName()) != 0 ) {
      //std::cout << "ATTRIBUTTING TO STRING: " << LHSE->getName() << "\n";
      Value *Variable = NamedStrs[LHSE->getName()];
      
      if(LHS->GetSelf()=="true")
        Builder->CreateCall(TheModule->getFunction("StoreStrOnDemand"),
                                                  {Builder->CreateGlobalString(LHSE->getName()),
                                                   Val});
      else
        Builder->CreateStore(Val, Variable);

    } else {

      return LogErrorV("O nome da variável é desconhecido.");
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
  if (not ShallCodegen)
    return ConstantFP::get(*TheContext, APFloat(0.0f));
  Value *OperandV = Operand->codegen();
  if (!OperandV)
    return nullptr;
  
  
  
  //std::cout << "Operand type: " << Operand->GetType();
  if (Opcode=='-')
  {
    if (Operand->GetType()=="tensor")
    {
      Value *tensor_name = Builder->CreateGlobalString(Operand->GetName());
      Value *LtensorPtr = Builder->CreateCall(TheModule->getFunction("LoadTensor"),
                                              {tensor_name});
      Value *R = ConstantFP::get(Type::getFloatTy(*TheContext), -1);

      DimsPtr = Builder->CreateAlloca(int8PtrTy);
      Value *dims_ptr = Builder->CreateCall(TheModule->getFunction("LoadDims"),
                                              {tensor_name});
      Builder->CreateStore(dims_ptr, DimsPtr);
      
      return Builder->CreateCall(TheModule->getFunction("CudaScalarMult"),
                                {LtensorPtr, dims_ptr, R}, "cudascalarmult");
    }
    return Builder->CreateFMul(ConstantFP::get(Type::getFloatTy(*TheContext), -1),
                              OperandV, "multmp");
  }

  //std::cout << "Opcode: " << Opcode << "\n";

  if (Opcode=';')
    return ConstantFP::get(Type::getFloatTy(*TheContext), 0);
  

  Function *F = getFunction(std::string("unary") + Opcode);
  if (!F)
    return LogErrorV("Operador unário desconhecido.");

  return Builder->CreateCall(F, OperandV, "unop");
}


Value *IfExprAST::codegen() {
  if (not ShallCodegen)
    return ConstantFP::get(*TheContext, APFloat(0.0f));
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

  
  Value *ThenV;
  for (auto &then_body : Then)
    ThenV = then_body->codegen();
  

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
    ElseV = else_body->codegen();

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

Value *ForExprAST::codegen() {
  if (not ShallCodegen)
    return ConstantFP::get(*TheContext, APFloat(0.0f));
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
  
  
  for (auto &body : Body)
    body->codegen();


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



Value *WhileExprAST::codegen() {
  if (not ShallCodegen)
    return ConstantFP::get(*TheContext, APFloat(0.0f));
	Function* TheFunction = Builder->GetInsertBlock()->getParent();

	BasicBlock *EntryBB = BasicBlock::Create(*TheContext, "entry_while", TheFunction);
	BasicBlock *LoopBB = BasicBlock::Create(*TheContext, "loop_while", TheFunction);
	BasicBlock *AfterBB = BasicBlock::Create(*TheContext, "end_while", TheFunction);

	
	Builder->CreateBr(EntryBB);

	// Handle Cond

	Builder->SetInsertPoint(EntryBB);
	Value* condVal = Cond->codegen();
	if (! condVal)
    return nullptr;

	condVal = Builder->CreateFCmpONE(condVal, ConstantFP::get(*TheContext, APFloat(0.0)), "loopcond");
	Builder->CreateCondBr(condVal, LoopBB, AfterBB);
	EntryBB = Builder->GetInsertBlock();


	// Handle Loop Body
	
  Builder->SetInsertPoint(LoopBB);
	Value* bodyVal;

  for (auto &body : Body)
    bodyVal = body->codegen();

	Builder->CreateBr(EntryBB);


	// Handle Loop End
	
	Builder->SetInsertPoint(AfterBB);

	return Constant::getNullValue(Type::getFloatTy(*TheContext));
}


Function *codegenAsyncFunction(std::vector<std::unique_ptr<ExprAST>> &asyncBody) {
  

  // find unique function name (_async 0, _async1, _async2 etc)
  int fnIndex = 0;
  while (TheModule->getFunction("__async_" + std::to_string(fnIndex)))
    fnIndex++;
  
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


  
  // emit EntryBB value
  BasicBlock *EntryBB = BasicBlock::Create(*TheContext, "entry", asyncFun);
  Builder->SetInsertPoint(EntryBB);
  

  // define body of function
  Value *V;

  for (auto &body : asyncBody)
    V = body->codegen();

  if (V)
  {
    /*
    fprintf(stderr, "\nRead top-level expression:");
    FnIR->print(errs());
    fprintf(stderr, "\n\n");
    */

    //Builder->CreateRet(ConstantFP::get(*TheContext, APFloat(0.0f)));
    Builder->CreateRet(Constant::getNullValue(int8PtrTy));
    verifyFunction(*asyncFun);
    return asyncFun;
  }
  
  asyncFun->eraseFromParent();

  return nullptr;
}


//int pthread_create(pthread_t *thread, pthread_attr_t *attr,
//                   void *(*start_routine) (void *arg), void *arg);



extern "C" void pthread_create_aux(pthread_t *thread, pthread_attr_t *attr,
                   void *(*start_routine) (void *arg), void *arg)
{
  pthread_t t;
  t=0;
  
  pthread_create(thread, attr, start_routine, arg);
  //pthread_create(&t, attr, start_routine, arg);

  //pthread_join(t, nullptr);

  //std::cout << "Join\n";
}


extern "C" void pthread_join_aux(pthread_t thread)
{
  void **value_ptr;
  value_ptr = nullptr;

  pthread_join(thread, value_ptr);  
}





Value *AsyncExprAST::codegen() {
  if (not ShallCodegen)
    return ConstantFP::get(*TheContext, APFloat(0.0f));

  
  // Create/Spawn Threads

  //BasicBlock *CurrentBB = Builder->GetInsertBlock();
  Function *TheFunction = Builder->GetInsertBlock()->getParent();
  BasicBlock *CurrentBB = BasicBlock::Create(*TheContext, "loop", TheFunction);
  Builder->CreateBr(CurrentBB);

  Function *asyncFun = codegenAsyncFunction(std::ref(Body));

  Builder->SetInsertPoint(CurrentBB);

  
  Function *pthread_create = TheModule->getFunction("pthread_create_aux");

  PointerType *pthreadTy = Type::getInt8Ty(*GlobalContext)->getPointerTo();
  Value *pthreadPtr = Builder->CreateAlloca(pthreadTy, nullptr);
  //Builder->CreateStore(ConstantInt::get(Type::getInt8Ty(*GlobalContext), 0), pthreadPtr);
  
  

  Value *voidPtrNull = Constant::getNullValue(
      Type::getInt8Ty(*TheContext)->getPointerTo());


  
  Builder->CreateCall(pthread_create,
    {pthreadPtr,
     voidPtrNull,
     asyncFun,
     voidPtrNull}
  );


  return pthreadPtr;
}



Value *FinishExprAST::codegen() {
  if (not ShallCodegen)
    return ConstantFP::get(*TheContext, APFloat(0.0f));

  std::vector<Value *> thread_pointers;
  

  for (int i=0; i < Bodies.size(); i++)
  {
    if (IsAsync[i])
      thread_pointers.push_back(Bodies[i]->codegen());
    else
      Bodies[i]->codegen();
  }


  PointerType *pthreadTy = Type::getInt8Ty(*GlobalContext)->getPointerTo();

  Function *pthread_join = TheModule->getFunction("pthread_join_aux");


  int i=0;
  for (Value *pthreadPtr : thread_pointers)
  {
    Value *pthread = Builder->CreateLoad(pthreadTy, pthreadPtr);

    Builder->CreateCall(pthread_join,
                        {pthread});
    
    i+=1;
  }
  
  thread_pointers.clear();

  return ConstantFP::get(*TheContext, APFloat(0.0f));
}



// Create Var
Value *VarExprAST::codegen() {
  if (not ShallCodegen)
    return ConstantFP::get(*TheContext, APFloat(0.0f));
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


  return ConstantFP::get(*TheContext, APFloat(0.0));
}





Value *StrExprAST::codegen() {
  if (not ShallCodegen)
    return ConstantFP::get(*TheContext, APFloat(0.0f));
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
    //std::cout << "STRING CODEGEN FOR " << VarName << "\n";
    OldBindings.push_back(NamedStrs[VarName]);

    
    // Remember this binding.
    NamedStrs[VarName] = Alloca;
    
  }

  // Codegen the body that is contained by the in expression
  Value *BodyVal = Body->codegen();
  if (!BodyVal)
    return nullptr;

  // Pop all our variables from scope.
  for (unsigned i = 0, e = VarNames.size(); i != e; ++i)
    NamedStrs[VarNames[i].first] = OldBindings[i];

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
  int product = DimsProd(cur_dim);
  float * tensor;
  float * tensor_cpu;


  if (std::strcmp(init, "randu") == 0)
    tensor_cpu = make_random_float(product);
  else if (std::strcmp(init, "zeros") == 0)
    tensor_cpu = make_zeros_float(product);
  else if (std::strcmp(init, "ones") == 0)
    tensor_cpu = make_ones_float(product);
  else if (std::strcmp(init, "xavu") == 0)
    tensor_cpu = make_xavier_uniform_float(product, cur_dim[cur_dim.size()-1], cur_dim[cur_dim.size()-2]);
  else if (std::strcmp(init, "xavu_relu") == 0)
    tensor_cpu = make_xavier_uniform_float_relu(product, cur_dim[cur_dim.size()-1], cur_dim[cur_dim.size()-2]);
  else if (std::strcmp(init, "randint") == 0)
    tensor_cpu = make_random_int(product, 10);
  

  

  cudaMalloc(&tensor, product*sizeof(float));
  cudaCheck(cudaMemcpy(tensor, tensor_cpu, product*sizeof(float), cudaMemcpyHostToDevice));
  

  
  delete[] tensor_cpu;

  //if (NamedTensors.find(cObjectTensorName) != NamedTensors.end() && tensorName!="y")
  //  cudaCheck(cudaFree(NamedTensors[cObjectTensorName]));

  NamedTensors[cObjectTensorName] = tensor;
  NamedDims[cObjectTensorName] = cur_dim;



  //PrintTensor(cObjectTensorName);

  cur_dim.clear();

  return 0;
}




Value *TensorExprAST::codegen() {
  if (not ShallCodegen)
    return ConstantFP::get(*TheContext, APFloat(0.0f));
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

 
  }


  return ConstantFP::get(*TheContext, APFloat(0.0));
}





Value *Conv2dExprAST::codegen() {
  if (not ShallCodegen)
    return ConstantFP::get(*TheContext, APFloat(0.0f));

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
    Value *InitVal;
    if (Init) {
      InitVal = Init->codegen();
      if (!InitVal)
        return nullptr;
    } else { // If not specified, use 0.0.
      InitVal = ConstantFP::get(*TheContext, APFloat(0.0));
    }


    
    int is_obj_attr_or_self = 0;
    if (GetSelf()!="false")
      is_obj_attr_or_self=1;
    
    std::cout << "Parsing Conv2d var for: " << VarName << "\n";

    Builder->CreateCall(TheModule->getFunction("CreateConv2dOnDemand"),
                                              {Builder->CreateGlobalString(VarName),
                                               ConstantInt::get(Type::getInt32Ty(*GlobalContext), is_obj_attr_or_self),
                                               Builder->CreateGlobalString(TensorInit),
                                               C->codegen(), OC->codegen(), Ks->codegen(), Stride->codegen(),
                                               Padding->codegen()});


  }

  // Codegen the body that is contained by the in expression


  return ConstantFP::get(*TheContext, APFloat(0.0));
}




Value *CallExprAST::codegen() {
  if (not ShallCodegen)
    return ConstantFP::get(*TheContext, APFloat(0.0f));
  // Look up the name in the global module table.
  std::string tgt_function = Callee;
  

  Function *TheFunction = Builder->GetInsertBlock()->getParent();
  std::string functionName = TheFunction->getName().str();
  std::string tgt_function_name;

  int nested_function;
  if (functionName=="__anon_expr" || starts_with(functionName.c_str(), "__async_"))
    nested_function=0;
  else
    nested_function=1;

  
  if(Class!="None")
  {
    if (!in_str(tgt_function, tensor_methods))
      tgt_function = Class+tgt_function;
    Builder->CreateCall(TheModule->getFunction("FirstArgOnDemand"),
                                                  {Builder->CreateGlobalString(PreDot),
                                                   ConstantInt::get(Type::getInt32Ty(*TheContext), nested_function)});
    
  }

  //std::cout << "\nCalling function: " << tgt_function <<"\n\n";

  Function *CalleeF;
  if (!IsVarForward)
  {
    CalleeF = getFunction(tgt_function);
    if (!CalleeF)
    {
      std::string _error = "Função referenciada, "+ tgt_function +", ainda não foi declarada";
      return LogErrorV(_error);
    }

    tgt_function_name = CalleeF->getName().str();


    // If argument mismatch error.
    if ((CalleeF->arg_size()) != Args.size() && !in_str(tgt_function_name, vararg_methods))
      return LogErrorV("Parâmetros passados incorretos.");
  }


  std::vector<Value *> ArgsV;  
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
  
  
  Value * ret = ConstantFP::get(*TheContext, APFloat(0.0f));
  
  //std::cout << "\nCreate call: "  << tgt_function_name << " from parent: " << functionName << "\n\n";
  if (CalleeOverride=="none")
    ret = Builder->CreateCall(CalleeF, ArgsV, "calltmp");
  else
  {
    //std::cout << "Override: " << CalleeOverride << "\n";
    if (CalleeOverride=="Conv2d")
    {
      CalleeF = getFunction("ConvForward2d");
      Value *conv_name = Builder->CreateGlobalString(tgt_function);
      Value *is_attr = ConstantInt::get(Type::getInt32Ty(*GlobalContext), (int)(PreDot=="self"));
      ArgsV.push_back(conv_name);
      ArgsV.push_back(is_attr);
      ret = Builder->CreateCall(CalleeF, ArgsV, "calltmp");


      std::cout << "Load dims for conv: " << tgt_function << "\n";
      
      Value *dims_ptr = Builder->CreateCall(getFunction("LoadDimsConv"), 
                          {conv_name, is_attr});
      DimsPtr = Builder->CreateAlloca(int8PtrTy);
      Builder->CreateStore(dims_ptr, DimsPtr);


    }
  }

  if (in_str(tgt_function_name, activation_functions))
  {
    Value *dims_ptr = Builder->CreateCall(TheModule->getFunction("LoadDims"),
                                          {Builder->CreateGlobalString(Args[0]->GetName())});
    DimsPtr = Builder->CreateAlloca(int8PtrTy);
    Builder->CreateStore(dims_ptr, DimsPtr);
  }
    
  if(Class!="None")
    Builder->CreateCall(TheModule->getFunction("DimnishFirstArgOnDemand"),
                                                  {Builder->CreateGlobalString(PreDot),
                                                   ConstantInt::get(Type::getInt32Ty(*TheContext), nested_function)});
                                            
  return ret;
}



Function *PrototypeAST::codegen() {
  if (not ShallCodegen)
    return nullptr;
  // Make the function type:  float(float,float) etc.

  std::vector<Type *> types;
  for (auto &type : Types)
  {
    if (type=="s")
      types.push_back(int8PtrTy);
    else
      types.push_back(Type::getFloatTy(*TheContext));
  }

  //std::vector<Type *> Floats(Args.size(), Type::getFloatTy(*TheContext));
  
  /*
  if (Args.size()>0)
    if (Args[0]=="self")
      Floats[0] = Type::getInt8Ty(*TheContext)->getPointerTo();
  */

  FunctionType *FT = FunctionType::get(Type::getFloatTy(*TheContext), types, false);
  

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
  if (not ShallCodegen)
    return nullptr;
  
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
    
    if (!in_str(Arg.getName().str(), tensorVars))
    {
      AllocaInst *Alloca = CreateEntryBlockAlloca(TheFunction, Arg.getName());

      // Store the initial value into the alloca.
      Builder->CreateStore(&Arg, Alloca);

      // Add arguments to variable symbol table.
      NamedValues[std::string(Arg.getName())] = Alloca;
    }
    
  }
  //std::cout << "\n\n";


  Value *RetVal;

  for (auto &body : Body)
    RetVal = body->codegen();

  if (RetVal) {
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
  //std::cout << "\nINITIALIZING A NEW MODULE"  << "\n";

  // Open a new context and module.
  TheContext = std::make_unique<LLVMContext>();
  TheModule = std::make_unique<Module>("my cool jit", *TheContext);
  TheModule->setDataLayout(TheJIT->getDataLayout());

  //std::cout << "Initialize Module\n";
  // todo: It's creating one initialize for each ";" (top level expression).

  // Create a new builder for the module.
  Builder = std::make_unique<IRBuilder<>>(*TheContext);

  floatPtrTy = Type::getFloatTy(*TheContext)->getPointerTo();
  int8PtrTy = Type::getInt8Ty(*TheContext)->getPointerTo();
  ShallCodegen = true;

  //===----------------------------------------------------------------------===//
  // Tensor -- Scalar   Operations
  //===----------------------------------------------------------------------===//

  // char *, float, int
  FunctionType *CudaScalarMultTy = FunctionType::get(
      floatPtrTy,
      {floatPtrTy, int8PtrTy, Type::getFloatTy(*TheContext)}, 
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
      floatPtrTy,
      {floatPtrTy, int8PtrTy, Type::getFloatTy(*TheContext)}, 
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
      floatPtrTy,
      {floatPtrTy, int8PtrTy, Type::getFloatTy(*TheContext)}, 
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
      floatPtrTy,
      {floatPtrTy, int8PtrTy, Type::getFloatTy(*TheContext)}, 
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
      floatPtrTy,
      {Type::getInt8Ty(*TheContext)->getPointerTo(),
       Type::getInt8Ty(*TheContext)->getPointerTo(),
       Type::getInt32Ty(*TheContext),
       floatPtrTy,
       floatPtrTy,
       int8PtrTy,
       int8PtrTy}, 
      false // Not vararg
  );

  Function::Create(
    CudaMultTy,
    Function::ExternalLinkage, // Linkage (e.g., external for linking with other modules)
    "CudaMult", // Function name
    TheModule.get() // Module to which the function belongs
  );



  FunctionType *LoadTensorTy = FunctionType::get(
      int8PtrTy,
      {floatPtrTy}, 
      false // Not vararg
  );

  Function::Create(
    LoadTensorTy,
    Function::ExternalLinkage, // Linkage (e.g., external for linking with other modules)
    "LoadTensor", // Function name
    TheModule.get() // Module to which the function belongs
  );
  

  FunctionType *PrintTensorFTy = FunctionType::get(
      Type::getFloatTy(*TheContext),
      {floatPtrTy,
       Type::getInt32Ty(*TheContext),
       Type::getInt32Ty(*TheContext),}, 
      false // Not vararg
  );

  Function::Create(
    PrintTensorFTy,
    Function::ExternalLinkage, // Linkage (e.g., external for linking with other modules)
    "PrintTensorF", // Function name
    TheModule.get() // Module to which the function belongs
  );



  FunctionType *LoadDimsTy = FunctionType::get(
      int8PtrTy,
      {int8PtrTy},
      false
  );

  Function::Create(
    LoadDimsTy,
    Function::ExternalLinkage,
    "LoadDims",
    TheModule.get()
  );


  FunctionType *LoadDimsConvTy = FunctionType::get(
      int8PtrTy,
      {int8PtrTy,
       Type::getInt32Ty(*TheContext)}, 
      false // Not vararg
  );

  Function::Create(
    LoadDimsConvTy,
    Function::ExternalLinkage, 
    "LoadDimsConv", 
    TheModule.get() 
  );


  FunctionType *PrintDimsTy = FunctionType::get(
      Type::getVoidTy(*TheContext),
      {int8PtrTy}, 
      false // Not vararg
  );

  Function::Create(
    PrintDimsTy,
    Function::ExternalLinkage, // Linkage (e.g., external for linking with other modules)
    "PrintDims", // Function name
    TheModule.get() // Module to which the function belongs
  );

  

  FunctionType *NewDimsOnMultTy = FunctionType::get(
      int8PtrTy,
      {int8PtrTy,
       int8PtrTy}, 
      false // Not vararg
  );

  Function::Create(
    NewDimsOnMultTy,
    Function::ExternalLinkage, // Linkage (e.g., external for linking with other modules)
    "NewDimsOnMult", // Function name
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
    "backprop",
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
      {Type::getInt8Ty(*TheContext)->getPointerTo()},
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
      floatPtrTy,
      {int8PtrTy},
      false
  );
  Function::Create(
    softmaxTy,
    Function::ExternalLinkage,
    "softmax",
    TheModule.get()
  );

  //char *
  FunctionType *reluTy = FunctionType::get(
      floatPtrTy,
      {int8PtrTy},
      false
  );
  Function::Create(
    reluTy,
    Function::ExternalLinkage,
    "relu",
    TheModule.get()
  );

  //char *
  FunctionType *geluTy = FunctionType::get(
      floatPtrTy,
      {int8PtrTy},
      false
  );
  Function::Create(
    geluTy,
    Function::ExternalLinkage,
    "gelu",
    TheModule.get()
  );

  
  //char *
  FunctionType *conv2dTy = FunctionType::get(
      Type::getFloatTy(*TheContext),
      {PointerType::get(Type::getInt8Ty(*TheContext),0), Type::getFloatTy(*TheContext),Type::getFloatTy(*TheContext),Type::getFloatTy(*TheContext),Type::getFloatTy(*TheContext),Type::getFloatTy(*TheContext),},
      false
  );
  Function::Create(
    conv2dTy,
    Function::ExternalLinkage,
    "Conv_2d",
    TheModule.get()
  );

  //char *, char *, int
  FunctionType *conv2dForwardTy = FunctionType::get(
      floatPtrTy,
      {PointerType::get(Type::getInt8Ty(*TheContext),0), PointerType::get(Type::getInt8Ty(*TheContext),0), Type::getInt32Ty(*TheContext)},
      false
  );
  Function::Create(
    conv2dForwardTy,
    Function::ExternalLinkage,
    "ConvForward2d",
    TheModule.get()
  );

  // float
  FunctionType *onehotTy = FunctionType::get(
      floatPtrTy,
      {Type::getFloatTy(*TheContext)},
      false
  );
  Function::Create(
    onehotTy,
    Function::ExternalLinkage,
    "onehot",
    TheModule.get()
  );

  // 
  FunctionType *sumTy = FunctionType::get(
      Type::getFloatTy(*TheContext),
      {},
      false
  );
  Function::Create(
    sumTy,
    Function::ExternalLinkage,
    "sum",
    TheModule.get()
  );

  // 
  FunctionType *meanTy = FunctionType::get(
      Type::getFloatTy(*TheContext),
      {},
      false
  );
  Function::Create(
    meanTy,
    Function::ExternalLinkage,
    "mean",
    TheModule.get()
  );

  // 
  FunctionType *maxTy = FunctionType::get(
      Type::getFloatTy(*TheContext),
      {},
      false
  );
  Function::Create(
    maxTy,
    Function::ExternalLinkage,
    "max",
    TheModule.get()
  );

  
  // char *, floats, Vararg
  FunctionType *viewTy = FunctionType::get(
      Type::getFloatTy(*TheContext),
      {Type::getInt8Ty(*TheContext)->getPointerTo(), Type::getFloatTy(*TheContext),Type::getFloatTy(*TheContext),Type::getFloatTy(*TheContext),Type::getFloatTy(*TheContext),Type::getFloatTy(*TheContext),Type::getFloatTy(*TheContext),Type::getFloatTy(*TheContext),Type::getFloatTy(*TheContext)},
      true // Vararg
  );
  Function::Create(
    viewTy,
    Function::ExternalLinkage,
    "view",
    TheModule.get()
  );
  

  //===----------------------------------------------------------------------===//
  // Loss CUDA Ops
  //===----------------------------------------------------------------------===//


  // char *, char *
  FunctionType *cross_entropyTy = FunctionType::get(
      Type::getFloatTy(*TheContext),
      {Type::getInt8Ty(*TheContext)->getPointerTo(), Type::getInt8Ty(*TheContext)->getPointerTo()}, 
      false // Not vararg
  );
  Function::Create(
    cross_entropyTy,
    Function::ExternalLinkage, // Linkage (e.g., external for linking with other modules)
    "cross_entropy", // Function name
    TheModule.get() // Module to which the function belongs
  );

  //===----------------------------------------------------------------------===//
  // DATASET Ops
  //===----------------------------------------------------------------------===//


  // float, chars *, ... 
  FunctionType *yieldTy = FunctionType::get(
      Type::getFloatTy(*TheContext),
      {Type::getFloatTy(*TheContext), Type::getInt8Ty(*TheContext)->getPointerTo(), Type::getInt8Ty(*TheContext)->getPointerTo(),Type::getInt8Ty(*TheContext)->getPointerTo(),Type::getInt8Ty(*TheContext)->getPointerTo(),Type::getInt8Ty(*TheContext)->getPointerTo(),Type::getInt8Ty(*TheContext)->getPointerTo()},
      true // vararg
  );
  Function::Create(
    yieldTy,
    Function::ExternalLinkage,
    "Datasetyield",
    TheModule.get()
  );

  // float, char *, ... 
  FunctionType *init_datasetTy = FunctionType::get(
      Type::getFloatTy(*TheContext),
      {Type::getFloatTy(*TheContext)},
      false
  );
  Function::Create(
    init_datasetTy,
    Function::ExternalLinkage,
    "Datasetinit_dataset",
    TheModule.get()
  );


  //===----------------------------------------------------------------------===//
  // File Handling Ops
  //===----------------------------------------------------------------------===//
  

  
  // char *
  FunctionType *load_imgTy = FunctionType::get(
      PointerType::get(Type::getFloatTy(*GlobalContext), 0),
      {PointerType::get(Type::getInt8Ty(*GlobalContext), 0)},
      false // Not vararg
  );
  Function::Create(
    load_imgTy,
    Function::ExternalLinkage,
    "load_img",
    TheModule.get()
  );
  

  // char *
  FunctionType *load_preprocess_imgTy = FunctionType::get(
      Type::getFloatTy(*TheContext),
      {Type::getInt8Ty(*TheContext)->getPointerTo(), Type::getInt8Ty(*TheContext)->getPointerTo()},
      false
  );
  Function::Create(
    load_preprocess_imgTy,
    Function::ExternalLinkage,
    "load_preprocess_img",
    TheModule.get()
  );


  //===----------------------------------------------------------------------===//
  // Parallel Ops
  //===----------------------------------------------------------------------===//


  //  
  FunctionType *sleepTy = FunctionType::get(
      Type::getFloatTy(*TheContext),
      {Type::getFloatTy(*TheContext)},
      false
  );
  Function::Create(
    sleepTy,
    Function::ExternalLinkage,
    "sleep",
    TheModule.get()
  );


  FunctionType *PrintVoidTy = FunctionType::get(
      Type::getVoidTy(*TheContext),
      {Type::getInt8Ty(*TheContext)->getPointerTo()},
      false
  );
  Function::Create(
    PrintVoidTy,
    Function::ExternalLinkage,
    "PrintVoid",
    TheModule.get()
  );



  auto int8PtrTy = Type::getInt8Ty(*TheContext)->getPointerTo();
  auto pthreadPtr = Type::getInt8Ty(*GlobalContext)->getPointerTo();
  auto pthreadPtrTy = pthreadPtr->getPointerTo();

  // (void *) fn (void * arg)
  FunctionType *funVoidPtrint8PtrTy = FunctionType::get(
    int8PtrTy, {int8PtrTy},
    false);
  // int pthread_create(pthread_t * thread, const pthread_attr_t * attr,
  //                  void * (*start_routine)(void *), void * arg)
  // using void * in place of pthread_attr_t *
  FunctionType *pthreadCreateTy = FunctionType::get(
                                      Type::getVoidTy(*TheContext),
                                      {pthreadPtrTy,
                                       int8PtrTy,
                                       (funVoidPtrint8PtrTy)->getPointerTo(),
                                       int8PtrTy},
                                      false
                                    );
  /*                                  
  Function::Create(
    pthreadCreateTy,
    Function::ExternalLinkage,
    "pthread_create",
    TheModule.get()
  );
  */
  
  TheModule->getOrInsertFunction("pthread_create_aux", pthreadCreateTy);


  // int pthread_join(pthread_t thread, void **value_ptr)
  FunctionType *pthreadJoinTy = FunctionType::get(
    Type::getVoidTy(*TheContext),
    {pthreadPtr},
    false);
  
  /*
  Function::Create(
    pthreadJoinTy,
    Function::ExternalLinkage,
    "pthread_join",
    TheModule.get()
  );
  */ 
  TheModule->getOrInsertFunction("pthread_join_aux", pthreadJoinTy);


  //===----------------------------------------------------------------------===//
  // Str Ops
  //===----------------------------------------------------------------------===//


  // char *
  FunctionType *globTy = FunctionType::get(
      Type::getInt8Ty(*TheContext)->getPointerTo(),
      {Type::getInt8Ty(*TheContext)->getPointerTo()},
      false // Not vararg
  );
  Function::Create(
    globTy,
    Function::ExternalLinkage,
    "_glob_b_",
    TheModule.get()
  );



  FunctionType *PrintFloatTy = FunctionType::get(
      Type::getVoidTy(*TheContext),
      {Type::getFloatTy(*TheContext)},
      false 
  );
  TheModule->getOrInsertFunction("PrintFloat", PrintFloatTy);

  FunctionType *UnbugFloatTy = FunctionType::get(
      Type::getVoidTy(*TheContext),
      {Type::getFloatTy(*TheContext)},
      false 
  );
  TheModule->getOrInsertFunction("UnbugFloat", UnbugFloatTy);

  // char *
  FunctionType *PrintStrTy = FunctionType::get(
      Type::getFloatTy(*TheContext),
      {Type::getInt8Ty(*TheContext)->getPointerTo()}, 
      false 
  );
  TheModule->getOrInsertFunction("PrintStr", PrintStrTy);


  // char *
  FunctionType *shuffle_strTy = FunctionType::get(
      Type::getInt8Ty(*TheContext)->getPointerTo(),
      {Type::getInt8Ty(*TheContext)->getPointerTo()}, 
      false 
  );
  Function::Create(
    shuffle_strTy,
    Function::ExternalLinkage, 
    "shuffle_str", 
    TheModule.get() 
  );


  //===----------------------------------------------------------------------===//
  // Other Ops
  //===----------------------------------------------------------------------===//


  // char *, int
  FunctionType *FirstArgOnDemandTy = FunctionType::get(
      //PointerType::get(Type::getVoidTy(*TheContext), 0),
      Type::getFloatTy(*TheContext),
      {Type::getInt8Ty(*TheContext)->getPointerTo(), Type::getInt32Ty(*TheContext)},
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
      {Type::getInt8Ty(*TheContext)->getPointerTo(), Type::getInt32Ty(*TheContext)},
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
      Type::getInt8Ty(*TheContext)->getPointerTo(),
      {Type::getInt8Ty(*TheContext)->getPointerTo(), Type::getInt8Ty(*TheContext)->getPointerTo()},
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
      Type::getInt8Ty(*TheContext)->getPointerTo(),
      //{Type::getInt8Ty(*TheContext)->getPointerTo(), Type::getInt8Ty(*TheContext)->getPointerTo()},
      {Type::getInt8Ty(*TheContext)->getPointerTo()},
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
      {Type::getInt8Ty(*TheContext)->getPointerTo(), Type::getFloatTy(*TheContext)},
      false // Not vararg
  );
  Function::Create(
    StoreOnDemandTy,
    Function::ExternalLinkage,
    "StoreOnDemand",
    TheModule.get()
  );


    // char *, float
  FunctionType *StoreStrOnDemandTy = FunctionType::get(
      Type::getFloatTy(*TheContext),
      {Type::getInt8Ty(*TheContext)->getPointerTo(), Type::getInt8Ty(*TheContext)->getPointerTo()},
      false // Not vararg
  );
  Function::Create(
    StoreStrOnDemandTy,
    Function::ExternalLinkage,
    "StoreStrOnDemand",
    TheModule.get()
  );


  // char *
  FunctionType *LoadOnDemandTy = FunctionType::get(
      //PointerType::get(Type::getVoidTy(*TheContext), 0),
      Type::getFloatTy(*TheContext),
      {Type::getInt8Ty(*TheContext)->getPointerTo()},
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
      //{Type::getInt8Ty(*TheContext)->getPointerTo()},
      false // Not vararg
  );
  Function::Create(
    StoreDimsOnDemandTy,
    Function::ExternalLinkage,
    "StoreDimsOnDemand",
    TheModule.get()
  );


  // char *, int, char *
  FunctionType *CreateTensorOnDemandTy = FunctionType::get(
      //PointerType::get(Type::getVoidTy(*TheContext), 0),
      Type::getFloatTy(*TheContext),
      {Type::getInt8Ty(*TheContext)->getPointerTo(),
       Type::getInt32Ty(*TheContext),
       Type::getInt8Ty(*TheContext)->getPointerTo()},
      false // Not vararg
  );
  Function::Create(
    CreateTensorOnDemandTy,
    Function::ExternalLinkage,
    "CreateTensorOnDemand",
    TheModule.get()
  );


// char *, int, char *, int, int, int, int, int
  FunctionType *CreateConv2dOnDemandTy = FunctionType::get(
      //PointerType::get(Type::getVoidTy(*TheContext), 0),
      Type::getFloatTy(*TheContext),
      {Type::getInt8Ty(*TheContext)->getPointerTo(),
       Type::getInt32Ty(*TheContext),
       Type::getInt8Ty(*TheContext)->getPointerTo(),
       Type::getFloatTy(*TheContext),
       Type::getFloatTy(*TheContext),
       Type::getFloatTy(*TheContext),
       Type::getFloatTy(*TheContext),
       Type::getFloatTy(*TheContext),
       Type::getFloatTy(*TheContext),
       Type::getFloatTy(*TheContext)},
      false // Not vararg
  );
  Function::Create(
    CreateConv2dOnDemandTy,
    Function::ExternalLinkage,
    "CreateConv2dOnDemand",
    TheModule.get()
  );




  // float, char *
  FunctionType *CallToStoredValuesTy = FunctionType::get(
      PointerType::get(Type::getFloatTy(*TheContext), 0),
      {Type::getFloatTy(*TheContext), Type::getInt8Ty(*TheContext)->getPointerTo()}, 
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
      {int8PtrTy,
       floatPtrTy,
       int8PtrTy}, 
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
      {Type::getInt8Ty(*TheContext)->getPointerTo()}, 
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

std::vector<std::thread> all_threads;

static void CodegenTopLevelExpression(std::unique_ptr<FunctionAST> &FnAST) {

    auto *FnIR =  FnAST->codegen();

    /*
    fprintf(stderr, "\nRead top-level expression:");
    FnIR->print(errs());
    fprintf(stderr, "\n\n");
    */

    // Create a ResourceTracker for memory managment
    // anonymous expression -- that way we can free it after executing.
    auto RT = TheJIT->getMainJITDylib().createResourceTracker();

    auto TSM = ThreadSafeModule(std::move(TheModule), std::move(TheContext));
    ExitOnErr(TheJIT->addModule(std::move(TSM), RT));
    // Add IR module


    InitializeModule();
    //std::cout << "Finished new module." << "\n\n";

    // Points __anon_expr
    auto Sym = ExitOnErr(TheJIT->lookup("__anon_expr"));
    //assert(Sym && "Function not found");
      
    //std::cout << "Jit lookup" << "\n";

    // Get the symbol's address and cast it to the right type (takes no
    // arguments, returns a float) so we can call it as a native function.
    auto *FP = Sym.getAddress().toPtr<float (*)()>();
    auto fp = FP();
    //std::cout << "Jit print" << "\n";
      
    //std::cout << "\nResult times 5 is " << fp*5 << "\n";
    fprintf(stderr, "%.2f\n", fp);

    // Delete the anonymous expression module from the JIT.
    ExitOnErr(RT->remove());    
}



static void HandleTopLevelExpression() {
  // Evaluate a top-level expression into an anonymous function.
  
  if (std::unique_ptr<FunctionAST> FnAST = ParseTopLevelExpr()) {
    CodegenTopLevelExpression(std::ref(FnAST));

  
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
        //LogError("Tab inesperado encontrado\n");
        getNextToken();
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



  
  cudnnCreate(&cudnn);



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
