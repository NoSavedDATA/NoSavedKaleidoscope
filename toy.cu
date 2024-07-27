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
#include <cstring>
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
#include <float.h>


// Cuda
#include <stdio.h>
#include <stdlib.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cublasLt.h>
#include <omp.h>

#include "include/cu_commons.h"


pthread_mutex_t mtx_self_float;
pthread_mutex_t mutex;

float TERMINATE_VARARG = -40370000000.0f;

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

bool ends_with(std::string str_input, std::string str_end)
{
  return str_input.size() >= str_end.size() && str_input.compare(str_input.size() - str_end.size(), str_end.size(), str_end) == 0;
}
bool begins_with(const std::string& str_input, const std::string& str_start) {
    return str_input.size() >= str_start.size() && str_input.compare(0, str_start.size(), str_start) == 0;
}
bool contains_str(const std::string& str_input, const std::string& str_sub) {
    return str_input.find(str_sub) != std::string::npos;
}
std::string remove_substring(const std::string& str, const std::string& substr) {
    std::string result = str;  // Copy the original string
    size_t pos = result.find(substr);
    if (pos != std::string::npos) {
        result.erase(pos, substr.length());
    }
    return result;
}
bool starts_with(const char* str, const char* sub) {
  return strncmp(str, sub, strlen(sub)) == 0;
}

char *str_to_char(std::string str)
{
    char *c_str = new char[str.length() + 1]; // +1 for the null terminator
    std::strcpy(c_str, str.c_str());

    return c_str;
}





int count_pattern(const std::string& text, const std::string& pattern) {
  int count = 0;
  size_t pos = 0;

  std::cout << "Trying to count"  << "\n";
  // Iterate while finding occurrences of the pattern
  while ((pos = text.find(pattern, pos)) != std::string::npos) {
    count++;
    pos += pattern.length(); // Move to the character after the found pattern
  }

  return count;
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

bool in_int(int value, const std::vector<int>& list) {
    return std::find(list.begin(), list.end(), value) != list.end();
}

bool in_float_vec(float value, const std::vector<float>& list) {
    return std::find(list.begin(), list.end(), value) != list.end();
}

bool in_float_ptr_vec(float *value, const std::vector<float *>& list) {
    return std::find(list.begin(), list.end(), value) != list.end();
}
std::vector<std::string> concat_str_vec(std::vector<std::string> l, std::vector<std::string>r)
{
  std::vector<std::string> concatenated_vectors = l;
  concatenated_vectors.insert(concatenated_vectors.end(), r.begin(), r.end());
  return concatenated_vectors;
}


// Tensor related
std::vector<std::string> return_tensor_functions, return_tensor_methods, return_tensor_fn,
return_pinned_methods, vararg_methods, string_methods, native_methods, native_functions, native_fn, tensor_inits;






PointerType *floatPtrTy, *int8PtrTy;
bool ShallCodegen = true;

//===----------------------------------------------------------------------===//
// Lexer
//===----------------------------------------------------------------------===//






// The lexer returns tokens [0-255] if it is an unknown character, otherwise one
// of these for known words.
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
  tok_str = -40, // ""

  // control
  tok_if = -6,
  tok_then = -7,
  tok_else = -8,
  tok_for = -9,
  tok_while = -10,
  tok_async = -22,
  tok_async_finish = -23,
  tok_lock = -26,
  tok_unlock = -27,
  tok_tab = 9,
  tok_return = -32,
  tok_as = -33,

  // operators
  tok_binary = -11,
  tok_unary = -12,
  tok_equal = -28,
  tok_diff = -34,
  tok_higher_eq = -35,
  tok_minor_eq = -36,
  tok_mod = -29,


  tok_space = -14,

  
  // var definition
  tok_var = -15,
  tok_tensor = -16,
  tok_param = -44,
  tok_pinned_tensor = -25,
  tok_var_str = -17,
  tok_str_vec = -24,
  tok_float_vec = -31,
  tok_attr_var = -18,
  tok_attr_tensor = -19,
  tok_conv2d = -21,
  tok_maxpool2d = -41,
  tok_avgpool2d = -42,
  tok_batchnorm2d = -43,
  tok_vec = -37,
  tok_post_class_attr_attr = -38,
  tok_post_class_attr_identifier = -39,


};


enum Types {
  type_float = 0,
  type_tensor = 1,
  type_pinned_tensor = 2,
  type_object = 3
};

enum NameSolverTypes {
  type_self = 0,
  type_attr = 1,
  type_vec = 2,
  type_var = 3,
  type_object_name = 4,
  type_object_vec = 5
};

enum NN_Mode {
  eval_mode = 0,
  training_mode = 1,
};

enum BackwardTypes {
  create_tensor_op=-3,
  leaf=-2,
  tensor_leaf = -1,
  weight_leaf = 0,
  bias_leaf = 1,
  attribution = 2,
  mult_op = 3,
  conv2d = 4,
  maxpool2d = 5,
  batchnorm2d = 6,
  relu_op = 7,
  gelu_op = 8,
  sigmoid_op = 9,
  tanh_op = 10,
  cross_entropy_op = 11,
  add_op = 12,
  sub_op = 25,
  hadamard_op = 13,
  div_op=26,
  softmax_op = 14,
  onehot_op = 15,
  view_op = 16,
  sum_op = 17,
  mean_op = 18,
  max_op = 19,
  argmax_op = 20,
  topk_op = 21,
  clip_op = 22,
  gpu_op = 23,
  cpu_op = 24,
  equal_op = 27,
};

int nn_mode=training_mode;
std::vector<int> leaf_ops, loss_ops, gradless_ops;


std::map<int, std::string> token_to_string = {
  { tok_eof, "eof" },

  // functions/classes
  { tok_def, "def" },
  { tok_class, "class" },
  { tok_self, "self" },
  { tok_class_attr, "object attr" },
  { tok_extern, "extern" },

  // primary
  { tok_identifier, "tok identifier" },
  { tok_number, "tok number" },
  { tok_str, "tok str `` ''" },
  { tok_str_vec, "tok str vector" },
  { tok_float_vec, "tok float vec" },

  

  // control
  { tok_if, "if" },
  { tok_then, "then" },
  { tok_else, "else" },
  { tok_for, "for" },
  { tok_while, "while" },
  { tok_async, "async" },
  { tok_async_finish, "finish" },
  { tok_tab, "tok tab" },
  { tok_return, "tok return"},
  { tok_as, "tok as"},
  { tok_vec, "tok vec"},


  // operators
  { tok_binary, "tok binary" },
  { tok_unary,"tok unary" },


  { tok_space, "tok_space" },

  { tok_post_class_attr_attr, ".attr." },
  { tok_post_class_attr_identifier, ".identifier" },
  
  // var definition
  { tok_var, "float" },
  { tok_tensor, "tensor" },
  { tok_var_str, "var str" },
  { tok_attr_var, "tok attr var" },
  { tok_attr_tensor, "tok attr tensor" },
  { tok_conv2d, "Conv2d" },
  { tok_maxpool2d, "MaxPool2d"},
  { tok_avgpool2d, "AvgPool2d"},
  { tok_batchnorm2d, "BatchNorm2d"},
  

  { 10, "tok space"},

  
  { 40, "(" },
  { 41, ")" },

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

  { 91, "[" },
  { 93, "]" },

  { tok_equal, "==" },
  { tok_diff, "!=" },
  { tok_higher_eq, ">=" },
  { tok_minor_eq, "<=" },
  { tok_mod, "//" },


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
std::vector<char> ops = {'+', '-', '*', '/', '@', '=', '>', '<', 10, -14, ',', '(', ')', ';', tok_equal, tok_diff, tok_higher_eq, tok_minor_eq};
std::vector<char> terminal_tokens = {';', tok_def, tok_extern, tok_class};


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

int LineCounter = 1;

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
    
  if (LastChar=='[')
  {
    LastChar = getchar();
    return '[';
  }

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

  
  if (LastChar=='.')
  {
    LastChar = getchar(); // eat .
    IdentifierStr = LastChar;
    bool name_ok=true;
    while (name_ok)
    {
      LastChar = getchar();
      
      
      if(isalnum(LastChar) || LastChar=='_')
        IdentifierStr += LastChar;
      else
        name_ok = false;

      
      if (LastChar=='.')
      {
        LastChar = getchar();
        return tok_post_class_attr_attr;
      }
    }
    
    return tok_post_class_attr_identifier;
  }

  if (isalpha(LastChar) || LastChar=='_') { // identifier: [a-zA-Z][a-zA-Z0-9]*
    IdentifierStr = LastChar;
    bool name_ok=true;
    while (name_ok)
    {
      LastChar = getchar();

      if (LastChar=='[')
        break;
      
      if(isalnum(LastChar) || LastChar=='_')
        IdentifierStr += LastChar;
      else
        name_ok = false;

      if (IdentifierStr == "tensor")
      {
        LastChar = getchar();
        return tok_tensor;
      }
      if (IdentifierStr == "param")
      {
        LastChar = getchar();
        return tok_param;
      }
      if (IdentifierStr == "pinned_tensor")
      {
        LastChar = getchar();
        return tok_pinned_tensor;
      }
      if (IdentifierStr == "Conv2d")
      {
        LastChar = getchar();
        return tok_conv2d;
      }
      if (IdentifierStr == "MaxPool2d")
      {
        LastChar = getchar();
        return tok_maxpool2d;
      }
      if (IdentifierStr == "AvgPool2d")
      {
        LastChar = getchar();
        return tok_avgpool2d;
      }
      if (IdentifierStr == "BatchNorm2d")
      {
        LastChar = getchar();
        return tok_batchnorm2d;
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
    if (IdentifierStr == "lock")
      return tok_lock;
    if (IdentifierStr == "unlock")
      return tok_unlock;
    if (IdentifierStr == "binary")
      return tok_binary;
    if (IdentifierStr == "unary")
      return tok_unary;
    if (IdentifierStr == "float")
      return tok_var;
    if (IdentifierStr == "glob")
      IdentifierStr = "_glob_b_";
    if (IdentifierStr == "tanh")
      IdentifierStr = "_tanh";
    if (IdentifierStr == "str")
      return tok_var_str;
    if (IdentifierStr == "str_vec")
      return tok_str_vec;
    if (IdentifierStr == "float_vec")
      return tok_float_vec;
    if (IdentifierStr == "return")
      return tok_return;
    if (IdentifierStr == "as")
      return tok_as;
    if (IdentifierStr == "vec")
      return tok_vec;
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
    

    int seen_spaces=0;

    while(LastChar==10 || LastChar==tok_tab || LastChar==32) {
      if(ThisChar==10)
      {
        LineCounter += 1;
        LastSeenTabs = SeenTabs;
        SeenTabs = 0;
        seen_spaces = 0;
      }
      if (LastChar==tok_tab)
        SeenTabs+=1;
      if (LastChar==32)
        seen_spaces+=1;
      if (seen_spaces==3)
      {
        seen_spaces=0;
        SeenTabs+=1;
      }

      ThisChar = (int)LastChar;
      LastChar = getchar(); 
    }
    //std::cout << "\nThisChar: " << ThisChar << " LastChar " << LastChar << "\n";

    //std::cout << "New seen tabs: " << SeenTabs << "\n";
    return tok_space;
  }


  LastChar = getchar();
  int otherChar = LastChar;



  if (ThisChar=='=' && otherChar=='=')
  {
    LastChar = getchar();
    return tok_equal;
  }
  if (ThisChar=='!' && otherChar=='=')
  {
    LastChar = getchar();
    return tok_diff;
  }
  if (ThisChar=='>' && otherChar=='=')
  {
    LastChar = getchar();
    return tok_higher_eq;
  }
  if (ThisChar=='<' && otherChar=='=')
  {
    LastChar = getchar();
    return tok_minor_eq;
  }

  if((ThisChar=='/')&&(otherChar == '/')){
    LastChar = getchar();
    return 77;
  }

  //std::cout << "Post char: " << ReverseToken(ThisChar) << "\n";

  // else: return ascii number of the character.
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
  std::string ReturnType = "None";
  std::string Name = "Unnamed";
  bool isSelf = false;
  bool isAttribute = false;
  std::string _pre_dot = "";
  bool isVec = false;
  bool isVarLoad = false;
  bool SolverIncludeScope = true;

  Value *TensorPtr;


  virtual Value *codegen(Value *first_arg, Value *scope_str, Value *previous_scope) = 0;
  virtual void SetType(std::string Type) {
    this->Type=Type;
    this->ReturnType=Type;
  }
  virtual std::string GetType() {
    return Type;
  }
  virtual void SetReturnType(std::string ReturnType) {
    this->ReturnType=ReturnType;
  }

  virtual void SetIsVarLoad(bool isVarLoad) {
    this->isVarLoad=isVarLoad;
  }
  virtual bool GetIsVarLoad() {
    return isVarLoad;
  }

  virtual void SetSelf(bool Self) {
    this->isSelf=Self;
  }
  virtual bool GetSelf() {
    return isSelf;
  }

  virtual void SetSolverIncludeScope(bool SolverIncludeScope) {
    this->SolverIncludeScope=SolverIncludeScope;
  }
  virtual bool GetSolverIncludeScope() {
    return SolverIncludeScope;
  }

  virtual void SetIsAttribute(bool Attribute) {
    this->isAttribute=Attribute;
  }
  virtual bool GetIsAttribute() {
    return isAttribute;
  }
  

  virtual void SetPreDot(std::string pre_dot) {
    this->_pre_dot=pre_dot;
  }
  virtual std::string GetPreDot() {
    return _pre_dot;
  }

  virtual std::string GetName() {
    return Name;
  }
  virtual void SetName(std::string Name) {
    this->Name=Name;
  }

  
  virtual void SetIsVec(bool isVec) {
    this->isVec=isVec;
  }
  virtual bool GetIsVec() {
    return isVec;
  }

  // Tensor related
  virtual std::vector<float> GetDims() {
    return Dims;
  }
  virtual void SetDims(std::vector<float> Dims) {
    this->Dims=Dims;
  }
  virtual Value *GetTensorPtr() {
    return TensorPtr;
  }
  
};



class NameSolverAST : public ExprAST {

  public:
    std::vector<std::tuple<std::string, int, std::vector<std::unique_ptr<ExprAST>>>> Names;
    NameSolverAST(std::vector<std::tuple<std::string, int, std::vector<std::unique_ptr<ExprAST>>>> Names)
                    : Names(std::move(Names)) {} 
  

  Value *codegen(Value *first_arg, Value *scope_str, Value *previous_scope) override;
};


/// NumberExprAST - Expression class for numeric literals like "1.0".
class NumberExprAST : public ExprAST {
  float Val;

  public:
    NumberExprAST(float Val) : Val(Val) {
      this->SetType("float");
    } 

  Value *codegen(Value *first_arg, Value *scope_str, Value *previous_scope) override;
};



class StringExprAST : public ExprAST {
  std::string Val;

  public:
    StringExprAST(std::string Val) : Val(Val) {
      this->SetType("str");
    } 

  Value *codegen(Value *first_arg, Value *scope_str, Value *previous_scope) override;
};



/// VariableExprAST - Expression class for referencing a variable, like "a".
class ObjAttrExprAST : public ExprAST {
  std::unique_ptr<ExprAST> Body;
  std::string PreDot;

  public:
    ObjAttrExprAST(std::unique_ptr<ExprAST> Body, const std::string &PreDot)
                : Body(std::move(Body)), PreDot(PreDot) {}

    Value *codegen(Value *first_arg, Value *scope_str, Value *previous_scope) override;
};


/// VariableExprAST - Expression class for referencing a variable, like "a".
class VariableExprAST : public ExprAST {

  public:
    std::unique_ptr<ExprAST> NameSolver;
    VariableExprAST(std::unique_ptr<ExprAST> NameSolver, std::string Type) {
      this->isVarLoad = true;
      this->NameSolver = std::move(NameSolver);
      this->SetType(Type);
      this->NameSolver->SetType(Type);
    }

    Value *codegen(Value *first_arg, Value *scope_str, Value *previous_scope) override;
    const std::string &getName() const { return Name; }
    std::string GetName() override {
    return Name;
  }
};


class VecIdxExprAST : public ExprAST {
  
  public:
    std::unique_ptr<ExprAST> NameSolver;
    std::vector<std::unique_ptr<ExprAST>> Idx;

    VecIdxExprAST(std::unique_ptr<ExprAST> NameSolver, std::vector<std::unique_ptr<ExprAST>> Idx, std::string Type)
                  : Idx(std::move(Idx)) {
      this->isVarLoad = true;
      this->NameSolver = std::move(NameSolver);
      this->SetType(Type);
      this->NameSolver->SetType(Type);
    }

    Value *codegen(Value *first_arg, Value *scope_str, Value *previous_scope) override;
    const std::string &getName() const { return Name; }
    std::string GetName() override { return Name; }
};



class ObjectVecIdxExprAST : public ExprAST {

  public:
    std::unique_ptr<ExprAST> Vec, Idx;
    std::string _post_dot;

    ObjectVecIdxExprAST(std::unique_ptr<ExprAST> Vec, std::string _post_dot, std::unique_ptr<ExprAST> Idx)
                  : Vec(std::move(Vec)), _post_dot(_post_dot), Idx(std::move(Idx)) {
      this->isVarLoad = true;
    }

    Value *codegen(Value *first_arg, Value *scope_str, Value *previous_scope) override;
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

  Value *codegen(Value *first_arg, Value *scope_str, Value *previous_scope) override;
};

class StrExprAST : public ExprAST {

  public:
    std::vector<std::pair<std::string, std::unique_ptr<ExprAST>>> VarNames;
    StrExprAST(
        std::vector<std::pair<std::string, std::unique_ptr<ExprAST>>> VarNames)
        : VarNames(std::move(VarNames)) {
          this->SetType("str");
        }

  Value *codegen(Value *first_arg, Value *scope_str, Value *previous_scope) override;
};


class StrVecExprAST : public ExprAST {

  public:
    std::vector<std::pair<std::string, std::unique_ptr<ExprAST>>> VarNames;
    std::string Type;
    
    StrVecExprAST(
        std::vector<std::pair<std::string, std::unique_ptr<ExprAST>>> VarNames,
        std::string Type)
        : VarNames(std::move(VarNames)), Type(Type) {}

  Value *codegen(Value *first_arg, Value *scope_str, Value *previous_scope) override;
};


class ObjectExprAST : public VarExprAST {

public:
  std::unique_ptr<ExprAST> Init;

  ObjectExprAST(
      std::vector<std::pair<std::string, std::unique_ptr<ExprAST>>> VarNames,
      std::string Type,
      std::unique_ptr<ExprAST> Init)
      : VarExprAST(std::move(VarNames), std::move(Type)), Init(std::move(Init)) {}

  Value *codegen(Value *first_arg, Value *scope_str, Value *previous_scope) override;
};



class TensorExprAST : public VarExprAST {
  public:
    std::vector<std::unique_ptr<ExprAST>> V_Dims;
    std::string TensorInit;
    bool IsWeight;

    TensorExprAST(
      std::vector<std::pair<std::string, std::unique_ptr<ExprAST>>> VarNames,
      std::string Type,
      std::vector<std::unique_ptr<ExprAST>> V_Dims,
      const std::string &TensorInit, bool IsWeight)
      : VarExprAST(std::move(VarNames), std::move(Type)),
                   V_Dims(std::move(V_Dims)), TensorInit(TensorInit), IsWeight(IsWeight) {}

  Value *codegen(Value *first_arg, Value *scope_str, Value *previous_scope) override;
};


class PinnedTensorExprAST : public VarExprAST {
  public:
    std::vector<std::unique_ptr<ExprAST>> V_Dims;
    std::string TensorInit;

    PinnedTensorExprAST(
      std::vector<std::pair<std::string, std::unique_ptr<ExprAST>>> VarNames,
      std::string Type,
      std::vector<std::unique_ptr<ExprAST>> V_Dims,
      const std::string &TensorInit)
      : VarExprAST(std::move(VarNames), std::move(Type)),
                   V_Dims(std::move(V_Dims)), TensorInit(TensorInit) {}

  Value *codegen(Value *first_arg, Value *scope_str, Value *previous_scope) override;
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

  Value *codegen(Value *first_arg, Value *scope_str, Value *previous_scope) override;
};



class MaxPool2dExprAST : public VarExprAST {
  public:
    std::unique_ptr<ExprAST> Ks, Stride, Padding;

    MaxPool2dExprAST(
      std::vector<std::pair<std::string, std::unique_ptr<ExprAST>>> VarNames,
      std::string Type,
      std::unique_ptr<ExprAST> Ks,
      std::unique_ptr<ExprAST> Stride, std::unique_ptr<ExprAST> Padding)
      : VarExprAST(std::move(VarNames), std::move(Type)),
                   Ks(std::move(Ks)),
                   Stride(std::move(Stride)), Padding(std::move(Padding)) {}

  Value *codegen(Value *first_arg, Value *scope_str, Value *previous_scope) override;
};


class BatchNorm2dExprAST : public VarExprAST {
  public:
    std::unique_ptr<ExprAST> C;

    BatchNorm2dExprAST(
      std::vector<std::pair<std::string, std::unique_ptr<ExprAST>>> VarNames,
      std::string Type,
      std::unique_ptr<ExprAST> C)
      : VarExprAST(std::move(VarNames), std::move(Type)),
                   C(std::move(C)) {}

  Value *codegen(Value *first_arg, Value *scope_str, Value *previous_scope) override;
};


/// UnaryExprAST - Expression class for a unary operator.
class UnaryExprAST : public ExprAST {
  char Opcode;
  std::unique_ptr<ExprAST> Operand;

public:
  UnaryExprAST(char Opcode, std::unique_ptr<ExprAST> Operand)
      : Opcode(Opcode), Operand(std::move(Operand)) {}

  Value *codegen(Value *first_arg, Value *scope_str, Value *previous_scope) override;
};


/// BinaryExprAST - Expression class for a binary operator.
class BinaryExprAST : public ExprAST {
  char Op;
  std::unique_ptr<ExprAST> LHS, RHS;

public:
  BinaryExprAST(char Op, std::unique_ptr<ExprAST> LHS,
                std::unique_ptr<ExprAST> RHS)
      : Op(Op), LHS(std::move(LHS)), RHS(std::move(RHS)) {}

  Value *codegen(Value *first_arg, Value *scope_str, Value *previous_scope) override;
};


class BinaryTensorScalarExprAST : public ExprAST {
  char Op;
  std::unique_ptr<ExprAST> LHS, RHS;

public:
  BinaryTensorScalarExprAST(char Op, std::unique_ptr<ExprAST> LHS,
                std::unique_ptr<ExprAST> RHS)
      : Op(Op), LHS(std::move(LHS)), RHS(std::move(RHS)) {}

  Value *codegen(Value *first_arg, Value *scope_str, Value *previous_scope) override;
};


class BinaryTensorTensorExprAST : public ExprAST {
  char Op;
  std::unique_ptr<ExprAST> LHS, RHS;

public:
  BinaryTensorTensorExprAST(char Op, std::unique_ptr<ExprAST> LHS,
                std::unique_ptr<ExprAST> RHS)
      : Op(Op), LHS(std::move(LHS)), RHS(std::move(RHS)) {}

  Value *codegen(Value *first_arg, Value *scope_str, Value *previous_scope) override;
};


class BinaryPinnedScalarExprAST : public ExprAST {
  char Op;
  std::unique_ptr<ExprAST> LHS, RHS;

public:
  BinaryPinnedScalarExprAST(char Op, std::unique_ptr<ExprAST> LHS,
                std::unique_ptr<ExprAST> RHS)
      : Op(Op), LHS(std::move(LHS)), RHS(std::move(RHS)) {}

  Value *codegen(Value *first_arg, Value *scope_str, Value *previous_scope) override;
};

class BinaryTensorPinnedExprAST : public ExprAST {
  char Op;
  std::unique_ptr<ExprAST> LHS, RHS;

public:
  BinaryTensorPinnedExprAST(char Op, std::unique_ptr<ExprAST> LHS,
                std::unique_ptr<ExprAST> RHS)
      : Op(Op), LHS(std::move(LHS)), RHS(std::move(RHS)) {}

  Value *codegen(Value *first_arg, Value *scope_str, Value *previous_scope) override;
};


class BinaryObjExprAST : public ExprAST {
  char Op;
  std::unique_ptr<ExprAST> LHS, RHS;

public:
  BinaryObjExprAST(char Op, std::unique_ptr<ExprAST> LHS,
                std::unique_ptr<ExprAST> RHS)
      : Op(Op), LHS(std::move(LHS)), RHS(std::move(RHS)) {}

  Value *codegen(Value *first_arg, Value *scope_str, Value *previous_scope) override;
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

  Value *codegen(Value *first_arg, Value *scope_str, Value *previous_scope) override;
};

class ReturnExprAST : public ExprAST {

  public:
    std::vector<std::unique_ptr<ExprAST>> Vars;
    std::vector<bool> IsAs;
    std::vector<std::unique_ptr<ExprAST>> Destiny;
    
    ReturnExprAST(std::vector<std::unique_ptr<ExprAST>> Vars, std::vector<bool> IsAs,
                  std::vector<std::unique_ptr<ExprAST>> Destiny)
        : Vars(std::move(Vars)), IsAs(std::move(IsAs)), Destiny(std::move(Destiny)) {}

  Value *codegen(Value *first_arg, Value *scope_str, Value *previous_scope) override;
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

  Value *codegen(Value *first_arg, Value *scope_str, Value *previous_scope) override;
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

  Value *codegen(Value *first_arg, Value *scope_str, Value *previous_scope) override;
};

/// WhileExprAST - Expression class for while.
class WhileExprAST : public ExprAST {
	std::unique_ptr<ExprAST> Cond;
  std::vector<std::unique_ptr<ExprAST>> Body;

  public:
    WhileExprAST(std::unique_ptr<ExprAST> Cond, std::vector<std::unique_ptr<ExprAST>> Body)
      : Cond(std::move(Cond)), Body(std::move(Body)) {}

	Value* codegen(Value *first_arg, Value *scope_str, Value *previous_scope) override;
};


/// AsyncExprAST - Expression class for async.
class AsyncExprAST : public ExprAST {
	std::vector<std::unique_ptr<ExprAST>> Body;

  public:
    AsyncExprAST(std::vector<std::unique_ptr<ExprAST>> Body)
      : Body(std::move(Body)) {}

	Value* codegen(Value *first_arg, Value *scope_str, Value *previous_scope) override;
};


/// FinishExprAST - Expression class for finish/async.
class FinishExprAST : public ExprAST {
  std::vector<std::unique_ptr<ExprAST>> Bodies;
  std::vector<bool> IsAsync;

  public:
    FinishExprAST(std::vector<std::unique_ptr<ExprAST>> Bodies,
                  std::vector<bool> IsAsync)
            : Bodies(std::move(Bodies)), IsAsync(std::move(IsAsync)) {}


	Value* codegen(Value *first_arg, Value *scope_str, Value *previous_scope) override;
};


/// LockExprAST
class LockExprAST : public ExprAST {
  std::vector<std::unique_ptr<ExprAST>> Bodies;
  std::string Name;

  public:
    LockExprAST(std::vector<std::unique_ptr<ExprAST>> Bodies,
                std::string Name)
            : Bodies(std::move(Bodies)), Name(Name) {}


	Value* codegen(Value *first_arg, Value *scope_str, Value *previous_scope) override;
};

class UnlockExprAST : public ExprAST {
  std::string Name;

  public:
    UnlockExprAST(std::string Name) : Name(Name) {}


	Value* codegen(Value *first_arg, Value *scope_str, Value *previous_scope) override;
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
                bool IsOperator = false, unsigned Prec = 0)
        : Name(Name), Class(Class), Method(Method), Args(std::move(Args)), Types(std::move(Types)),
          IsOperator(IsOperator), Precedence(Prec) {}

  Function *codegen();
  const std::string &getName() const { return Name; }
  const std::string &getClass() const { return Class; }
  const std::string &getMethod() const { return Method; }

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


  if (BinopPrecedence.find(CurTok) == BinopPrecedence.end()) // if not found
    return -1;

  // Make sure it's a declared binop.
  int TokPrec = BinopPrecedence[CurTok];
  if (TokPrec <= 0)
    return -1;
  return TokPrec;
}



/// LogError* - These are little helper functions for error handling.
std::unique_ptr<ExprAST> LogErrorS(std::string Str) {
  ShallCodegen = false;
  //fprintf(stderr, "\033[31m Error: \033[0m%s\n", Str);
  if (Str!=" ")
    std::cout << "\nLinha: " << LineCounter << "\n   \033[31m Error: \033[0m " << Str << "\n\n";
  
  
  return nullptr;
}

std::unique_ptr<ExprAST> LogError(std::string Str) {
  //fprintf(stderr, "\033[31m Error: \033[0m%s\n", Str);
  LogErrorS(Str);

  while(CurTok!=tok_space && CurTok!=',' && CurTok!=')' && !in_char(CurTok, terminal_tokens))
    getNextToken();
  
  return nullptr;
}


std::unique_ptr<ExprAST> LogError_toNextToken(std::string Str) {
  //fprintf(stderr, "\033[31m Error: \033[0m%s\n", Str);
  LogErrorS(Str);

  getNextToken();
  
  return nullptr;
}


std::unique_ptr<ExprAST> LogErrorBreakLine(std::string Str) {
  //fprintf(stderr, "\033[31m Error: \033[0m%s\n", Str);
  LogErrorS(Str);

  while(CurTok!=tok_space && !in_char(CurTok, terminal_tokens))
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
  //fprintf(stderr, "\033[31mError: \033[0m%s\n", buf);
  std::cout << "\nLinha: " << LineCounter << "\n   \033[31m Error: \033[0mUnexpected token " << ReverseToken(CurTok) << ". Expected an expression.\n\n";
  
  while(CurTok!=tok_space && !in_char(CurTok, terminal_tokens))
    getNextToken();

  return nullptr;
}


std::unique_ptr<PrototypeAST> LogErrorP(const char *Str) {
  LogError(Str);
  while(CurTok!=tok_space && !in_char(CurTok, terminal_tokens))
    getNextToken();
  return nullptr;
}


std::unique_ptr<PrototypeAST> LogErrorP_to_comma(const char *Str) {
  LogError(Str);
  while(CurTok!=tok_space && CurTok!=',' && CurTok!=')' && !in_char(CurTok, terminal_tokens))
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

static std::unique_ptr<ExprAST> ParseExpression(std::string class_name="");
static std::unique_ptr<ExprAST> ParsePrimary(std::string class_name);

/// numberexpr ::= number
static std::unique_ptr<ExprAST> ParseNumberExpr() {
  auto Result = std::make_unique<NumberExprAST>(NumVal);
  getNextToken(); // consume the number
  return std::move(Result);
}

static std::unique_ptr<ExprAST> ParseStringExpr() {
  auto Result = std::make_unique<StringExprAST>(IdentifierStr);
  getNextToken(); // consume the "
  return std::move(Result);
}



/// parenexpr ::= '(' expression ')'
static std::unique_ptr<ExprAST> ParseParenExpr(std::string class_name="") {
  getNextToken(); // eat (.
  auto V = ParseExpression(class_name);
  if (!V)
    return nullptr;

  if (CurTok != ')')
    return LogError("Expected ')' on parenthesis expression.");
  std::cout << "\n\n\nV type: " << V->GetType() << "\n\n\n\n";

  getNextToken(); // eat ).
  return V;
}

//global
std::vector<std::string> tensorVars;
std::vector<std::string> pinnedTensorVars;
std::vector<std::string> floatVars;
std::vector<std::string> strVars;
std::vector<std::string> objectVars;
std::map<std::string, std::string> functionVars;
std::map<std::string, std::string> floatFunctions;
std::map<std::string, std::string> stringMethods;
std::vector<std::string> str_vecVars;
std::vector<std::string> float_vecVars;
std::map<std::string, pthread_mutex_t *> lockVars;



//global
static std::vector<std::string> Classes;
static std::map<std::string, std::string> Object_toClass;
static std::map<std::string, std::string> Object_toClassVec;


static std::unique_ptr<ExprAST> ParseObjectInstantiationExpr(std::string _class, std::string class_name) {
  getNextToken();
  //std::cout << "Object name: " << IdentifierStr << " and Class: " << Classes[i]<< "\n";
  bool is_vec=false;
  bool is_self=false;
  bool is_attr=false;
  std::string pre_dot="";
  std::unique_ptr<ExprAST> VecInitSize = nullptr;
      
  //std::cout << "\n\n\n\nCUR TOK IS: " << ReverseToken(CurTok) << "\n\n\n\n\n\n";
  if (CurTok==tok_vec)
  {
    getNextToken();
    is_vec=true;
        
    if(CurTok=='[')
    {
      getNextToken();
      VecInitSize = ParsePrimary(class_name);
      if (CurTok!=']')
        LogError("Expected ] at object vec");
      getNextToken();
    }
    Object_toClassVec[IdentifierStr] = _class; //todo: this doesn't deal with self. expr
  }


  if (CurTok==tok_self)
  {
    getNextToken();
    is_self=true;
  }


  std::vector<std::pair<std::string, std::unique_ptr<ExprAST>>> VarNames;
  while (true) {
    std::string Name = IdentifierStr;
    objectVars.push_back(Name);
    if (!is_vec)
      Object_toClass[IdentifierStr] = _class;
    getNextToken(); // eat identifier.

        
    std::unique_ptr<ExprAST> Init = nullptr;  
    VarNames.push_back(std::make_pair(Name, std::move(Init)));

    // End of var list, exit loop.
    if (CurTok != ',')
      break;
    getNextToken(); // eat the ','.

    if (CurTok != tok_identifier)
      return LogError("Expected object identifier names.");
  }

  auto aux = std::make_unique<ObjectExprAST>(std::move(VarNames), "object", std::move(VecInitSize));
  aux->SetSelf(is_self);
  aux->SetIsAttribute(is_attr);
  aux->SetPreDot(pre_dot);
  aux->SetIsVec(is_vec);

  if (CurTok==tok_space)
    getNextToken();

  return aux;
}


static std::vector<std::unique_ptr<ExprAST>> ParseIdx(std::string class_name="") {

  std::vector<std::unique_ptr<ExprAST>> Idx;
    
  Idx.push_back(ParseExpression(class_name));
  while(CurTok==',')
  {
    getNextToken(); // eat ,
    Idx.push_back(ParseExpression(class_name));
  }
  Idx.push_back(std::make_unique<NumberExprAST>(TERMINATE_VARARG));

  return Idx;
}



/// identifierexpr
///   ::= identifier
///   ::= identifier '(' expression* ')'
static std::unique_ptr<ExprAST> ParseIdentifierExpr(std::string class_name="") {
  
  for(int i=0; i<Classes.size(); i++)
    if(IdentifierStr==Classes[i])  // Object object
      return ParseObjectInstantiationExpr(Classes[i], class_name);

  std::vector<std::tuple<std::string, int, std::vector<std::unique_ptr<ExprAST>>>> Names;
  std::string IdName, type;
  IdName = IdentifierStr;
  //std::cout << "Identifier " << IdName <<  "\n";
  getNextToken(); // eat identifier.
  
  Names.push_back(std::make_tuple(IdName, type_var, std::vector<std::unique_ptr<ExprAST>>{}));
  if (CurTok != '(' && CurTok != '[') // Simple variable ref.
  {
    if (in_str(IdName, pinnedTensorVars))
      type = "pinned_tensor";
    if (in_str(IdName, tensorVars))
      type = "tensor";
    if (in_str(IdName, floatVars))
      type = "float";
    if (in_str(IdName, strVars))
      type = "str";
    if (in_str(IdName, objectVars))
      type = "object";


    auto name_solver_expr = std::make_unique<NameSolverAST>(std::move(Names));
    auto aux = std::make_unique<VariableExprAST>(std::move(name_solver_expr), type);
    
    
    if (CurTok==tok_space)
      getNextToken();

    return aux;
  }



  if (CurTok=='[')
  {
    getNextToken(); // eat [
    
    std::vector<std::unique_ptr<ExprAST>> Idx;
    Idx = ParseIdx(class_name);
    
    if (in_str(IdName, str_vecVars))
      type = "str_vec";
    if (in_str(IdName, float_vecVars))
      type = "float_vec";
    if (in_str(IdName, tensorVars))
      type = "tensor";

    auto name_solver_expr = std::make_unique<NameSolverAST>(std::move(Names));
    auto aux = std::make_unique<VecIdxExprAST>(std::move(name_solver_expr), std::move(Idx), type);
    aux->SetIsVec(true);
    
    getNextToken(); // eat ]
    
    return std::move(aux);
  }
  

  // Call.
  getNextToken(); // eat (
  
  std::vector<std::unique_ptr<ExprAST>> Args;
  if (CurTok != ')') {
    while (true) {
      
      if (auto Arg = ParseExpression(class_name))
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
  bool return_tensor = false;
  std::string callee_override = "none";
  if (functionVars.find(IdName) != functionVars.end()) // if found
  {
    is_var_forward = true;
    return_tensor = true;
    callee_override = functionVars[IdName];
  }
  if (floatFunctions.find(IdName) != floatFunctions.end()) // if found
  {
    is_var_forward = true;
    callee_override = floatFunctions[IdName];
  }
  if (IdName=="to_float")
  {
    callee_override = "ToFloat";
    is_var_forward = true;
  }
  
  auto aux = std::make_unique<CallExprAST>(IdName, std::move(Args), "None", "None", is_var_forward, callee_override);

  
  
  if (in_str(IdName, return_tensor_fn) || return_tensor)
    aux->SetType("tensor");
  if (in_str(IdName, return_pinned_methods))
    aux->SetType("pinned_tensor");

  return aux;
}





/// ifexpr ::= 'if' expression 'then' expression 'else' expression
static std::unique_ptr<ExprAST> ParseIfExpr(std::string class_name="") {
  
  int cur_level_tabs = SeenTabs;

  getNextToken(); // eat the if.

  
  //std::cout << "If tabs level: " << cur_level_tabs <<  "\n";
  

  // condition.
  auto Cond = ParseExpression(class_name);
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
    
    auto body = ParseExpression(class_name);
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

      if (SeenTabs <= cur_level_tabs && CurTok != tok_space)
        break;

      while (CurTok == tok_space)
        getNextToken();

      if (SeenTabs <= cur_level_tabs)
        break;
      
      auto body = ParseExpression(class_name);
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




static std::vector<std::unique_ptr<ExprAST>> ParseIdentedBodies(int cur_level_tabs, std::string class_name="")
{
  std::vector<std::unique_ptr<ExprAST>> Body, NullBody;
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


    auto body = ParseExpression(class_name);
    if (!body)
      return std::move(NullBody);
    Body.push_back(std::move(body));
    //getNextToken();
  }

  if (CurTok==tok_space)
    getNextToken();

  return std::move(Body);
}


/// forexpr ::= 'for' identifier '=' expr ',' expr (',' expr)? 'in' expression
static std::unique_ptr<ExprAST> ParseForExpr(std::string class_name="") {

  int cur_level_tabs = SeenTabs;

  //std::cout << "\nSeen tabs on for: " << SeenTabs << "\n\n";

  getNextToken(); // eat the for.


  if (CurTok != tok_identifier)
    return LogError("Expected for's control variable identifier.");

  std::string IdName = IdentifierStr;
  getNextToken(); // eat identifier.

  floatVars.push_back(IdName);

  if (CurTok != '=')
    return LogError("Expected for's control variable initial value.");
  getNextToken(); // eat '='.

  auto Start = ParseExpression(class_name);
  if (!Start)
    return nullptr;
  if (CurTok != ',')
    return LogError("Expected ',' after for's control variable initial value.");
  getNextToken();



  auto End = ParseExpression(class_name);
  if (!End)
    return nullptr;

  


  std::unique_ptr<ExprAST> Step = std::make_unique<NumberExprAST>(1.0);
  if (CurTok == ',') { // The step value is optional.
    getNextToken();
    auto aux = ParseExpression(class_name);
    if (aux)
      Step = std::move(aux);
  }
  
  std::vector<std::unique_ptr<ExprAST>> Body;

  Body = ParseIdentedBodies(cur_level_tabs, class_name);

  return std::make_unique<ForExprAST>(IdName, std::move(Start), std::move(End),
                                       std::move(Step), std::move(Body));
}



/// whileexpr ::= 'while' identifier '=' expr ',' expr (',' expr)? 'in' expression
static std::unique_ptr<ExprAST> ParseWhileExpr(std::string class_name="") {

  int cur_level_tabs = SeenTabs;

  getNextToken(); // eat the while.


  //if (CurTok != tok_identifier)
  //  return LogError("Identificador da variável de controle esperado depois do while.");


  auto Cond = ParseExpression(class_name);
  if (!Cond)
    return nullptr;
  
  std::vector<std::unique_ptr<ExprAST>> Body = ParseIdentedBodies(cur_level_tabs, class_name);

  return std::make_unique<WhileExprAST>(std::move(Cond), std::move(Body));
}



static std::unique_ptr<ExprAST> ParseAsyncExpr(std::string class_name="") {

  int cur_level_tabs = SeenTabs;

  getNextToken(); // eat the async.

  
  std::vector<std::unique_ptr<ExprAST>> Bodies;
  std::cout << "\nasync tabs level: " << cur_level_tabs <<  "\n";
  //std::cout << "Pre expression token: " << ReverseToken(CurTok) << "\n";

  if (CurTok != tok_space)
    Bodies.push_back(std::move(ParseExpression(class_name)));
  else
    Bodies = ParseIdentedBodies(cur_level_tabs, class_name);
  
  
  
  //std::cout << "Post async: " << ReverseToken(CurTok) << "\n";

  return std::make_unique<AsyncExprAST>(std::move(Bodies));
}



static std::unique_ptr<ExprAST> ParseFinishExpr(std::string class_name="") {

  int cur_level_tabs = SeenTabs;
  //std::cout << "Finish tabs level: " << cur_level_tabs <<  "\n";

  getNextToken(); // eat the finish.


  std::vector<std::unique_ptr<ExprAST>> Bodies;
  std::vector<bool> IsAsync;
  

  if (CurTok!=tok_space)
    LogError("Finish requer quebra de linha.");
  getNextToken(); 


  while(!in_char(CurTok, terminal_tokens))
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
    

    /*
    if (CurTok == tok_async)
    {
      Bodies.push_back(std::move(ParseAsyncExpr(class_name)));
      IsAsync.push_back(true);
    }
    else
    {
      Bodies.push_back(std::move(ParseExpression(class_name)));
      IsAsync.push_back(false);
    }
    */
    Bodies.push_back(std::move(ParseExpression(class_name)));
    IsAsync.push_back(false);
  }


  return std::make_unique<FinishExprAST>(std::move(Bodies),
                                         std::move(IsAsync));
}


/// varexpr ::= 'var' identifier ('=' expression)?
//                    (',' identifier ('=' expression)?)* 'in' expression
static std::unique_ptr<ExprAST> ParseVarExpr(std::string class_name="") {
  getNextToken(); // eat the var.
  std::cout << "Parsing float expr\n";

  // mem2reg is alloca-driven: it looks for allocas and if it can handle them, it promotes them. It DOES NOT APPLY TO GLOBAL variables or heap allocations.
  // mem2reg only promotes allocas whose uses are direct loads and stores. If the address of the stack object is passed to a function,
  //or if any funny pointer arithmetic is involved, the alloca will not be promoted.

  std::vector<std::pair<std::string, std::unique_ptr<ExprAST>>> VarNames;

  // At least one variable name is required.
  if (CurTok != tok_identifier)
    return LogError("Expected an identifier after float.");

  while (true) {
    std::string Name = IdentifierStr;
    floatVars.push_back(IdentifierStr);
    getNextToken(); // eat identifier.

    // Read the optional initializer.
    std::unique_ptr<ExprAST> Init = nullptr;
    if (CurTok == '=')
    {
      getNextToken(); // eat the '='.

      Init = ParseExpression(class_name);
      if (!Init)
        return nullptr;
    }

    VarNames.push_back(std::make_pair(Name, std::move(Init)));

    // End of var list, exit loop.
    if (CurTok != ',')
      break;
    getNextToken(); // eat the ','.

    if (CurTok != tok_identifier)
      return LogError("Expected one or more identifiers after float.");
  }

  if (CurTok==tok_space)
    getNextToken();


  return std::make_unique<VarExprAST>(std::move(VarNames), "var");
}




static std::unique_ptr<ExprAST> ParseStrExpr() {
  getNextToken(); // eat str
  

  std::vector<std::pair<std::string, std::unique_ptr<ExprAST>>> VarNames;

  // At least one variable name is required.
  if (CurTok != tok_identifier)
    return LogError("Esperado identificador após var.");

  
  std::string pre_dot="";
  bool is_self = false;
  bool is_attr = false;
  if (CurTok == tok_self)
  {
    is_self=true; //TODO: set self per VarName instead.
    getNextToken();
  }
  if (CurTok == tok_class_attr)
  {
    is_attr=true;
    pre_dot = IdentifierStr;
    getNextToken();
  }

  while (true) {
    std::string Name = IdentifierStr;
    strVars.push_back(Name);
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

  auto aux = std::make_unique<StrExprAST>(std::move(VarNames));

  aux->SetSelf(is_self);
  aux->SetIsAttribute(is_attr);
  aux->SetPreDot(pre_dot);
  

  if (CurTok==tok_space)
    getNextToken();


  return aux;
}


static std::unique_ptr<ExprAST> ParseStrVecExpr() {
  int vec_type = CurTok;
  std::string vec_type_str;

  if (vec_type==tok_str_vec)
    vec_type_str = "str";
  if (vec_type==tok_float_vec)
    vec_type_str = "float";
    
  getNextToken(); // eat str_vec

  
  
  
  std::vector<std::pair<std::string, std::unique_ptr<ExprAST>>> VarNames;

  // At least one variable name is required.
  if (CurTok != tok_identifier)
    return LogError("Expected identifier after vector var.");

  

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

    if (vec_type==tok_str_vec)
      str_vecVars.push_back(Name);
    if (vec_type==tok_float_vec)
      float_vecVars.push_back(Name);
    

    // End of var list, exit loop.
    if (CurTok != ',')
      break;
    getNextToken(); // eat the ','.

    if (CurTok != tok_identifier)
      return LogError("Esperado um ou mais identificadores após var.");
  }

  if (CurTok==tok_space)
    getNextToken();

  return std::make_unique<StrVecExprAST>(std::move(VarNames), vec_type_str);
}




struct BackwardNode {
  int B, C, OC;
  int x_size, w_size;
  float *inp, *w, *out;
  float *b=nullptr;
  int b_size=0;

  std::string op, tensor_name, param_name;

  void NewNode(int B, int C, int OC, int x_size, int w_size, float *inp, float *w,
                  float *out, std::string op, std::string tensor_name, std::string param_name) {
    this->B = B;
    this->C = C;
    this->OC = OC;
    this->x_size = x_size;
    this->w_size = w_size;
    this->inp = inp;
    this->w = w;
    this->out = out;
    this->op = op;
    this->tensor_name = tensor_name;
    this->param_name = param_name;
  }

  void SetBias(float *b, int b_size)
  {
    this->b=b;
    this->b_size=b_size;
  }
};




struct Tensor {
  float *tensor_ptr;
  float *cpu_tensor_ptr;
  std::vector<float> dims;
  float dims_prod;
  float *b=nullptr;
  float *dy=nullptr;
  int b_size=0;

  bool leaf, weight;
  std::string view_of = "";
  std::string name;
  std::string scopeless_name;
  int op;

  Tensor *R_Node, *L_Node;
  bool visited;

  void NewTensor(float *new_tensor_ptr, std::vector<float> new_dims, float new_dims_prod,
                 bool new_is_leaf, std::string new_name){
    tensor_ptr = new_tensor_ptr;
    dims = new_dims;
    dims_prod = new_dims_prod;
    leaf = new_is_leaf;
    name = new_name;
    cpu_tensor_ptr = nullptr;
    op=leaf;
    L_Node=nullptr;
    R_Node=nullptr;
    dy=nullptr;
    visited=false;
    weight=false;
    op=tensor_leaf;
  }

  void NewPinned(float *new_tensor_ptr, float *new_cpu_tensor_ptr,
                 std::vector<float> new_dims, float new_dims_prod,
                 bool new_is_leaf, std::string new_name){
    tensor_ptr = new_tensor_ptr;
    cpu_tensor_ptr = new_cpu_tensor_ptr;
    dims = new_dims;
    dims_prod = new_dims_prod;
    leaf = new_is_leaf;
    name = new_name;
    weight=false;
  }

  void AttrTensor(float *new_tensor_ptr, std::vector<float> new_dims, float new_dims_prod){
    tensor_ptr = new_tensor_ptr;
    dims = new_dims;
    dims_prod = new_dims_prod;
  }

  
  void AttrNodes(Tensor *new_L_Tensor, Tensor *new_R_Tensor, int op_type)
  {
    L_Node = new_L_Tensor;
    R_Node = new_R_Tensor;
    op = op_type;
    leaf=false;
    visited=false;
    dy=nullptr;
    weight=false;
  }

  void AttrLNode(Tensor *new_L_Tensor, int op_type)
  {
    L_Node = new_L_Tensor;
    R_Node=nullptr;
    op = op_type;
    leaf=false;
    visited=false;
    dy=nullptr;
    weight=false;
  }

  void AttributionBackwardNode(std::string _name, Tensor *new_R_Tensor)
  {
    name = _name;
    R_Node = new_R_Tensor;
    op = attribution;
    leaf=false;
    visited=false;
    
    L_Node=nullptr;
    dy=nullptr;
    weight=false;
  }
  
  void SetBias(float *b, int b_size)
  {
    this->b=b;
    this->b_size=b_size;
    leaf=true;
  }
};

Tensor *createTensor(float* tensor_ptr, const std::vector<float>& dims, float kDataLen,
                     bool is_leaf, std::string name) {
    Tensor *new_tensor = new Tensor();
    new_tensor->NewTensor(tensor_ptr, dims, kDataLen, is_leaf, name);
    return new_tensor;
}
Tensor *createPinned(float* tensor_ptr, float *tensor_cpu, const std::vector<float>& dims, float kDataLen,
                     std::string name) {
    Tensor *new_tensor = new Tensor();
    new_tensor->NewPinned(tensor_ptr, tensor_cpu, dims, kDataLen, true, name);
    return new_tensor;
}
Tensor *createBackward(std::string name, Tensor *tensor) {
    Tensor *new_tensor = new Tensor();
    new_tensor->AttributionBackwardNode(name, tensor);
    return new_tensor;
}


bool in_tensor_ptr_vec(Tensor *value, const std::vector<Tensor *>& list) {
    return std::find(list.begin(), list.end(), value) != list.end();
}



//global
using backward_tuple = std::tuple<int, int, int, int, int, float *, float *, float *, std::string, std::string, std::string>;
std::vector<BackwardNode> todo_backwards;
std::vector<Tensor *> todo_backward_tensors;

// Tensors
static std::map<std::string, Tensor *> NamedTensorsT;
static std::map<std::string, float *> NamedPinnedTensors;
static std::map<std::string, std::vector<float>> NamedDims;
static std::vector<Tensor> TensorsToDelete;


unsigned char* current_data_attr;
std::vector<float> current_data_attr_dims;


static std::map<std::string, std::string> objectVecs;
static std::map<std::string, int> objectVecsLastId;



extern "C" float eval()
{
  std::cout << "SETTING NN MODE TO EVAL" << "\n";
  nn_mode = eval_mode;
  return 0;
}

extern "C" float train()
{
  std::cout << "SETTING NN MODE TO TRAIN" << "\n";
  nn_mode = training_mode;
  return 0;
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
int DimsProd(std::vector<float> dims)
{
  if (dims.size()==1)
    return (int) dims[0];

  float aux=1;
  for (int i = 0; i < dims.size(); i++)
    aux = aux*dims[i];
  return (int)aux;
}

std::vector<float> BatchLessDims(std::vector<float> dims)
{
  // Removes first dim (batch dim).
  if (dims.size()<=1)
    LogError("Cannot remove the batch dimension of a unidimensional tensor.");

  std::vector<float> new_dims;

  for (int i=0; i<dims.size()-1;i++)
    new_dims.push_back(dims[i+1]);

  return new_dims;
}

std::vector<float> RemoveLastDim(std::vector<float> dims)
{
  // Removes first dim (batch dim).
  if (dims.size()<=1)
    LogError("Cannot remove the batch dimension of a unidimensional tensor.");

  std::vector<float> new_dims;

  for (int i=0; i<dims.size()-1;i++)
    new_dims.push_back(dims[i]);

  return new_dims;
}

std::vector<float> RemoveFirstDim(std::vector<float> dims)
{
  // Removes first dim (batch dim).

  std::vector<float> new_dims;

  for (int i=0; i<dims.size()-1;i++)
    new_dims.push_back(dims[i+1]);

  return new_dims;
}





extern "C" float *load_img(char *img_name)
{
  int width, height, channels;
  
  unsigned char* image_data = stbi_load(img_name, &width, &height, &channels, 0);

  if (image_data) {
    
    current_data_attr_dims.clear();
    current_data_attr_dims.push_back((float)width);
    current_data_attr_dims.push_back((float)height);
    current_data_attr_dims.push_back((float)channels);

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
    std::string _error = "Failed to open image: " + img_n + ".";
    LogErrorS(_error);
  }

  return nullptr;
}


extern "C" float * gload_img(Tensor tensor, char *img_name, float batch_idx)
{
  //std::cout << "LOADING IMAGE FOR: " << tensor.name <<  "\nImage: " << img_name << "\n";
  

  int width, height, channels;
  
  unsigned char* image_data = stbi_load(img_name, &width, &height, &channels, 0);

  if (image_data) {
    
    std::vector<float> dims = tensor.dims;

    //std::cout << "GLOAD IMG, dims of " << tensor.name << "\n";
    //PrintDims(dims);

    std::vector<float> batchless_dims = BatchLessDims(dims);
    int batchless_dims_prod = DimsProd(batchless_dims);
    
    current_data_attr_dims.clear();
    current_data_attr_dims.push_back((float)width);
    current_data_attr_dims.push_back((float)height);
    current_data_attr_dims.push_back((float)channels);


    if (batchless_dims_prod < width*height*channels)
    {
      std::string t_n = tensor.name;
      std::string _error = "The image dimensions are incompatible with the tensor " + t_n + " dimensions.";
      LogErrorS(_error);


      std::cout << "\nTENSOR BATCHLESS DIMS:" << "\n";
      PrintDims(batchless_dims);

      std::cout << "\nImage required dims: [" << width << ", " << height << ", " << channels << "]\n\n";

      return nullptr;
    }
    if (batch_idx > dims[0])
    {
      std::string _error = "Tried to load a pinned tensor on batch index " + std::to_string(batch_idx) + ", whereas this tensor batch size is " + std::to_string(dims[0]) + ".";
      LogErrorS(_error);
    }



    //float *image_data_float = new float[width * height * channels];
    float *image_data_float = tensor.cpu_tensor_ptr;
    int batch_offset = (int) batchless_dims_prod*batch_idx;
    //std::cout << "batch idx: " << batch_idx << ", batch offset: " << batch_offset << "\n";
  
    // Loop through each pixel and convert to float between 0.0 and 1.0
    for (int y = 0; y < height; ++y) {
      for (int x = 0; x < width; ++x) {
        for (int c = 0; c < channels; ++c) {
          // Assuming unsigned char has 8 bits, scale by 1/255.0 to get a float value between 0.0 and 1.0
          image_data_float[batch_offset + (y * width + x) * channels + c] = (float)image_data[(y * width + x) * channels + c] / 255.0f;
        }
      }
    }
    stbi_image_free(image_data);

    return image_data_float;
    
  } else {
    std::string img_n = img_name;
    std::string _error = "Failed to open image: " + img_n + ".\n\n";
    LogErrorS(_error);
  }

  return nullptr;
}







extern "C" float * wload_img(Tensor *tensor, char *img_name, float worker_idx, float batch_idx)
{
  //std::cout << "LOADING IMAGE FOR: " << tensor->name <<  "\n";
  //std::cout << "Image: " << img_name <<  "\n";


  int width, height, channels;

  //std::cout << "GLOAD IMG, dims of " << img_name << "\n";
  
  unsigned char* image_data = stbi_load(img_name, &width, &height, &channels, 0);

  if (image_data) {
    
    std::vector<float> dims = tensor->dims;

    
    

    std::vector<float> workerless_dims = BatchLessDims(dims);
    int workerless_dims_prod = DimsProd(workerless_dims);

    std::vector<float> batchless_dims = BatchLessDims(workerless_dims);
    int batchless_dims_prod = DimsProd(batchless_dims);
    
    current_data_attr_dims.clear();
    current_data_attr_dims.push_back((float)width);
    current_data_attr_dims.push_back((float)height);
    current_data_attr_dims.push_back((float)channels);


    if (batchless_dims_prod < width*height*channels)
    {
      std::string t_n = tensor->name;
      std::string _error = "The image dimensions are incompatible with the tensor \033[95m" + t_n + "\033[0m dimensions.";
      LogErrorS(_error);


      std::cout << "\nTENSOR BATCHLESS DIMS:" << "\n";
      PrintDims(batchless_dims);

      std::cout << "\nImage required dims: [" << width << ", " << height << ", " << channels << "]\n\n";
      
      return nullptr;
    }
    if (batch_idx > dims[1])
    {
      std::string _error = "Tried to load a pinned tensor on batch index " + std::to_string(batch_idx) + ", whereas this tensor batch size is " + std::to_string(dims[1]) + ".";
      LogErrorS(_error);
    }



    float *image_data_float = tensor->cpu_tensor_ptr;
    int idx_offset = (int) (batchless_dims_prod*batch_idx + workerless_dims_prod*worker_idx);

    //std::cout << "worker idx: " << worker_idx << ", batch idx: " << batch_idx << ", batch offset: " << idx_offset << "\n";
  
    // Loop through each pixel and convert to float between 0.0 and 1.0
    for (int y = 0; y < height; ++y) {
      for (int x = 0; x < width; ++x) {
        for (int c = 0; c < channels; ++c) {
          // Assuming unsigned char has 8 bits, scale by 1/255.0 to get a float value between 0.0 and 1.0
          image_data_float[idx_offset + (y * width + x) * channels + c] = (float)image_data[(y * width + x) * channels + c] / 255.0f;
        }
      }
    }
    stbi_image_free(image_data);

    //std::cout << "returning float image" << "\n";
    return image_data_float;
    
  } else {
    std::string img_n = img_name;
    std::string _error = "Failed to open image: " + img_n + ".\n\n";
    LogErrorS(_error);
  }

  return nullptr;
}


extern "C" float cpu(Tensor *tensor)
{

  float *tensor_ptr, *tensor_cpu;
  tensor_ptr = tensor->tensor_ptr;
  tensor_cpu = tensor->cpu_tensor_ptr;

  if (tensor_ptr==nullptr)
    LogErrorS("Cannot load tensor to cpu from an null tensor.");

  if (tensor_cpu!=nullptr)
    cudaCheck(cudaFree(tensor_cpu));

  float dims_prod = tensor->dims_prod;

  cudaMallocHost(&tensor_cpu, dims_prod*sizeof(float));
  cudaMemcpy(tensor_cpu, tensor_ptr, dims_prod*sizeof(float), cudaMemcpyDeviceToHost);

  tensor->cpu_tensor_ptr = tensor_cpu;


  return 0;
}

extern "C" float cpu_idx(Tensor *tensor, float idx)
{

  float *tensor_cpu;
  tensor_cpu = tensor->cpu_tensor_ptr;


  if (tensor_cpu==nullptr)
    LogErrorS("Cannot idx a null cpu tensor.");

  float dims_prod = tensor->dims_prod;
  if (idx>dims_prod)
    LogErrorS("Idx higher than dims prod at cpu_idx().");

  

  return tensor_cpu[(int)idx];
}






extern "C" float * split_str_to_float(char *in_string, int gather_position)
{
  std::vector<std::string> splitted = split_str(in_string, '/');

  float * ret = new float[1];

  if(gather_position<0)
    gather_position = splitted.size()+gather_position;

  ret[0] = std::stof(splitted[gather_position]);

  return ret;
}





//
static std::unique_ptr<ExprAST> ParseSelfExpr(std::string class_name="") {

  std::string pre_dot = "";
  std::string type = "None";
  std::string object_class;
  bool is_class_attr=false;
  bool is_self=false;
  bool is_vec=false;
  std::vector<std::tuple<std::string, int, std::vector<std::unique_ptr<ExprAST>>>> Names;
  std::string IdName;



  if (CurTok!=tok_self)
  {
    is_class_attr = true;
    pre_dot="";
  } else 
  {
    is_self=true;
    getNextToken(); // eat self
    Names.push_back(std::make_tuple("_", type_self, std::vector<std::unique_ptr<ExprAST>>{}));
  }


  
  int i=0;
  while (CurTok==tok_identifier||CurTok==tok_class_attr||CurTok==tok_post_class_attr_attr||CurTok==tok_post_class_attr_identifier)
  {
    int _type = type_attr;
    
    if (in_str(IdentifierStr, objectVars))
    {
      for (int j=0; j<objectVars.size();j++)
        std::cout << "objectVars: " << objectVars[j] << "\n";
      std::cout << "\n\n" << IdentifierStr << " IS ON OBJECT VARS" <<  "\n\n";
      _type = type_object_name;
    }
      
    if (i==0&&CurTok==(tok_class_attr))
    {
      Names.push_back(std::make_tuple(IdentifierStr, _type, std::vector<std::unique_ptr<ExprAST>>{}));
      _type = type_attr;
    }
      
    if (i>0)
      is_class_attr = true;

    if (CurTok!=tok_identifier&&CurTok!=tok_post_class_attr_identifier)
    {
      object_class=IdentifierStr;
      pre_dot+=IdentifierStr;
    }

    is_vec=false;
    if (CurTok==tok_identifier||CurTok==tok_post_class_attr_identifier) // Need to handle vector
      IdName = IdentifierStr;

    std::cout << "\n\ntok pre vec: " << ReverseToken(CurTok) << "\n";

    getNextToken(); // eat attr/identifier

      
    if (CurTok=='[')
    {
      std::cout << "tokvec: " << ReverseToken(CurTok) << "\n";
      getNextToken(); // eat [
      std::vector<std::unique_ptr<ExprAST>> idx = ParseIdx(class_name);
      getNextToken(); // eat ]
      _type = type_vec;
      //int _type = (Object_toClassVec.count(IdName)>0) ? type_object_vec : type_vec;
      Names.push_back(std::make_tuple(IdName, _type, std::move(idx)));
      is_vec=true;
    } else
      Names.push_back(std::make_tuple(IdName, _type, std::vector<std::unique_ptr<ExprAST>>{}));

    std::cout << "tok: " << ReverseToken(CurTok) << "\n";
    i+=1;
  }
    

  
  std::cout << "Post tok: " << ReverseToken(CurTok) << "\n";
  
  // Turns string from object model of class type Model into Model
  if (Object_toClass.count(object_class)>0)
    object_class = Object_toClass[object_class]; 
  
  
  
  
  std::cout << "\n\nParseSelfExpr of " << IdentifierStr << " HAS CLASS: " << class_name << " and pre-dot: " << pre_dot << "\n\n\n";



  

  if (!is_vec&&CurTok!='(') // Simple variable ref.
  {
    std::cout << "Parsing a var" << "\n";
    
    if (in_str(IdName, pinnedTensorVars))
      type = "pinned_tensor";
    if (in_str(IdName, tensorVars))
      type = "tensor";
    if (in_str(IdName, objectVars))
      type = "object";
    if (functionVars.find(IdName) != functionVars.end())
      type = "tensor";
    if (stringMethods.find(IdName) != stringMethods.end())
      type = "str";
    if (in_str(IdName, floatVars))
      type = "float";
    if (in_str(IdName, float_vecVars))
      type = "float_vec";
    if (in_str(IdName, strVars))
      type = "str";
    if (in_str(IdName, str_vecVars))
      type = "str_vec";

    std::cout << "Var type: " << type << "\n";

    auto name_solver_expr = std::make_unique<NameSolverAST>(std::move(Names));
    auto aux = std::make_unique<VariableExprAST>(std::move(name_solver_expr), type);

    aux->SetSelf(is_self);
    aux->SetIsAttribute(is_class_attr);
    aux->SetPreDot(pre_dot);
    
    
    if (CurTok==tok_space)
      getNextToken();

    return aux;
  }



  if (is_vec)
  {
    std::unique_ptr<ExprAST> aux;
    std::vector<std::unique_ptr<ExprAST>> Idx;
    
    
    
    if (in_str(IdName, str_vecVars))
      type= "str_vec";
    if (in_str(IdName, float_vecVars))
      type = "float_vec";
    if (in_str(IdName, pinnedTensorVars))
      type = "pinned_tensor";
    if (in_str(IdName, tensorVars))
      type= "tensor";
    if (in_str(IdName, objectVars))
      type = "object_vec";

    if(type=="object_vec")
      Idx = std::vector<std::unique_ptr<ExprAST>>{};
    else
      Idx = std::move(std::get<2>(Names[Names.size()-1]));

    auto name_solver_expr = std::make_unique<NameSolverAST>(std::move(Names));
    aux = std::make_unique<VecIdxExprAST>(std::move(name_solver_expr), std::move(Idx), type);
    aux->SetIsVec(true);


    aux->SetSelf(is_self);
    aux->SetIsAttribute(is_class_attr);
    aux->SetPreDot(pre_dot);


    return std::move(aux);
  }




  // ParseCall.

  getNextToken(); // eat (
  std::vector<std::unique_ptr<ExprAST>> Args;
  if (CurTok != ')') {
    while (true) {
      if (auto Arg = ParseExpression(class_name))
      {
        std::cout << "Parsed arg " << Arg->GetName() << "\n";
        Args.push_back(std::move(Arg));
      }
        
      else
        return nullptr;

      if (CurTok == ')')
        break;

      if (CurTok != ',')
        return LogError("Esperado ')' ou ',' na lista de argumentos");
      getNextToken();
    }
  }

  // varargs
  if (in_str(IdName, vararg_methods))
    Args.push_back(std::make_unique<NumberExprAST>(TERMINATE_VARARG));
  

  // Eat the ')'.
  getNextToken();




  // Override function calls: e.g: conv1 -> Conv2d
  bool is_var_forward = false;
  bool return_tensor = false;
  bool return_string = false;
  std::string callee_override = "none";
  if (functionVars.count(IdName) > 0)
  {
    is_var_forward = true;
    return_tensor = true;
    callee_override = functionVars[IdName];

  } else if (floatFunctions.find(IdName) != floatFunctions.end())
  {
    is_var_forward = true;
    callee_override = floatFunctions[IdName];

  } else if (stringMethods.find(IdName) != stringMethods.end())
  {
    is_var_forward = true;
    return_string = true;
    callee_override = stringMethods[IdName];

  } else {
    if (is_self && !is_class_attr)
      IdName = class_name + IdName;
  }

  
  std::cout << "\nCalling method: " << IdName << " for pre-dot: " << pre_dot << "\n\n";

  //if (IdName == "len")



  auto aux = std::make_unique<CallExprAST>(IdName, std::move(Args), object_class, pre_dot, is_var_forward, callee_override);

  if (in_str(IdName, return_tensor_fn) || return_tensor)
  {
    aux->SetType("tensor");
  }
  if (return_string)
    aux->SetType("str");

  
  if (CurTok==tok_space)
    getNextToken();


  if (is_self)
    aux->SetSelf(true);    
  if (is_class_attr)
    aux->SetIsAttribute(true);
  aux->SetPreDot(pre_dot);
  
  return aux;
}


//
static std::unique_ptr<ExprAST> ParsePinnedTensorExpr() {
  
  getNextToken(); // eat pinned_tensor.
  
  if (CurTok != '[')
    return LogError("pinned tensor declaration expected [");
    getNextToken();

  std::vector<std::unique_ptr<ExprAST>> dims;
  std::vector<std::pair<std::string, std::unique_ptr<ExprAST>>> VarNames;
  std::string init = "zeros";
  //std::make_unique<NumberExprAST>(NumVal)
  
  while (true) {
    if (CurTok != tok_number && CurTok != tok_identifier && CurTok != tok_self)
      return LogError("Expected a number or var on the tensor dimension.");
    
    if (CurTok==tok_number)
    {
      if (std::fmod(NumVal, 1.0) != 0)
        LogWarning("A tensor's dimension should be int, not float.");
    
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
    return LogError("] not found.");
    getNextToken();


  std::string pre_dot="";
  bool is_self = false;
  bool is_attr = false;
  if (CurTok == tok_self)
  {
    is_self=true;
    getNextToken();
  }
  if (CurTok == tok_class_attr)
  {
    is_attr=true;
    pre_dot = IdentifierStr;
    std::cout << "Obj attr pinned_tensor: " << pre_dot << ".\n";
    getNextToken();
  }

  if (CurTok != tok_identifier)
    return LogError("Expected pinned tensor identifier name.");

  while (true) {
    std::string Name = IdentifierStr;
    pinnedTensorVars.push_back(IdentifierStr);
    getNextToken(); // eat identifier.

    
    std::unique_ptr<ExprAST> Init = nullptr;
    VarNames.push_back(std::make_pair(Name, std::move(Init)));

    // End of var list, exit loop.
    if (CurTok != ',')
      break;
    getNextToken(); // eat the ','.

    if (CurTok != tok_identifier)
      return LogError("Expected pinned tensor identifier names.");
  }



  auto aux = std::make_unique<PinnedTensorExprAST>(std::move(VarNames), "pinned_tensor",
                                             std::move(dims), init);
  aux->SetSelf(is_self);
  aux->SetIsAttribute(is_attr);
  aux->SetPreDot(pre_dot);

  
  if (CurTok==tok_space)
    getNextToken();
  
  return aux;
}





//
static std::unique_ptr<ExprAST> ParseTensorExpr(std::string class_name="") {
  bool is_weight;
  if (CurTok==tok_tensor)
    is_weight=false;
  if (CurTok==tok_param)
    is_weight=true;

  getNextToken(); // eat the tensor.

  
  if (CurTok != '[')
    return LogError("tensor declaration expected [");
  getNextToken();
  
  std::vector<std::unique_ptr<ExprAST>> dims;
  std::vector<std::pair<std::string, std::unique_ptr<ExprAST>>> VarNames;
  std::string init = "xavu_relu";
  //std::make_unique<NumberExprAST>(NumVal)
  
  while (true) {
    if (CurTok != tok_number && CurTok != tok_identifier && CurTok != tok_self)
      return LogError("Expected a number or var on the tensor dimension.");
    
    if (CurTok==tok_number)
    {
      if (std::fmod(NumVal, 1.0) != 0)
        LogWarning("A tensor's dimension should be int, not float.");
    
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
      //dims.push_back(std::move(ParseExpression(class_name)));
      dims.push_back(std::move(ParsePrimary(class_name)));
    }

    
    if (CurTok != ',')
      break;
    getNextToken(); // eat the ','.
  }

  
  if (CurTok != ']')
    return LogError("] not found at tensor declaration.");
    getNextToken();



  std::string pre_dot="";
  bool is_self = false;
  bool is_attr = false;
  if (CurTok == tok_self)
  {
    is_self=true; //TODO: set self per VarName instead.
    getNextToken();
  }
  if (CurTok == tok_class_attr)
  {
    is_attr=true;
    pre_dot = IdentifierStr;
    getNextToken();
  }

  if (CurTok != tok_identifier)
    return LogError("Expected tensor identifier name.");

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
      return LogError("Expected tensor identifier names.");
  }



  auto aux = std::make_unique<TensorExprAST>(std::move(VarNames), "tensor",
                                             std::move(dims), init, is_weight);
  aux->SetSelf(is_self);
  aux->SetIsAttribute(is_attr);
  aux->SetPreDot(pre_dot);
  
  if (CurTok==tok_space)
    getNextToken();
  
  return aux;
}




//
static std::unique_ptr<ExprAST> ParseConv2dExpr() {
  
  getNextToken(); // eat the Conv2d.
  
  if (CurTok != '[')
    return LogError("Conv2d declaration expected [");
    getNextToken();

  std::vector<std::unique_ptr<ExprAST>> dims;
  std::vector<std::pair<std::string, std::unique_ptr<ExprAST>>> VarNames;
  std::string init = "xavu_relu";
  //std::make_unique<NumberExprAST>(NumVal)
  
  while (true) {
    if (CurTok != tok_number && CurTok != tok_identifier && CurTok != tok_self)
      return LogError("Expected tensor dimension number.");
    
    if (CurTok==tok_number)
    {
      if (std::fmod(NumVal, 1.0) != 0)
        LogWarning("Tensor dimensions must be of type int. They are not supposed to be float.");
    
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
    return LogError("Expected ].");
    getNextToken();

  if (dims.size()<5)
    return LogError("Convolution declaration requires input and output channels, kernel size, stride and padding.");


  std::string pre_dot="";
  bool is_self = false;
  bool is_attr = false;
  if (CurTok == tok_self)
  {
    is_self=true;
    getNextToken();
  }
  if (CurTok == tok_class_attr)
  {
    is_attr=true;
    pre_dot = IdentifierStr;
    std::cout << "Obj attr pinned_tensor: " << pre_dot << ".\n";
    getNextToken();
  }

  if (CurTok != tok_identifier)
    return LogError("Expected conv identifier name.");



  while (true) {
    std::string Name = IdentifierStr;
    
    getNextToken(); // eat identifier.

    
    std::unique_ptr<ExprAST> Init = nullptr;
    VarNames.push_back(std::make_pair(Name, std::move(Init)));
    functionVars[Name] = "ConvForward2d";

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
  aux->SetSelf(is_self);
  aux->SetIsAttribute(is_attr);
  aux->SetPreDot(pre_dot);

  
  if (CurTok==tok_space)
    getNextToken();
  
  return aux;
}





//
static std::unique_ptr<ExprAST> ParseMaxPool2dExpr() {
  std::string type;
  if (CurTok==tok_maxpool2d)
    type = "max";
  if (CurTok==tok_avgpool2d)
    type = "avg";

  getNextToken(); // eat the MaxPool2d.
  
  if (CurTok != '[')
    return LogError("MaxPool2d declaration expected [");
    getNextToken();

  std::vector<std::unique_ptr<ExprAST>> dims;
  std::vector<std::pair<std::string, std::unique_ptr<ExprAST>>> VarNames;
  std::string init = "xavu_relu";
  //std::make_unique<NumberExprAST>(NumVal)
  
  while (true) {
    if (CurTok != tok_number && CurTok != tok_identifier && CurTok != tok_self)
      return LogError("Expected tensor dimension number.");
    
    if (CurTok==tok_number)
    {
      if (std::fmod(NumVal, 1.0) != 0)
        LogWarning("Tensor dimensions must be of type int. They are not supposed to be float.");
    
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
    return LogError("Expected ].");
    getNextToken();

  if (dims.size()<3)
    return LogError("MaxPool2d requires kernel size, stride and padding.");


  std::string pre_dot="";
  bool is_self = false;
  bool is_attr = false;
  if (CurTok == tok_self)
  {
    is_self=true;
    getNextToken();
  }
  if (CurTok == tok_class_attr)
  {
    is_attr=true;
    pre_dot = IdentifierStr;
    std::cout << "Obj attr pinned_tensor: " << pre_dot << ".\n";
    getNextToken();
  }

  if (CurTok != tok_identifier)
    return LogError("Expected conv identifier name.");



  while (true) {
    std::string Name = IdentifierStr;
    
    getNextToken(); // eat identifier.

    
    std::unique_ptr<ExprAST> Init = nullptr;
    VarNames.push_back(std::make_pair(Name, std::move(Init)));
    functionVars[Name] = "MaxPoolForward2d";

    // End of var list, exit loop.
    if (CurTok != ',')
      break;
    getNextToken(); // eat the ','.

    if (CurTok != tok_identifier)
      return LogError("Esperado um ou mais identificadores após var.");
  }



  auto aux = std::make_unique<MaxPool2dExprAST>(std::move(VarNames), type,
                                             std::move(dims[0]), std::move(dims[1]), std::move(dims[2]));
  aux->SetSelf(is_self);
  aux->SetIsAttribute(is_attr);
  aux->SetPreDot(pre_dot);

  
  if (CurTok==tok_space)
    getNextToken();
  
  return aux;
}





//
static std::unique_ptr<ExprAST> ParseBatchNorm2dExpr() {
  std::string type;

  getNextToken(); // eat the BatchNorm2d.
  
  if (CurTok != '[')
    return LogError("BatchNorm2d declaration expected [");
    getNextToken();

  std::vector<std::unique_ptr<ExprAST>> dims;
  std::vector<std::pair<std::string, std::unique_ptr<ExprAST>>> VarNames;
  std::string init = "xavu_relu";
  //std::make_unique<NumberExprAST>(NumVal)
  
  while (true) {
    if (CurTok != tok_number && CurTok != tok_identifier && CurTok != tok_self)
      return LogError("Expected tensor dimension number.");
    
    if (CurTok==tok_number)
    {
      if (std::fmod(NumVal, 1.0) != 0)
        LogWarning("Tensor dimensions must be of type int. They are not supposed to be float.");
    
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
    return LogError("Expected ].");
    getNextToken();

  if (dims.size()<1)
    return LogError("BatchNorm2d requires input channels, kernel size, stride and padding.");


  std::string pre_dot="";
  bool is_self = false;
  bool is_attr = false;
  if (CurTok == tok_self)
  {
    is_self=true;
    getNextToken();
  }
  if (CurTok == tok_class_attr)
  {
    is_attr=true;
    pre_dot = IdentifierStr;
    std::cout << "Obj attr pinned_tensor: " << pre_dot << ".\n";
    getNextToken();
  }

  if (CurTok != tok_identifier)
    return LogError("Expected BatchNorm2d identifier name.");



  while (true) {
    std::string Name = IdentifierStr;
    
    getNextToken(); // eat identifier.

    
    std::unique_ptr<ExprAST> Init = nullptr;
    VarNames.push_back(std::make_pair(Name, std::move(Init)));
    functionVars[Name] = "BatchNormForward2d";

    // End of var list, exit loop.
    if (CurTok != ',')
      break;
    getNextToken(); // eat the ','.

    if (CurTok != tok_identifier)
      return LogError("Expected one or more identifiers at BatchNorm2d.");
  }



  auto aux = std::make_unique<BatchNorm2dExprAST>(std::move(VarNames), type,
                                             std::move(dims[0]));
  aux->SetSelf(is_self);
  aux->SetIsAttribute(is_attr);
  aux->SetPreDot(pre_dot);

  
  if (CurTok==tok_space)
    getNextToken();
  
  return aux;
}




static std::unique_ptr<ExprAST> ParseLockExpr(std::string class_name="") {
  int cur_level_tabs = SeenTabs;
  getNextToken(); // eat the lock.


  std::string Name = "mutex";
  if (CurTok==tok_str)
  {
    Name = IdentifierStr;
    getNextToken(); // eat lock string "" name
  }



  if (lockVars.count(Name) == 0)
  {
    pthread_mutex_t* _mutex = new pthread_mutex_t;
    if (pthread_mutex_init(&mutex, NULL) != 0) {
      printf("Mutex initialization failed\n");
      return nullptr;
    }
    
    lockVars[IdentifierStr] = _mutex;
    
  }
  
  std::vector<std::unique_ptr<ExprAST>> Body = ParseIdentedBodies(cur_level_tabs, class_name);


  Body.push_back(std::make_unique<UnlockExprAST>(Name));

  return std::make_unique<LockExprAST>(std::move(Body), Name);
}


static std::unique_ptr<ExprAST> ParseMustBeVar(std::string class_name="", std::string expr_name="") {

  std::unique_ptr<ExprAST> expr;

  if (CurTok==tok_class_attr||CurTok==tok_self)
    expr = ParseSelfExpr(class_name);
  else if (CurTok==tok_identifier)
    expr = ParseIdentifierExpr(class_name);
  else
  {
    std::string _error = expr_name + " expression expected a simple identifier, not another expression.";
    LogError(_error);
  }

  return std::move(expr);
}



static std::unique_ptr<ExprAST> ParseReturnExpr(std::string class_name="") {
  getNextToken(); // eat return
  std::cout << "Parsing var expr\n";

  std::vector<std::unique_ptr<ExprAST>> Vars, Destiny;
  std::vector<bool> IsAs;

  // At least one variable name is required.
  if (CurTok != tok_identifier)
    return LogError("Expected identifier after return.");

  while (true) {
    std::unique_ptr<ExprAST> expr, aux;


    expr = ParseMustBeVar(class_name, "return");

    
    if (CurTok == tok_as)
    {
      getNextToken(); // eat as
      aux = std::move(expr);

      expr = ParseMustBeVar(class_name, "return");

      IsAs.push_back(true);
      Vars.push_back(std::move(aux));
      Destiny.push_back(std::move(expr));

    } else {
      Destiny.push_back(std::move(expr));

      expr = std::make_unique<NumberExprAST>(0.0f);
      IsAs.push_back(false);
      Vars.push_back(std::move(expr));
    }

    // End of var list, exit loop.
    if (CurTok != ',')
      break;
    getNextToken(); // eat the ','.

    
  }

  if (CurTok==tok_space)
    getNextToken();


  return std::make_unique<ReturnExprAST>(std::move(Vars), std::move(IsAs), std::move(Destiny));
}




/// primary
///   ::= identifierexpr
///   ::= numberexpr
///   ::= parenexpr
///   ::= ifexpr
///   ::= forexpr
///   ::= varexpr
static std::unique_ptr<ExprAST> ParsePrimary(std::string class_name="") {
  switch (CurTok) {
  default:
    //return std::move(std::make_unique<NumberExprAST>(0.0f));
    return LogErrorT(CurTok);
  case tok_identifier:
    return ParseIdentifierExpr();
  case tok_class_attr:
    return ParseSelfExpr(class_name);
  case tok_self:
    return ParseSelfExpr(class_name);
  case tok_number:
    return ParseNumberExpr();
  case tok_str:
    return ParseStringExpr();
  case tok_var_str:
    return ParseStrExpr();
  case tok_str_vec:
    return ParseStrVecExpr();
  case tok_float_vec:
    return ParseStrVecExpr();
  case '(':
    return ParseParenExpr();
  case tok_if:
    return ParseIfExpr(class_name);
  case tok_for:
    return ParseForExpr(class_name);
  case tok_while:
    return ParseWhileExpr(class_name);
  case tok_async_finish:
    return ParseFinishExpr(class_name);
  case tok_async:
    return ParseAsyncExpr(class_name);
  case tok_lock:
    return ParseLockExpr(class_name);
  case tok_return:
    return ParseReturnExpr(class_name);
  case tok_var:
    return ParseVarExpr(class_name);
  case tok_tensor:
    return ParseTensorExpr(class_name);
  case tok_param:
    return ParseTensorExpr(class_name);
  case tok_pinned_tensor:
    return ParsePinnedTensorExpr();
  case tok_conv2d:
    return ParseConv2dExpr();
  case tok_maxpool2d:
    return ParseMaxPool2dExpr();
  case tok_avgpool2d:
    return ParseMaxPool2dExpr();
  case tok_batchnorm2d:
    return ParseBatchNorm2dExpr();
  case tok_space:
    getNextToken();
    return ParsePrimary(class_name);
  }
}

/// unary
///   ::= primary
///   ::= '!' unary
static std::unique_ptr<ExprAST> ParseUnary(std::string class_name="") {
  //std::cout <<"Parse unary\n";
  if(CurTok==tok_space)
    getNextToken();
  // If the current token is not an operator, it must be a primary expr.
  
  //std::cout << "Unary current token " << CurTok << "\n";
  if (!isascii(CurTok) || CurTok == '(' || CurTok == ',')
  {
    //std::cout << "Returning, non-ascii found.\n";
    return ParsePrimary(class_name);
  }
  
  
  // If this is a unary operator, read it.
  int Opc = CurTok;
  
  //std::cout << "Unary expr\n";
  getNextToken();
  if (auto Operand = ParseUnary(class_name))
    return std::make_unique<UnaryExprAST>(Opc, std::move(Operand));
  return nullptr;
}


/// binoprhs
///   ::= ('+' unary)*
static std::tuple<std::unique_ptr<ExprAST>, int> ParseBinOpRHS(int ExprPrec,
                                              std::unique_ptr<ExprAST> LHS,
                                              std::string class_name="") {
  
  
  int LhsTok = 0;
  int RhsTok = 0;

  int L_cuda = type_float;
  int R_cuda = type_float;

  std::string LName, RName;
  if (LHS->GetType()=="tensor")
    L_cuda = type_tensor;
  if (LHS->GetType()=="pinned_tensor")
    L_cuda = type_pinned_tensor;

  while (true)
  {
    // If this is a binop, find its precedence.
    int TokPrec = get_tokenPrecedence();

    // If this is a binop that binds at least as tightly as the current binop,
    // consume it, otherwise we are done.
    

    if (TokPrec==BinopPrecedence[':'])
    {
      getNextToken();
      return std::make_tuple(std::move(LHS), L_cuda);
    }
    if (TokPrec < ExprPrec)
      return std::make_tuple(std::move(LHS), L_cuda);
    
      


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


    // Get the Right Hand Side token for debugging only
    RhsTok = CurTok;

    
    auto RHS = ParseUnary(class_name); // Returns an identifier, number or expression result
    if (!RHS)
      return std::make_tuple(nullptr,0);


    if (RHS->GetType()=="tensor")
      R_cuda=type_tensor;
    if (RHS->GetType()=="pinned_tensor")
      R_cuda=type_pinned_tensor;
    if (RHS->GetType()=="object"||RHS->GetType()=="object_vec")
      R_cuda=type_object;
    
    
    
    

    // If BinOp binds less tightly with RHS than the operator after RHS, let
    // the pending operator take RHS as its LHS.
    int NextPrec = get_tokenPrecedence();
    

    if (TokPrec < NextPrec)
    {
      //std::cout << NextPrec << " Next Prec\n";
        
      auto tuple = ParseBinOpRHS(TokPrec + 1, std::move(RHS));
      RHS = std::move(std::get<0>(tuple));
      R_cuda = std::get<1>(tuple);


      if (!RHS)
        return std::make_tuple(nullptr,0);
      
    }


    //std::cout << "\nBinary expression of BinOp and Rhs:" << "\n";
    //std::cout << ReverseToken(BinOp) << " " << ReverseToken(RhsTok) << "\n";
    //std::cout << "L type: " << L_cuda << " R type: " << R_cuda << "\n\n";

    if (R_cuda==type_object)
    {
      LHS = std::make_unique<BinaryObjExprAST>(BinOp, std::move(LHS), std::move(RHS));
      LHS->SetType("object");
    }
    else if (L_cuda==type_tensor && R_cuda==type_pinned_tensor)
    {
      //std::cout << "\nParse BinaryTensorPinned " << ReverseToken(BinOp) <<  "\n";
      LHS = std::make_unique<BinaryTensorPinnedExprAST>(BinOp,
                                                      std::move(LHS), std::move(RHS));
      LHS->SetType("tensor");
    }
    else if (L_cuda==type_pinned_tensor && R_cuda==type_float)
    {
      LHS = std::make_unique<BinaryPinnedScalarExprAST>(BinOp,
                                                      std::move(LHS), std::move(RHS));
      LHS->SetType("pinned_tensor");
    }
    else if (L_cuda==type_tensor && R_cuda==type_float)
    {
      LHS = std::make_unique<BinaryTensorScalarExprAST>(BinOp,
                                                      std::move(LHS), std::move(RHS));
      LHS->SetType("tensor");  
    }
    else if (L_cuda==type_float && R_cuda==type_tensor)
    {
      std::cout << "Reverse LHS and RHS\n";
      //std::cout << "Bin op: " << BinOp << "\n";


      if (BinOp==47)
        BinOp = 77; // scalar / tensor

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
      LHS->SetType("tensor");  
      L_cuda=type_tensor;
      R_cuda=type_float;
    }
    else if (L_cuda==type_tensor && R_cuda==type_tensor)
    { 
      LHS = std::make_unique<BinaryTensorTensorExprAST>(BinOp,
                                                      std::move(LHS), std::move(RHS));
      R_cuda=type_float;
      LHS->SetType("tensor");
    }
    else
      LHS = std::make_unique<BinaryExprAST>(BinOp, std::move(LHS), std::move(RHS));
     

    LhsTok = RhsTok;    
  
  }
}


/// expression
///   ::= unary binoprhs
///
static std::unique_ptr<ExprAST> ParseExpression(std::string class_name) {
  
  //std::cout << "Parse Expression\n";
  
  auto LHS = ParseUnary(class_name);
  if (!LHS)
    return nullptr;

  return std::get<0>(ParseBinOpRHS(0, std::move(LHS), class_name));
}

/// prototype
///   ::= id '(' id* ')'
///   ::= binary LETTER number? (id, id)
///   ::= unary LETTER (id)
static std::unique_ptr<PrototypeAST> ParsePrototype(std::string class_name="") {
  std::string FnName = class_name;
  std::string _class, method;
  method = "";
  _class = class_name;

  unsigned Kind = 0; // 0 = identifier, 1 = unary, 2 = binary.
  unsigned BinaryPrecedence = 30;

  switch (CurTok) {
  default:
    return LogErrorP("Esperado nome da função no protótipo");
  case tok_identifier:
    FnName += IdentifierStr;
    method = IdentifierStr;
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


  std::string type;
  std::vector<std::string> ArgNames, Types;


  if (class_name!="") // If it is a class method, add self
  {
    Types.push_back("s");
    ArgNames.push_back("self");
  }
  Types.push_back("s");
  ArgNames.push_back("scope_str");
  Types.push_back("s");
  ArgNames.push_back("previous_scope");

  while (CurTok != ')')
  {
    type="str";
    if (IdentifierStr=="t")
      type="tensor";
    if (IdentifierStr=="c")
      type="function";
    if (IdentifierStr=="f")
      type="float";

    if (IdentifierStr!="t" && IdentifierStr!="f" && IdentifierStr!="s" && IdentifierStr!="c")
      LogErrorP_to_comma("Prototype var type must be t, f, s or c");
    else {
      Types.push_back(IdentifierStr);
      getNextToken(); // eat arg type

      ArgNames.push_back(IdentifierStr);

      if (type=="float")
        floatVars.push_back(IdentifierStr);
      else if (type=="tensor")
        tensorVars.push_back(IdentifierStr);
      else if (type=="function")
        functionVars[IdentifierStr] = "ConvForward2d";
      else
        strVars.push_back(IdentifierStr);
      
      getNextToken(); // eat arg name
    }
    


    if (CurTok == ')')
        break;
      
    if (CurTok != ',')
    {
      return LogErrorP("Expected ')' or ',' at prototype arguments list.");
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


  return std::make_unique<PrototypeAST>(FnName, _class, method, ArgNames, Types, Kind != 0,
                                         BinaryPrecedence);
}



/// definition ::= 'def' prototype expression
static std::unique_ptr<FunctionAST> ParseDefinition(std::string class_name="") {

  int cur_level_tabs = SeenTabs;

  getNextToken(); // eat def.


  auto Proto = ParsePrototype(class_name);
  if (!Proto)
  {
    std::string _error = "Error defining " + class_name + " prototype.";  
    LogError(_error);
    return nullptr;
  } 
  
  
  std::vector<std::unique_ptr<ExprAST>> Body;

  while(!in_char(CurTok, terminal_tokens))
  {
    if (SeenTabs <= cur_level_tabs && CurTok != tok_space)
      break;
      

    if (CurTok==tok_space)
      getNextToken();

    if (SeenTabs <= cur_level_tabs)
      break;

    Body.push_back(std::move(ParseExpression(class_name)));
  }

  //std::cout << "function number of expressions: " << Body.size() << "\n";

  if (Body.size()==0)
  {
    std::string _error = "Function " + class_name + "'s body was not declared.";  
    LogError(_error);
    return nullptr;
  } 

  return std::make_unique<FunctionAST>(std::move(Proto), std::move(Body));
  
}


/// toplevelexpr ::= expression
static std::unique_ptr<FunctionAST> ParseTopLevelExpr() {
  //std::cout << "Top Level Expression\n";

  
  std::vector<std::unique_ptr<ExprAST>> Body;
  while(!in_char(CurTok, terminal_tokens))
  {
    Body.push_back(std::move(ParseExpression()));
    //std::cout << "\n\nTop level expr cur tok: " << ReverseToken(CurTok) <<  ".\n";
    //std::cout << "Top level expr number of expressions: " << Body.size() <<  ".\n\n\n";
  }
  

  // Make an anonymous proto.
  auto Proto = std::make_unique<PrototypeAST>("__anon_expr", "", "",
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
static std::map<std::string, Value *> NamedValues;
static std::map<std::string, char *> NamedStrs;
static std::map<std::string, AllocaInst *> NamedStrVecs;
static std::map<std::string, AllocaInst *> NamedFloatVecs;
static std::map<std::string, std::vector<char *>> ClassStrVecs;
static std::map<std::string, std::vector<float>> ClassFloatVecs;
static std::map<std::string, float> NamedClassValues;
static std::map<std::string, std::string> NamedObjects;

static std::map<std::string, std::vector<std::pair<std::string, std::string>>> ScopeVarsToClean;


// Aux to not lose pointers
std::map<std::string, std::string> AuxRandomStrs;
std::map<std::string, std::vector<char *>> StrVecAuxHash;
std::map<std::string, std::vector<float>>  FloatVecAuxHash;


// Cuda Parallellism
constexpr int num_parallel_streams = 2;
cudaStream_t parallel_streams[num_parallel_streams];
cudaEvent_t parallel_events[num_parallel_streams];

// Optimizer
static std::map<std::string, float *> NamedParamGrads;


// File Handling
std::vector<char *> glob_str_files;




// Handle Class self with phantom argument



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
    return LogError("Expected class name");
  std::string Name = IdentifierStr;

  Classes.push_back(Name);

  getNextToken();

  if(CurTok==tok_space)
    getNextToken();
  

  if (CurTok!=tok_def)
    return LogError("A class definition requires it's functions.");

  int i=0;
  while(CurTok==tok_def)
  {
    
    auto Func = ParseDefinition(Name);
    if (!Func)
      return nullptr;
      //return LogError("Falha no parsing da função da Classe.");
    if (!ends_with(Func->getProto().getName(),"__init__") && i==0)
      return LogError("Class requires __init__ method");
    
    //std::cout << "THE FUNCTION WAS CREATED AS: " << Func->getProto().getName() << "\n";


    std::string proto_name = Func->getProto().getName();    


    FunctionProtos[proto_name] =
      std::make_unique<PrototypeAST>(Func->getProto());
    ExitOnErr(TheJIT->addAST(std::move(Func)));

    if(CurTok==';')
      getNextToken();

    if(CurTok==tok_space)
      getNextToken();

    i+=1;
  }
  
  return nullptr;
}



extern "C" void sleep(float id)
{
  std::cout << "\n\nSleep " << id << " begin" << "\n";
  std::random_device rd;
  std::mt19937 gen(rd());

  std::uniform_int_distribution<> dis(3, 7); // Generate between 1 and 100
  int random_number = dis(gen);

  //std::this_thread::sleep_for(std::chrono::seconds(random_number));
  std::this_thread::sleep_for(std::chrono::seconds((int)id));

  std::cout << "Sleep " << id << " finish" << "\n";

  //return id;
}


extern "C" float silent_sleep(float id)
{
  std::random_device rd;
  std::mt19937 gen(rd());

  std::uniform_int_distribution<> dis(3, 7); // Generate between 1 and 100
  int random_number = dis(gen);

  //std::this_thread::sleep_for(std::chrono::seconds(random_number));
  std::this_thread::sleep_for(std::chrono::seconds((int)id));

 
  return 0;
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




int resultingDimsProdOnMult(std::vector<float> Ldims, std::vector<float> Rdims)
{
  float aux=1;
  for (int i = 0; i < Ldims.size()-1; i++)
    aux = aux * Ldims[i];
  aux = aux * Rdims[0];
  return (int)aux;
}


extern "C" void *gpu(Tensor tensor)
{
  //std::cout << "\nGpu transfer for: " << tensor.name << "\n";
  
  float *tensor_ptr, *tensor_cpu;

  tensor_ptr = tensor.tensor_ptr;
  tensor_cpu = tensor.cpu_tensor_ptr;
  float dims_prod = tensor.dims_prod;
  

  
  cudaError_t err = cudaMemcpy(tensor_ptr, tensor_cpu, dims_prod * sizeof(float), cudaMemcpyHostToDevice);


  if (err != cudaSuccess) {
    // Handle error
    fprintf(stderr, "CPU to GPU tensor transfer failed: %s\n", cudaGetErrorString(err));
  } 
  //std::cout << "Transfer succeed" << "\n\n";

  
  Tensor *new_tensor = createTensor(tensor_ptr, tensor.dims, dims_prod, false, "gpu");
  new_tensor->op = gpu_op;
  return new_tensor;
}


extern "C" void *gpuw(Tensor tensor, float idx)
{
  //std::cout << "\nGpu transfer for: " << tensor.name << " on worker " << idx << "\n";
  
  float *tensor_ptr, *tensor_cpu;

  tensor_ptr = tensor.tensor_ptr;
  tensor_cpu = tensor.cpu_tensor_ptr;
  std::vector<float> dims, batchless_dims;
  dims = tensor.dims;
  

  batchless_dims = BatchLessDims(dims);
  float batchless_dims_prod = (float)DimsProd(batchless_dims);


  tensor_cpu = tensor.cpu_tensor_ptr + static_cast<int>(idx*batchless_dims_prod);


  cudaError_t err = cudaMemcpy(tensor_ptr, tensor_cpu, batchless_dims_prod * sizeof(float), cudaMemcpyHostToDevice);

  if (err != cudaSuccess) {
    // Handle error
    fprintf(stderr, "CPU to GPU tensor transfer failed: %s\n", cudaGetErrorString(err));
  } 
  //std::cout << "Transfer succeed" << "\n\n";

  Tensor *new_tensor = createTensor(tensor_ptr, batchless_dims, batchless_dims_prod, false, "gpu");
  new_tensor->op = gpu_op;
  return new_tensor;
}



std::vector<float> NewDimsOnMult(std::vector<float> Ldims, std::vector<float> Rdims)
{
  

  std::vector<float> new_dims;
  if (Ldims[Ldims.size()-1]!=Rdims[Rdims.size()-1])
  {
    LogError("The last dimension of multiplied tensors must be the same.");
    std::cout << "Dim LHS: ";
    PrintDims(Ldims);
    std::cout << "Dim RHS: ";
    PrintDims(Rdims);
    return {}; 
  }
  for (int i = 0; i < Ldims.size()-1; i++)
    new_dims.push_back(Ldims[i]);
  new_dims.push_back(Rdims[0]);


  return new_dims;
}


extern "C" void *NewDimsOnIdx(std::vector<float> dims)
{
  std::vector<float> new_dims;

  for (int i = 0; i < dims.size()-1; i++)
    new_dims.push_back(dims[i+1]);


  // Aux to not lose pointers
  std::string random_str = RandomString(15);
  NamedDims[random_str] = new_dims; // Deal with new_dims being deleted after scope finished.
  AuxRandomStrs[random_str] = "dim";


  std::cout << "NewDimsOnIdx" << "\n";
  PrintDims(NamedDims[random_str]);

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

extern "C" float unbug(){
  return 0;
}

extern "C" float UnbugFloat(float value){
  return value;
}


extern "C" float PrintStr(char* value){
  std::cout << "Str: " << value << "\n";
  return 0;
}


extern "C" float PrintStrVec(std::vector<char*> vec)
{
  for (int i=0; i<vec.size(); i++)
    std::cout << vec[i] << "\n";

  return 0;
}


extern "C" float PrintFloatVec(std::vector<float> vec)
{

  std::cout << "Float vector:\n[";
  for (int i=0; i<vec.size()-1; i++)
    std::cout << "" << vec[i] << ", ";
  std::cout << "" << vec[vec.size()-1];
  std::cout << "]\n\n";

  return 0;
}


extern "C" float first_nonzero(char *self)
{
  //std::cout << "first_nonzero call of: " << self <<"\n";

  std::vector<float> vec;
  vec = ClassFloatVecs[self];

  /*
  std::cout << "[";
  for (int i=0; i<vec.size(); i++)
    std::cout << vec[i] << ", ";
  std::cout << "]" << "\n";
  */

  float idx = -1;
  for (int i=0; i<vec.size(); i++)
    if (vec[i]!=0)
    {
      idx = i;
      break;
    }

  delete[] self;
  return idx;
}



extern "C" float LenStrVec(std::vector<char*> vec)
{
  return (float) vec.size();
}



extern "C" void * ShuffleStrVec(std::vector<char*> vec)
{
  std::random_device rd;
  std::mt19937 g(rd());

  //std::cout << "Shuffling vector: " << &vec << " of size " << vec.size() << "\n";
  std::shuffle(vec.begin(), vec.end(), g);

  
  return &vec;
}


//deprecated
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


extern "C" void *LoadTensor(char *tensor_name){
  //std::cout << "\n\nLOAD TENSOR: " << tensor_name <<  "\n\n\n";
  Tensor *ret = NamedTensorsT[tensor_name];
  delete[] tensor_name;
  return ret;
}


extern "C" void *LoadDims(char *tensor_name) // TODO: invert this back
{
  std::cout << "LOADING DIMS"  << "\n";
  PrintDims(NamedDims[tensor_name]);

  std::string random_str = RandomString(15);
  NamedDims[random_str] = NamedDims[tensor_name];
  AuxRandomStrs[random_str] = "dim";
  delete[] tensor_name;

  return &NamedDims[random_str];
}


extern "C" float print(char* str, float x){
  std::string _str = str;
  std::cout << "\n" << _str << " " << x << "\n";
  return 0;
}


extern "C" float PrintTensor(char* tensorName){
  std::cout << "Printing Tensor " << tensorName << "\n";



  Tensor *tensor = NamedTensorsT[tensorName];
  int arr_size = tensor->dims_prod;
  float *tensor_cpu = new float[arr_size];

  
  std::vector<float> dims = tensor->dims;
  
  
  
  cudaDeviceSynchronize();
  cudaCheck(cudaMemcpy(tensor_cpu, tensor->tensor_ptr, arr_size*sizeof(float), cudaMemcpyDeviceToHost));


  std::cout << "\nTensor \033[95m" << tensorName << "\033[0m:\n\n";
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
    if (tensor_cpu[i]>=0)
      precision=4;
    else
      precision=3;
    std::cout << std::fixed  << std::setprecision(precision) << tensor_cpu[i];


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
  std::cout << "\n\n";

  delete[] tensor_cpu;

  return 0;
}



extern "C" float print_tensor(Tensor tensor){
    char* tensorName = new char[tensor.name.size() + 1]; // Allocate memory for the C-style string
    std::strcpy(tensorName, tensor.name.c_str()); // Copy the string

    PrintTensor(tensorName);

    delete[] tensorName;
  return 0;
}


extern "C" float PrintTensorF(const float *cuda_tensor, int d1, int d2){

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



Value *NameSolverAST::codegen(Value *first_arg, Value *scope_str, Value *previous_scope) {
  if (not ShallCodegen)
    return ConstantFP::get(*TheContext, APFloat(0.0f));

  std::cout << "\n\n\nName solver type: " << Type << "\n\n\n\n";


  Value *name;
  int type;
  std::vector<std::unique_ptr<ExprAST>> idx;

  
  bool include_scope = GetSolverIncludeScope();
  Value *var_name = Builder->CreateCall(TheModule->getFunction("GetEmptyChar"), {});

  


  if(Names.size()>1)
    for (int i=0; i<Names.size()-1;i++)
    {
      name = Builder->CreateGlobalString(std::get<0>(Names[i]));
      type = std::get<1>(Names[i]);
      idx = std::move(std::get<2>(Names[i]));

      std::cout << "NameSolver[" << i<< "]:  " << std::get<0>(Names[i]) << ", type: " << type << "\n";

      if (i==0)
      {
        if (type==type_self)
          var_name = Builder->CreateCall(TheModule->getFunction("ConcatStrFreeLeft"),
                                                          {var_name, first_arg});
        else
        {
          if((Type=="object"||Type=="tensor"||Type=="float"||type==type_object_name||Type=="str")&&include_scope)
            var_name = Builder->CreateCall(TheModule->getFunction("ConcatStrFreeLeft"),
                                                          {var_name, scope_str});
        }
      }

      if (type==type_object_name)
      {
        var_name = Builder->CreateCall(TheModule->getFunction("ConcatStrFreeLeft"),
                                                        {var_name, name});
        var_name = Builder->CreateCall(TheModule->getFunction("LoadObject"),
                                                        {var_name});

      }

      if (type==type_attr||type==type_var)
        var_name = Builder->CreateCall(TheModule->getFunction("ConcatStrFreeLeft"),
                                                        {var_name, name});

      if (type==type_vec)
      {
        Value *_idx = idx[0]->codegen(first_arg, scope_str, previous_scope);
        var_name = Builder->CreateCall(TheModule->getFunction("ConcatStrFreeLeft"),
                                                        {var_name, name});
        var_name = Builder->CreateCall(TheModule->getFunction("ConcatNumToStrFree"),
                                                        {var_name, _idx});
        var_name = Builder->CreateCall(TheModule->getFunction("LoadObjectScopeName"),
                                                        {var_name});
      }
    }

  if(Names.size()==1)// Concat scope only
    if((Type=="object"||Type=="tensor"||Type=="float"||Type=="str")&&include_scope)
      var_name = Builder->CreateCall(TheModule->getFunction("ConcatStrFreeLeft"),
                                                        {var_name, scope_str});



  name = Builder->CreateGlobalString(std::get<0>(Names[Names.size()-1]));
  idx = std::move(std::get<2>(Names[Names.size()-1]));

  //std::cout << "\n\n\nNAMESOLVER TYPE OF LAST: " << type << "\n";
  //std::cout << "For: " << std::get<0>(Names[Names.size()-1]) << "\n";
  //std::cout << "Type: " << Type << "\n";
  var_name = Builder->CreateCall(TheModule->getFunction("ConcatStrFreeLeft"),
                                                    {var_name, name});
  if (Type=="object_vec")
  {
    Value *_idx = idx[0]->codegen(first_arg, scope_str, previous_scope);
    var_name = Builder->CreateCall(TheModule->getFunction("ConcatNumToStrFree"),
                                                      {var_name, _idx});
  }


  
  
  return var_name;
}

Value *NumberExprAST::codegen(Value *first_arg, Value *scope_str, Value *previous_scope) {
  if (not ShallCodegen)
    return ConstantFP::get(*TheContext, APFloat(0.0f));
  
  return ConstantFP::get(*TheContext, APFloat(Val));
}

Value *StringExprAST::codegen(Value *first_arg, Value *scope_str, Value *previous_scope) {
  if (not ShallCodegen)
    return ConstantFP::get(*TheContext, APFloat(0.0f));
  SetName(Val);
  return Builder->CreateGlobalString(Val);
}




//===----------------------------------------------------------------------===//
// Dataset
//===----------------------------------------------------------------------===//


extern "C" void * zeros_vec(float size) {
  // TODO: turn into python like expression [0]*size

  std::vector<float> vec = std::vector<float>(static_cast<size_t>(size), 0.0f);
  

  // Aux to not lose pointers
  std::string random_str = RandomString(15);
  FloatVecAuxHash[random_str] = vec;
  AuxRandomStrs[random_str] = "float_vec";
    
  return &FloatVecAuxHash[random_str];
}

extern "C" void * ones_vec(float size) {
  // TODO: turn into python like expression [0]*size

  std::vector<float> vec = std::vector<float>(static_cast<size_t>(size), 1.0f);
  

  // Aux to not lose pointers
  std::string random_str = RandomString(15);
  FloatVecAuxHash[random_str] = vec;
  AuxRandomStrs[random_str] = "float_vec";
    
  return &FloatVecAuxHash[random_str];
}


extern "C" void * _glob_b_(char *pattern) {
  glob_t glob_result;

  std::vector<char *> ret;

  if (glob(pattern, GLOB_TILDE, NULL, &glob_result) == 0) {
      for (size_t i = 0; i < glob_result.gl_pathc; ++i) {

        ret.push_back(strdup(glob_result.gl_pathv[i]));
      }
      globfree(&glob_result);
  }


  if (ret.size()<1)
    LogErrorS("Glob falhou ao encontrar arquivos.");
    
  // Aux to not lose pointers
  std::string random_str = RandomString(15);
  StrVecAuxHash[random_str] = ret;
  AuxRandomStrs[random_str] = "str_vec";
    
  return &StrVecAuxHash[random_str];
}



float *current_data;



extern "C" float load_preprocess_img(Tensor tensor, char *img_name)
{
  float *img;
  img = load_img(img_name); 
  
  std::vector<float> dims = tensor.dims;

  
  // Last three dims.
  int img_dims_prod = dims[dims.size()-1]*dims[dims.size()-2]*dims[dims.size()-3];


  current_data = new float[img_dims_prod];
  cudaCheck(cudaMallocHost(&current_data, img_dims_prod*sizeof(float)));


  for (int j = 0; j < img_dims_prod; ++j)
    current_data[j] = img[j];
  delete[] img;


  float *x = tensor.tensor_ptr;
  //cudaMalloc(&x, img_dims_prod*sizeof(float));
  cudaCheck(cudaMemcpy(x, current_data, img_dims_prod*sizeof(float), cudaMemcpyHostToDevice));


  return 0;
}


//===----------------------------------------------------------------------===//
// Tensor Functionalities
//===----------------------------------------------------------------------===//


extern "C" void *view(Tensor *tensor, float first_dim, ...)
{
  //std::cout << "Executing: " << tensor.name << "." << "view" << "\n";
   
  std::vector<float> new_dims, new_dims_no_minus, current_dims;
  bool has_minus = false;
  current_dims = tensor->dims;

  
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
    if (dim==TERMINATE_VARARG)
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
      LogErrorS("Automatic view dimension calculus resulted on a non-integer dimension.");
      PrintDims(current_dims);
      std::cout << "Current dims product: " << current_dims_prod  << ".\n";
      PrintDims(new_dims);
      std::cout << "New dims product: " << std::to_string(DimsProd(new_dims_no_minus))  << ".\n";
      return 0;
    }
    
    for (int i=0; i<new_dims.size(); i++)
      if (new_dims[i]==-1)
        new_dims[i] = hidden_dim;
    
  } else {
    if (current_dims_prod != new_dims_prod)
    {
      LogErrorS("Incompatible view dimensions.");
      PrintDims(current_dims);
      std::cout << "Current dims product: " << current_dims_prod  << ".\n";
      PrintDims(new_dims);
      std::cout << "New dims product: " << new_dims_prod  << ".\n";
      return 0;
    }
  }

  

  Tensor *new_tensor = createTensor(tensor->tensor_ptr, new_dims, DimsProd(new_dims), false, "");
  new_tensor->view_of = tensor->name;
  new_tensor->op=view_op;
  return new_tensor;
}


//===----------------------------------------------------------------------===//
// Tensor -- Scalar   Operations
//===----------------------------------------------------------------------===//

std::vector<int> CalculateGridAndBlockSizes(int dims_prod, int pre_block_size=-1)
{

  int grid_size, block_size, shared_mem_size;

  if (pre_block_size==-1)
  {
    if (dims_prod<64)
      block_size=32;
    else if (dims_prod>=64 && dims_prod<128)
      block_size=64;
    else
      block_size=128;  
  } else
    block_size = pre_block_size;

  grid_size = ceil_div(dims_prod, block_size);
  shared_mem_size = 2 * block_size / 32 * sizeof(float);

  std::vector<int> ret = {grid_size, block_size, shared_mem_size};
  return ret;
}

__global__ void vec_mult(const float a, float* x, float* y, int dims_prod) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < dims_prod) {
    y[idx] = x[idx] * a;
  }
}
__global__ void vec_div(const float a, float* x, float* y, int dims_prod) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < dims_prod) {
    y[idx] = x[idx] / a;
  }
}
__global__ void vec_reverse_div(const float a, float* x, float* y, int dims_prod) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < dims_prod) {
    y[idx] = a / x[idx];
  }
}
__global__ void vec_add(const float a, float* x, float* y, int dims_prod) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < dims_prod) {
    y[idx] = x[idx] + a;
  }
}
__global__ void vec_sub(const float a, float* x, float* y, int dims_prod) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < dims_prod) {
    y[idx] = x[idx] - a;
  }
}
__global__ void vec_equal(const float a, float* x, float* y, int dims_prod) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < dims_prod) {
    y[idx] = (x[idx]==a) ? 1.0f : 0.0f;
  }
}
__global__ void vec_diff(const float a, float* x, float* y, int dims_prod) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < dims_prod) {
    y[idx] = (x[idx]!=a) ? 1.0f : 0.0f;
  }
}
__global__ void vec_higher(const float a, float* x, float* y, int dims_prod) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < dims_prod) {
    y[idx] = (x[idx]>a) ? 1.0f : 0.0f;
  }
}
__global__ void vec_higher_eq(const float a, float* x, float* y, int dims_prod) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < dims_prod) {
    y[idx] = (x[idx]>=a) ? 1.0f : 0.0f;
  }
}
__global__ void vec_minor(const float a, float* x, float* y, int dims_prod) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < dims_prod) {
    y[idx] = (x[idx]<a) ? 1.0f : 0.0f;
  }
}
__global__ void vec_minor_eq(const float a, float* x, float* y, int dims_prod) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < dims_prod) {
    y[idx] = (x[idx]<=a) ? 1.0f : 0.0f;
  }
}
__global__ void vec_log(const float* x, float* y, int dims_prod) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < dims_prod) {
    y[idx] = logf(x[idx]);
  }
}
__global__ void vec_log2(const float* x, float* y, int dims_prod) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < dims_prod) {
    y[idx] = log2f(x[idx]);
  }
}

__global__ void tensor_div(float *w, float *x, float *y, int dims_prod) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < dims_prod) {
    y[idx] = x[idx] / w[idx];
  }
}

__global__ void tensor_clip(float* x, float *y, float _min, float _max, int dims_prod) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < dims_prod) {
    if (x[idx]>_max)
      y[idx] = _max;
    else if (x[idx]<_min)
      y[idx] = _min;
    else
      y[idx] = x[idx];
  }
}



extern "C" void *CudaScalarMult(Tensor tensor, float R) {
  //std::cout << "CudaScalarMult by " << R << "\n";
  
  int kDataLen = tensor.dims_prod;

  float *device_y;
  cudaCheck(cudaMalloc(&device_y, kDataLen * sizeof(float)));
  

  int grid_size = kDataLen;
  int block_size = 32;
  size_t shared_mem_size = 2 * block_size / 32 * sizeof(float);
  vec_mult<<<grid_size, block_size, shared_mem_size>>>(R, tensor.tensor_ptr, device_y, kDataLen);

  Tensor *new_tensor = createTensor(device_y, tensor.dims, kDataLen, false, "");
  
  return new_tensor;
}


extern "C" void *CudaScalarDiv(Tensor tensor, float R) {

  int kDataLen = tensor.dims_prod;


  float* device_y;
  cudaCheck(cudaMalloc(&device_y, kDataLen * sizeof(float)));


  int grid_size = kDataLen;
  int block_size = 32;
  size_t shared_mem_size = 2 * block_size / 32 * sizeof(float);
  vec_div<<<grid_size, block_size, shared_mem_size>>>(R, tensor.tensor_ptr, device_y, kDataLen);

  
  Tensor *new_tensor = createTensor(device_y, tensor.dims, kDataLen, false, "");
  return new_tensor;
}

extern "C" void *CudaReverseScalarDiv(Tensor tensor, float R) {

  int kDataLen = tensor.dims_prod;


  float* device_y;
  cudaCheck(cudaMalloc(&device_y, kDataLen * sizeof(float)));


  int grid_size = kDataLen;
  int block_size = 32;
  size_t shared_mem_size = 2 * block_size / 32 * sizeof(float);
  vec_reverse_div<<<grid_size, block_size, shared_mem_size>>>(R, tensor.tensor_ptr, device_y, kDataLen);

  Tensor *new_tensor = createTensor(device_y, tensor.dims, kDataLen, false, "");
  return new_tensor;
}

extern "C" void *CudaScalarAdd(Tensor tensor, float R) {
  
  int dims_prod = tensor.dims_prod;


  float* device_y;
  cudaCheck(cudaMalloc(&device_y, dims_prod * sizeof(float)));
  
  
  int grid_size = dims_prod;
  int block_size = 32;
  size_t shared_mem_size = 2 * block_size / 32 * sizeof(float);
  vec_add<<<grid_size, block_size, shared_mem_size>>>(R, tensor.tensor_ptr, device_y, dims_prod);
  
  Tensor *new_tensor = createTensor(device_y, tensor.dims, dims_prod, false, "");
  return new_tensor;
}

extern "C" void *CudaScalarSub(Tensor tensor, float R) {

  int kDataLen = tensor.dims_prod;


  float* device_y;
  cudaCheck(cudaMalloc(&device_y, kDataLen * sizeof(float)));


  int grid_size = kDataLen;
  int block_size = 32;
  size_t shared_mem_size = 2 * block_size / 32 * sizeof(float);
  vec_sub<<<grid_size, block_size, shared_mem_size>>>(R, tensor.tensor_ptr, device_y, kDataLen);

  Tensor *new_tensor = createTensor(device_y, tensor.dims, kDataLen, false, "");
  return new_tensor;
}

extern "C" void *CudaScalarEqual(Tensor tensor, float R) {

  int kDataLen = tensor.dims_prod;


  float* device_y;
  cudaCheck(cudaMalloc(&device_y, kDataLen * sizeof(float)));


  int grid_size = kDataLen;
  int block_size = 32;
  size_t shared_mem_size = 2 * block_size / 32 * sizeof(float);
  vec_equal<<<grid_size, block_size, shared_mem_size>>>(R, tensor.tensor_ptr, device_y, kDataLen);

  Tensor *new_tensor = createTensor(device_y, tensor.dims, kDataLen, false, "");
  return new_tensor;
}
extern "C" void *CudaScalarDiff(Tensor tensor, float R) {

  int kDataLen = tensor.dims_prod;


  float* device_y;
  cudaCheck(cudaMalloc(&device_y, kDataLen * sizeof(float)));


  int grid_size = kDataLen;
  int block_size = 32;
  size_t shared_mem_size = 2 * block_size / 32 * sizeof(float);
  vec_diff<<<grid_size, block_size, shared_mem_size>>>(R, tensor.tensor_ptr, device_y, kDataLen);

  Tensor *new_tensor = createTensor(device_y, tensor.dims, kDataLen, false, "");
  return new_tensor;
}
extern "C" void *CudaScalarMinor(Tensor tensor, float R) {

  int kDataLen = tensor.dims_prod;


  float* device_y;
  cudaCheck(cudaMalloc(&device_y, kDataLen * sizeof(float)));


  int grid_size = kDataLen;
  int block_size = 32;
  size_t shared_mem_size = 2 * block_size / 32 * sizeof(float);
  vec_minor<<<grid_size, block_size, shared_mem_size>>>(R, tensor.tensor_ptr, device_y, kDataLen);

  Tensor *new_tensor = createTensor(device_y, tensor.dims, kDataLen, false, "");
  return new_tensor;
}
extern "C" void *CudaScalarMinorEq(Tensor tensor, float R) {

  int kDataLen = tensor.dims_prod;


  float* device_y;
  cudaCheck(cudaMalloc(&device_y, kDataLen * sizeof(float)));


  int grid_size = kDataLen;
  int block_size = 32;
  size_t shared_mem_size = 2 * block_size / 32 * sizeof(float);
  vec_minor_eq<<<grid_size, block_size, shared_mem_size>>>(R, tensor.tensor_ptr, device_y, kDataLen);

  Tensor *new_tensor = createTensor(device_y, tensor.dims, kDataLen, false, "");
  return new_tensor;
}
extern "C" void *CudaScalarHigher(Tensor tensor, float R) {

  int kDataLen = tensor.dims_prod;


  float* device_y;
  cudaCheck(cudaMalloc(&device_y, kDataLen * sizeof(float)));


  int grid_size = kDataLen;
  int block_size = 32;
  size_t shared_mem_size = 2 * block_size / 32 * sizeof(float);
  vec_higher<<<grid_size, block_size, shared_mem_size>>>(R, tensor.tensor_ptr, device_y, kDataLen);

  Tensor *new_tensor = createTensor(device_y, tensor.dims, kDataLen, false, "");
  return new_tensor;
}
extern "C" void *CudaScalarHigherEq(Tensor tensor, float R) {

  int kDataLen = tensor.dims_prod;


  float* device_y;
  cudaCheck(cudaMalloc(&device_y, kDataLen * sizeof(float)));


  int grid_size = kDataLen;
  int block_size = 32;
  size_t shared_mem_size = 2 * block_size / 32 * sizeof(float);
  vec_higher_eq<<<grid_size, block_size, shared_mem_size>>>(R, tensor.tensor_ptr, device_y, kDataLen);

  Tensor *new_tensor = createTensor(device_y, tensor.dims, kDataLen, false, "");
  return new_tensor;
}


//TODONOW
extern "C" void *logE(Tensor tensor) {
  //std::cout << "logE of: " << tensor.name << "\n";

  float * device_x = tensor.tensor_ptr;
  std::vector<float> dims = tensor.dims;
  int kDataLen = tensor.dims_prod;

  float* device_y;
  cudaCheck(cudaMalloc(&device_y, kDataLen * sizeof(float)));

  int grid_size = kDataLen;
  int block_size = 32;
  size_t shared_mem_size = 2 * block_size / 32 * sizeof(float);
  vec_log<<<grid_size, block_size, shared_mem_size>>>(device_x, device_y, kDataLen);

  Tensor *new_tensor = createTensor(device_y, dims, kDataLen, false, "");
  return new_tensor;
}

extern "C" void *logE2(Tensor tensor) {
  std::cout << "logE2 of: " << tensor.name << "\n";

  float * device_x = tensor.tensor_ptr;
  std::vector<float> dims = tensor.dims;
  int kDataLen = tensor.dims_prod;

  float* device_y;
  cudaCheck(cudaMalloc(&device_y, kDataLen * sizeof(float)));

  int grid_size = kDataLen;
  int block_size = 32;
  size_t shared_mem_size = 2 * block_size / 32 * sizeof(float);
  vec_log2<<<grid_size, block_size, shared_mem_size>>>(device_x, device_y, kDataLen);

  Tensor *new_tensor = createTensor(device_y, dims, kDataLen, false, "");
  return new_tensor;
}


extern "C" float CalculateIdxOffset(char *tensor_name, float first_idx, ...) {
  
  //std::cout << "CalculateIdxOffset of " << tensor_name << "\n";

  Tensor *tensor = NamedTensorsT[tensor_name];

  std::vector<float> idxs, new_dims_no_minus, dims;
  int current_dims_prod;
  bool has_minus = false;
  dims = tensor->dims;

  int idx_at = 0;

  
  va_list args;
  va_start(args, first_idx);

  if (first_idx!=-1)
    new_dims_no_minus.push_back(first_idx);
  else
    has_minus=true;
  
    
  idxs.push_back(first_idx);

  dims = RemoveFirstDim(dims);
  
  current_dims_prod = DimsProd(dims);

  idx_at += (int)(current_dims_prod*first_idx);



  //std::cout << "Get idx of " << tensor_name << "\nCalculateIdxOffset pushing dim: " << first_idx << "\n";

  for (int i=0; i<10; i++)
  {
    if (i==9)
    {
      LogErrorS("A tensor with 10 dimensions???");
      return 0;
    }

    float idx = va_arg(args, float);
    if (idx==-40370000000)
      break;

    idxs.push_back(idx);
    
    dims = RemoveFirstDim(dims);
    
    current_dims_prod = DimsProd(dims);

    idx_at += (int)(current_dims_prod*idx);

    //std::cout << "CalculateIdxOffset pushing dim: " << idx << "\n";
    

    if (idx!=-1)
      new_dims_no_minus.push_back(idx);
    else
      has_minus=true;
  }
  va_end(args);



  return idx_at;
}


extern "C" void AttrPinnedOnIdx(char *tensor_name, float val, float idx_at) {
  Tensor *tensor = NamedTensorsT[tensor_name];
  //std::cout << "AttrPinnedOnIdx for " << tensor->name << " at index " << idx_at << "\n";
  //std::cout << "Value: " << val <<"\n";

  std::vector<float> dims = tensor->dims;
  int dims_prod = DimsProd(dims);
  if (idx_at>(dims_prod-1))
  {
    std::string _error = "\n\t- Idexating at pos: \033[32m"+std::to_string((int)idx_at);
    _error = _error + "\033[0m on pinned_tensor \033[95m"+std::string(tensor_name);
    _error = _error + "\033[0m;\n\t- Max idx allowed:  \033[32m"+std::to_string(dims_prod)+"\033[0m.";

    LogErrorS(_error);
    std::cout << "Dimensions:" << "\n";
    PrintDims(dims);
    std::cout << "\n";
  }

  float *base_address = tensor->cpu_tensor_ptr;
  
  
  //std::cout << "idx " << idx_at << ", val " << val << "\n";

  float *device_x = base_address + static_cast<int>(idx_at);

  *device_x = val;
}


extern "C" char * FirstArgOnDemand(char *first_arg, char *pre_dotc, char *_class, char *method, int nested_function, int isSelf, int isAttribute)
{

  std::string _first_arg = first_arg;
  std::string pre_dot = pre_dotc;

  //std::cout << "\n\n\nIncoming first arg: " << first_arg << " from pre-dot: " << pre_dot << ";\n   class: " << _class << ", method: " << method << "\n   is nested: " << nested_function <<".\n";
  //std::cout << "   is self: " << isSelf << ", is attribute: " << isAttribute << "\n\n\n";

  if (!isSelf && isAttribute)
  {
    std::string ret = NamedObjects[pre_dot];
    return str_to_char(ret);
    //return const_cast<char*>(ret.c_str());
  }
  
  if (pre_dot!="self")
  {
    if (nested_function)
      _first_arg = _first_arg+pre_dot;
    else
      _first_arg = pre_dot; 
  }

  return const_cast<char*>(_first_arg.c_str());
}


extern "C" void InstantiateObject(char *scope, char *obj_name)
{
  //std::cout << "\n\nInstantiateObject of: " << scope << obj_name << "\n";
  std::string _obj_name = obj_name;

  NamedObjects[scope+_obj_name] = _obj_name + RandomString(13);
  //std::cout << "Saving " << NamedObjects[scope+_obj_name]  << "\n\n";
}


extern "C" char *objHash(char *scope, char *obj_name)
{
  std::string _obj_name = obj_name;
  std::string ret = NamedObjects[scope+_obj_name];
  return str_to_char(ret);
}


extern "C" char *LoadObject(char *obj_name)
{
  //std::cout << "LOADING OBJECT FROM " << obj_name << "\n";
  std::string ret = NamedObjects[obj_name];
  delete[] obj_name;
  //std::cout << "Load object of: " << ret << "\n";
  return str_to_char(ret);
}

extern "C" char *GetEmptyChar()
{
  std::string x = "";
  return str_to_char(x);
}

extern "C" void FreeCharFromFunc(char *_char, char *func) {
  std::cout << "FREEING " << _char << " at function: " << func << "\n";
  delete[] _char;
  std::cout << "freed" << "\n";
}


extern "C" void FreeChar(char *_char) {
  //std::cout << "FREEING " << _char << "\n";
  delete[] _char;
}


extern "C" char *CopyString(char *in_str)
{
  char* copied = (char*)malloc(strlen(in_str) + 1);
  strcpy(copied, in_str);

  return copied;
}

extern "C" char * ConcatStr(char *lc, char *rc)
{
  //std::cout << "Concat strings " << lc << " & " << rc << "\n";
  std::string l = lc;
  std::string r = rc;

  std::string result_str = l + r;
  char* result_cstr = new char[result_str.length() + 1]; // +1 for null terminator
  std::strcpy(result_cstr, result_str.c_str());

  
  return result_cstr;
}

extern "C" char * ConcatStrFreeLeft(char *lc, char *rc)
{
  std::string l = lc;
  std::string r = rc;

  std::string result_str = l + r;
  char* result_cstr = new char[result_str.length() + 1]; // +1 for null terminator
  std::strcpy(result_cstr, result_str.c_str());

  delete[] lc;
  
  return result_cstr;
}

extern "C" char * ConcatStrFreeRight(char *lc, char *rc)
{
  std::string l = lc;
  std::string r = rc;

  std::string result_str = l + r;
  char* result_cstr = new char[result_str.length() + 1]; // +1 for null terminator
  std::strcpy(result_cstr, result_str.c_str());

  delete[] rc;
  
  return result_cstr;
}

extern "C" char * ConcatStrFree(char *lc, char *rc)
{
  std::string l = lc;
  std::string r = rc;

  std::string result_str = l + r;
  char* result_cstr = new char[result_str.length() + 1]; // +1 for null terminator
  std::strcpy(result_cstr, result_str.c_str());

  delete[] lc, rc;
  return result_cstr;
}


extern "C" char * ConcatNumToStr(char *lc, float r)
{
  //std::cout << "\nCONCAT NUM TO STR " << lc << " & " << std::to_string(r) << "\n";

  std::string l = lc;
  int _r = r;

  std::string result_str = l + std::to_string(_r);
  char* result_cstr = new char[result_str.length() + 1]; // +1 for null terminator
  std::strcpy(result_cstr, result_str.c_str());

  //std::cout << "Concatenated into " << result_cstr << "\n\n";
  
  return result_cstr;
}

extern "C" char * ConcatNumToStrFree(char *lc, float r)
{
  //std::cout << "\nCONCAT NUM TO STR " << lc << " & " << std::to_string(r) << "\n";

  std::string l = lc;
  int _r = r;

  std::string result_str = l + std::to_string(_r);
  char* result_cstr = new char[result_str.length() + 1]; // +1 for null terminator
  std::strcpy(result_cstr, result_str.c_str());

  delete[] lc;
  
  return result_cstr;
}

extern "C" void AddToScopeCleanList(char *scope, char *name, char *type)
{
  
  std::vector<std::pair<std::string, std::string>> scope_vars = ScopeVarsToClean[scope];
  
  for(auto &pair : scope_vars)
    if (pair.first==name)
    {
      delete[] name;
      return;
    }
      

  ScopeVarsToClean[scope].push_back(std::make_pair(name, type));
  
  delete[] name;
}
extern "C" void CleanScopeVars(char *scope)
{
  //TODO: this leads to segmentation fault with 3+ workers. 
  /*
  std::vector<std::pair<std::string, std::string>> scope_vars = ScopeVarsToClean[scope];

  for(auto &pair : scope_vars)
  {
    if (pair.second=="str")
    {
      auto it = NamedStrs.find(pair.first);
      if (it != NamedStrs.end()) {
        delete[] it->second;
        NamedStrs.erase(it);
      }
    }
  }
  */

  //delete[] scope;
}


extern "C" char * RandomStrOnDemand()
{

  std::string random_str = RandomString(14);

  char* result_cstr = new char[random_str.length() + 1]; // +1 for null terminator
  std::strcpy(result_cstr, random_str.c_str());

  
  return result_cstr;
}


extern "C" void StoreOnDemand(char *name, float value){
  
  NamedClassValues[name] = value;
  delete[] name;
}
extern "C" void StoreOnDemandNoFree(char *name, float value){
  
  NamedClassValues[name] = value;
}


extern "C" void StoreArgOnDemand(char *name, float value){
  //std::cout << "StoreArgOnDemand: " << name  << " " << value << "\n";
  NamedClassValues[name] = value;
}

extern "C" float StoreStrOnDemand(char *name, char * value){
  
  NamedStrs[name] = CopyString(value);
  delete[] name;

  return 0;
}
extern "C" void *LoadStrOnDemand(char *name){
  
  char *ret = CopyString(NamedStrs[name]);
  delete[] name;

  return ret;
}


extern "C" float StoreStrVecOnDemand(char *name, std::vector<char *> value){
  //std::cout << "STORING " << self << "." << object_var_name << " on demand as StrVec type.\n";
  ClassStrVecs[name] = value;
  delete[] name;
  return 0;
}

extern "C" float StoreFloatVecOnDemand(char *name, std::vector<float> value){
  //std::cout << "STORING " << self << "." << object_var_name << " on demand as float vec type" << ".\n";

  ClassFloatVecs[name] = value;
  delete[] name;
  return 0;
}

extern "C" float StoreFloatVecOnDemandOnIdx(char *name, float idx, float value){
  //std::cout << "STORING " << self << "." << object_var_name << " on demand as float vec type" << ".\n";

  ClassFloatVecs[name][(int)idx] = value;
  delete[] name;
  return 0;
}




extern "C" float LoadOnDemand(char *object_var_name) {
  //std::cout << "Load on demand for: " << object_var_name << "\n";
  //std::cout << "Value: " << NamedClassValues[object_var_name] << "\n";

  float ret = NamedClassValues[object_var_name];
  delete[] object_var_name;
  return ret;
}

extern "C" float LoadOnDemandNoFree(char *object_var_name) {
  //std::cout << "Load on demand for: " << object_var_name << "\n";
  //std::cout << "Value: " << NamedClassValues[object_var_name] << "\n";

  float ret = NamedClassValues[object_var_name];
  
  return ret;
}




extern "C" void LockMutex(char *mutex_name)
{
  pthread_mutex_t *_mutex = lockVars[mutex_name];
  pthread_mutex_lock(_mutex);
}

extern "C" void UnlockMutex(char *mutex_name)
{
  pthread_mutex_t *_mutex = lockVars[mutex_name];
  pthread_mutex_unlock(_mutex);
}

extern "C" void *LoadStrVecOnDemand(char *object_var_name) {
  //std::cout << "Load StrVec On Demand var to load: " << object_var_name << "\n";
  
  void *ret = &ClassStrVecs[object_var_name];
  delete[] object_var_name;
  return ret;
}


extern "C" void *LoadFloatVecOnDemand(char *object_var_name) {
  std::cout << "Load StrVec On Demand var to load: " << object_var_name << "\n";
  
  void *ret = &ClassFloatVecs[object_var_name];
  delete[] object_var_name;
  return ret;
}


Value *ObjAttrExprAST::codegen(Value *ignored_first_arg, Value *scope_str, Value *previous_scope) {
  /*
  if (not ShallCodegen)
    return ConstantFP::get(*TheContext, APFloat(0.0f));

  Value *first_arg, *ret;

  first_arg = Builder->CreateAlloca(int8PtrTy);
  Builder->CreateStore(Builder->CreateGlobalString(PreDot), first_arg);


  ret = Body->codegen(first_arg, scope_str, previous_scope);
  */
  return ConstantFP::get(*TheContext, APFloat(0.0f));
}



bool seen_var_attr = false;
Value *VariableExprAST::codegen(Value *first_arg, Value *scope_str, Value *previous_scope) {
  if (not ShallCodegen)
    return ConstantFP::get(*TheContext, APFloat(0.0f));
  // Look this variable up in the function.


  

  Function *TheFunction = Builder->GetInsertBlock()->getParent();
  std::string functionName = TheFunction->getName().str();
  
  std::cout << "Create value V" << "\n";
  Value * ret = ConstantFP::get(*TheContext, APFloat(0.0f));
  Value *V;


  Value *var_name, *object_name, *object_var_name;
  


  /*
  std::cout << "\nVARIABLE EXPR CODEGEN: " << Name << "\n";
  for (const auto &entry : NamedStrVecs)
    std::cout << "NamedStrVec: " << entry.first << "\n";
  for (const auto &entry : NamedValues)
    std::cout << "NamedValues: " << entry.first << "\n";
  for (const auto &entry : NamedClassValues)
    std::cout << "NamedClassValues: " << entry.first << "\n";
  */

  std::string type = GetType();
  std::string pre_dot = GetPreDot();
  bool is_self = GetSelf();
  bool is_attr = GetIsAttribute();
  
  //std::string __print = "\n\nLOAD OF " + std::string(Name) + " ";
  //Builder->CreateCall(TheModule->getFunction("print"),
  //    {Builder->CreateGlobalString(__print), ConstantFP::get(*TheContext, APFloat(0.0f))});

  

  var_name = NameSolver->codegen(first_arg, scope_str, previous_scope);

  NameSolverAST *name_solver = static_cast<NameSolverAST *>(NameSolver.get());
  std::string Name = std::get<0>(name_solver->Names[name_solver->Names.size()-1]);
  

  if (is_self||is_attr)
  {
    if (type=="float")
    {
        V = Builder->CreateCall(TheModule->getFunction("LoadOnDemand"), {var_name});
        V = Builder->CreateCall(TheModule->getFunction("UnbugFloat"), {V}, "unbugfloat");
        return V;
      
    }
    if (type=="str_vec")
    {
      V = Builder->CreateCall(TheModule->getFunction("LoadStrVecOnDemand"),
                                                      {var_name});
      //Builder->CreateCall(TheModule->getFunction("PrintStrVec"), {V});
      return V;
      
    }
    if (type=="float_vec"){
      V = Builder->CreateCall(TheModule->getFunction("LoadFloatVecOnDemand"),
                                                      {var_name});
      Builder->CreateCall(TheModule->getFunction("PrintFloatVec"), {V});
      return V;
    }
  }



  if (type=="float")
  {
    std::cout << "\nVariable Float " << Name << " codegen.\n";
    
    V = Builder->CreateCall(TheModule->getFunction("LoadOnDemand"),{var_name});
    

    //if (!seen_var_attr) //TODO: Solve this bug
    //  V = Builder->CreateCall(TheModule->getFunction("UnbugFloat"), {V}, "unbugfloat");

    return V;

  } else if (type=="object") {
    return var_name;

  } else if (type=="str") {

    
    //std::cout << "Type: " << Type << "\n\n";

    for (const auto &entry : NamedTensorsT)
    {
      std::cout << "Returning None because a tensor with name " << Name << " was found on strings map " << "\n";
      if (ends_with(entry.first, Name))
        return ret;
    }
    

    V = Builder->CreateCall(TheModule->getFunction("LoadStrOnDemand"), {var_name});

    //if (!seen_var_attr)
    //  Builder->CreateCall(TheModule->getFunction("PrintStr"), {V});
    
    return V;
  } else if (type=="str_vec") {

    //std::cout << "\nVariable Str Vector " << Name << " Codegen. \nNamedStrVecs.count(Name): " << NamedStrVecs.count(Name) <<"\n\n";



    V = NamedStrVecs[Name];
    
    V = Builder->CreateLoad(int8PtrTy, V, Name.c_str());
    if (!seen_var_attr)
      Builder->CreateCall(TheModule->getFunction("PrintStrVec"), {V});

    return V;
  } else if (NamedPinnedTensors.count(Name)>0) {
    //std::cout << "\nVariable Tensor " << Name << " Codegen.\n";
  

    if (!seen_var_attr)
      Builder->CreateCall(TheModule->getFunction("PrintTensor"), {var_name});
    
    
    //Builder->CreateCall(TheModule->getFunction("PrintTensor"), {var_name});

    return Builder->CreateCall(TheModule->getFunction("LoadTensor"), {var_name});
  } else if (type=="tensor") {
    //std::cout << "\nVariable Tensor " << Name << " Codegen.\n";


    if (!seen_var_attr)
      Builder->CreateCall(TheModule->getFunction("PrintTensor"), {var_name});
    
    return Builder->CreateCall(TheModule->getFunction("LoadTensor"), {var_name});
  }
  else
  {
    std::string _error = "Variable " + Name + " does not exist";
    LogErrorS(_error);
  }
}






extern "C" char * AuxFn(char * arg1)
{

  std::cout << "Aux fn: " << arg1 << "\n";
  

  return arg1;
}





Value *VecIdxExprAST::codegen(Value *first_arg, Value *scope_str, Value *previous_scope) {
  if (not ShallCodegen)
    return ConstantFP::get(*TheContext, APFloat(0.0f));
  // Look this variable up in the function.

  std::cout << "Now Loading Vec indexation for type: " << Type << "  \n";


  Function *TheFunction = Builder->GetInsertBlock()->getParent();
  std::string functionName = TheFunction->getName().str();
  
  
  Value * ret = ConstantFP::get(*TheContext, APFloat(0.0f));
  Value *V, *idx;

  if (Type!="object_vec")
    idx = Idx[0]->codegen(first_arg, scope_str, previous_scope);


  Value *var_name, *object_name, *object_var_name;
  var_name = NameSolver->codegen(first_arg, scope_str, previous_scope);
  
  NameSolverAST *name_solver = static_cast<NameSolverAST *>(NameSolver.get());
  std::string Name = std::get<0>(name_solver->Names[0]);
  


  std::string pre_dot = GetPreDot();
  bool is_self = GetSelf();
  bool is_attr = GetIsAttribute();
  std::cout << "is self: " << is_self << ", is_attr: " << is_attr << "\n";

  if (is_self||is_attr)
  {
    
    if (Type=="str_vec"){
      
      V = Builder->CreateCall(TheModule->getFunction("IndexClassStrVec"), {var_name, idx});
      
      return V;
    }

    if (Type=="float_vec"){
      V = Builder->CreateCall(TheModule->getFunction("IndexClassFloatVec"), {var_name, idx});
      return V;
    }

    if (Type=="object_vec")
      return var_name;
  }


  if (Type=="str_vec")
  {
    V = NamedStrVecs[Name];
    V = Builder->CreateLoad(int8PtrTy, V, Name.c_str());


    V = Builder->CreateCall(TheModule->getFunction("IndexStrVec"), {V, idx});

    return V;
  }
  if (Type=="float_vec")
  {
    V = NamedFloatVecs[Name];
    V = Builder->CreateLoad(int8PtrTy, V, Name.c_str());


    V = Builder->CreateCall(TheModule->getFunction("IndexStrVec"), {V, idx});

    return V;
  }

  if (Type=="tensor")
  {
    std::cout << "vec idx of tensor, idx type: " << Idx[0]->GetType() << "\n";

    if (Idx[0]->GetType()!="tensor")
    {
      std::vector<Value *> idx_calc_args;
      idx_calc_args.push_back(var_name);
      for (int i=0; i<Idx.size(); i++)
        idx_calc_args.push_back(Idx[i]->codegen(first_arg, scope_str, previous_scope));
      Value *idx_at = Builder->CreateCall(TheModule->getFunction("CalculateIdxOffset"),
                          idx_calc_args);

      return Builder->CreateCall(TheModule->getFunction("IdxTensor"), {var_name, idx_at});
    } else {
      VariableExprAST *idx = static_cast<VariableExprAST *>(Idx[0].get());
      Value *idx_tensor_name = idx->NameSolver->codegen(first_arg, scope_str, previous_scope);
      
      return Builder->CreateCall(TheModule->getFunction("IdxTensorWithTensor"), {var_name, idx_tensor_name});
      
    }
    
  }

  std::string _error = "Unknown vector: " + Name + ".";
  LogErrorS(_error);
  std::cout << "Type " << Type << "\n";

  return ret;
}


Value *ObjectVecIdxExprAST::codegen(Value *first_arg, Value *scope_str, Value *previous_scope) {
  if (not ShallCodegen)
    return ConstantFP::get(*TheContext, APFloat(0.0f));
  // Look this variable up in the function.
  std::cout << "ObjectVecIdxExprAST codegen" << "\n";
  
  VecIdxExprAST *vec = static_cast<VecIdxExprAST *>(Vec.get());
  std::cout << "vec name " << vec->GetName() << "\n";
  std::cout << "ObjectVecIdxExprAST is vec: " << GetIsVec() << "\n";

  Value *idx = vec->Idx[0]->codegen(first_arg, scope_str, previous_scope);


  Value *var_name, *object_name, *object_var_name, *post_dot_str;
  var_name = Builder->CreateGlobalString(vec->GetName());
  post_dot_str = Builder->CreateGlobalString(_post_dot);
  
  std::string pre_dot = GetPreDot();
  bool is_self = GetSelf();
  bool is_attr = GetIsAttribute();
  
  
  if (is_self||is_attr)
  {
    // Gets from pre_dot if it is a class attribute
    if (is_attr) {
      object_name = Builder->CreateGlobalString(pre_dot);
      var_name = Builder->CreateGlobalString(Name);

      var_name = Builder->CreateCall(TheModule->getFunction("ConcatStr"),
                                                      {object_name, var_name});
    }
    if (is_self)
      var_name = Builder->CreateCall(TheModule->getFunction("ConcatStr"),
                                                      {Builder->CreateLoad(int8PtrTy, first_arg), var_name});
  }

  if (Type=="tensor")
    return Builder->CreateCall(TheModule->getFunction("object_vec_idxTensor"),
                                                      {var_name, idx, post_dot_str});
  if (Type=="object")
    return Builder->CreateCall(TheModule->getFunction("object_vec_idxObject"),
                                                      {var_name, idx, post_dot_str});


  return ConstantFP::get(*TheContext, APFloat(0.0f));
}



__global__ void repeat_interleave_kernel_last_dim(const float *tensor,
                           float *probs,
                           int B, int C) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    //int i = threadIdx.x;
    
    if (i < B * C) {
        int b = i / (C);
        int v = i % C;

        float *probs_b = probs + b * C;
        float ix = tensor[b];

        probs_b[v] = ix;
    }
}




extern "C" void *IdxTensor(char *tensor_name, float idx_at)
{
  
  //std::cout << "\n\n\nIDX " << tensor_name << "\n\n\n\n";  
  
  
  Tensor *tensor = NamedTensorsT[tensor_name];


  float *new_tensor;

  std::vector<float> dims = tensor->dims;
  std::vector<float> new_dims;

  if (dims.size()==1)
    new_dims = {1.0f};
  else
    for (int i = 0; i < dims.size()-1; i++)
      new_dims.push_back(dims[i+1]);
  int new_dims_prod = DimsProd(new_dims);

  int dims_prod = DimsProd(dims);
  if (idx_at>(dims_prod-1))
  {
    std::string _error = "\n\t- Idexating at pos: \033[32m"+std::to_string((int)idx_at);
    _error = _error + "\033[0m on tensor \033[95m"+std::string(tensor_name);
    _error = _error + "\033[0m;\n\t- Max idx allowed:  \033[32m"+std::to_string(dims_prod)+"\033[0m.";

    LogErrorS(_error);
    std::cout << "Dimensions:" << "\n";
    PrintDims(dims);
    std::cout << "\n";

    return nullptr;
  }
  //std::cout << "IDX AT: " << idx_at << "\n";


  float *base_address = tensor->tensor_ptr;
  float *device_x = base_address + static_cast<int>(idx_at);


  cudaMalloc(&new_tensor, new_dims_prod*sizeof(float));
  cudaCheck(cudaMemcpy(new_tensor, device_x, new_dims_prod*sizeof(float), cudaMemcpyHostToHost));

  /*
  PrintTensorF(new_tensor, 1, 1);
  PrintDims(new_dims);
  std::cout << "dims prod:" << new_dims_prod  << "\n";
  */

  Tensor *indexed = createTensor(new_tensor, new_dims, new_dims_prod, false, "");
  return indexed;
}


__global__ void idx_last_dim_kernel(float *tgt,
                           const float *tensor, const float *idx_tensor, 
                           int dims_prod, int last_dim_size) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int C = last_dim_size;
    
    if (i < dims_prod) {
        int b = i / C;
        int v = i % C;
        // i = b * C + v

        float *tgt_b = tgt + b;
        float idx_b = idx_tensor[b];

        if (v==idx_b)
        {
          float ix = tensor[i];
          tgt[b] = ix;
        }
    }
}

extern "C" void *IdxTensorWithTensor(char *tensor_name, char *idx_tensor_name)
{
  std::cout << "Idx tensor " << tensor_name << " with tensor " << idx_tensor_name << "\n";

  //std::cout << "\n\n\nIDX " << tensor_name << "\n\n\n\n";  
  
  
  Tensor *tensor = NamedTensorsT[tensor_name];
  Tensor *idx_tensor = NamedTensorsT[idx_tensor_name];


  float *tensor_ptr, *idx_tensor_ptr, *new_tensor;
  float new_dims_prod;
  std::vector<float> dims, idx_dims, new_dims;

  tensor_ptr = tensor->tensor_ptr;
  idx_tensor_ptr = idx_tensor->tensor_ptr;

  dims = tensor->dims;
  idx_dims = idx_tensor->dims;


  //TODO: gather with smaller dimensions
  /*
  if (dims.size()==1)
    new_dims = {1.0f};
  else
    for (int i = 0; i < dims.size()-1; i++)
      new_dims.push_back(dims[i+1]);
  */
  

  std::cout << "dim size diff: " << dims.size()-idx_dims.size()  << "\n";
  if((dims.size()-idx_dims.size())==1)
  {
    new_dims_prod = idx_tensor->dims_prod;
    new_dims = idx_tensor->dims;

    std::cout << "INDEX OVER LAST DIM" << "\n";

    cudaMalloc(&new_tensor, new_dims_prod*sizeof(float));
    cudaMemset(new_tensor, 0, new_dims_prod*sizeof(float));
    
    int grid_size = tensor->dims_prod;
    int block_size = 32;
    size_t shared_mem_size = 2 * block_size / 32 * sizeof(float);
    idx_last_dim_kernel<<<grid_size, block_size, shared_mem_size>>>(new_tensor, tensor_ptr, idx_tensor_ptr, tensor->dims_prod, tensor->dims_prod/idx_tensor->dims_prod);
  }

  
  //cudaCheck(cudaMemcpy(new_tensor, device_x, new_dims_prod*sizeof(float), cudaMemcpyHostToHost));


  Tensor *indexed = createTensor(new_tensor, new_dims, new_dims_prod, false, "");
  return indexed;
}



extern "C" float CopyArgTensor(Tensor *tensor, char *new_tensor_name, char *previous_scope, char *scope)
{
  std::string tensor_name = tensor->name;
  //std::cout << "\n\n\nCOPY ARG TENSOR OF " << previous_scope << tensor_name << " into " << scope<<new_tensor_name <<"\n";
  
  std::string arg_tensor_name = scope;
  arg_tensor_name = arg_tensor_name + new_tensor_name;
  

  std::vector<float> dims = tensor->dims;
  int dims_prod = tensor->dims_prod;

  float *arg_tensor, *tensor_ptr;

  tensor_ptr = tensor->tensor_ptr;

  cudaMalloc(&arg_tensor, dims_prod*sizeof(float));
  cudaMemcpy(arg_tensor, tensor_ptr, dims_prod*sizeof(float), cudaMemcpyHostToHost);
  cudaCheck(cudaGetLastError());

  Tensor *new_tensor = createTensor(arg_tensor, dims, DimsProd(dims), true, arg_tensor_name);
  new_tensor->scopeless_name = tensor->scopeless_name;
  NamedTensorsT[arg_tensor_name] = new_tensor;
  return 0;
}


extern "C" float RemoveTensorScope(char *tensor_name, char *scope, char *tgt_tensorc, char *previous_scope)
{
  std::string scope_tensor_name = scope;
  scope_tensor_name = scope_tensor_name + tensor_name;

  std::string tgt_tensor = tgt_tensorc;
  tgt_tensor = previous_scope + tgt_tensor;

  //std::cout << "\n\n\nRETURNING " << scope_tensor_name << " into " << tgt_tensor << "\n\n\n\n";

  
  Tensor *tensor, *scope_tensor;
  tensor = NamedTensorsT[tgt_tensor];

  scope_tensor = NamedTensorsT[scope_tensor_name];
  std::vector<float> dims = scope_tensor->dims;
  int dims_prod = scope_tensor->dims_prod;


  cudaCheck(cudaFree(tensor->tensor_ptr));
  tensor->AttrTensor(scope_tensor->tensor_ptr, scope_tensor->dims, scope_tensor->dims_prod);
  
  NamedTensorsT[tgt_tensor] = tensor;

  delete scope_tensor; //TODO: check if this is not leading to out of memory
  NamedTensorsT.erase(scope_tensor_name);
  return 0;
}


extern "C" float RemoveTensorScopeAttrOnIndex(char *tensor_name, char *scope, char *tgt_tensorc, char *previous_scope, float idx_at)
{
  std::string scope_tensor_name = scope;
  scope_tensor_name = scope_tensor_name + tensor_name;

  std::string tgt_tensor = tgt_tensorc;
  tgt_tensor = previous_scope + tgt_tensor;


  std::cout << "\n\n\nRETURNING " << scope_tensor_name << " into " << tgt_tensor << " at idx\n\n\n\n";  



  Tensor *tensor, *scope_tensor;
  tensor = NamedTensorsT[tgt_tensor];

  scope_tensor = NamedTensorsT[scope_tensor_name];
  std::vector<float> dims = tensor->dims;
  int dims_prod = tensor->dims_prod;

  int scope_dims_prod = scope_tensor->dims_prod;

  
  if (idx_at>(dims_prod-1))
  {
    std::string _error = "\n\t- Idexating at pos: \033[32m"+std::to_string((int)idx_at);
    _error = _error + "\033[0m on tensor \033[95m"+std::string(tgt_tensor);
    _error = _error + "\033[0m;\n\t- Max idx allowed:  \033[32m"+std::to_string(dims_prod)+"\033[0m.";

    LogErrorS(_error);
    std::cout << "Dimensions:" << "\n";
    PrintDims(dims);
    std::cout << "\n";

    return -1;
  }

  if ((idx_at+scope_dims_prod)>(dims_prod))
  {
    std::string _error = "\n\t- Attributing at pos: \033[32m"+std::to_string((int)idx_at)+"\033[0m with a tensor of size \033[32m"+std::to_string(scope_dims_prod)+"\033[0m";
    _error = _error + "\033[0m on tensor \033[95m"+std::string(tgt_tensor);
    _error = _error + "\033[0m;\n\t- Max idx allowed:  \033[32m"+std::to_string(dims_prod)+"\033[0m.";

    LogErrorS(_error);
    std::cout << "Dimensions:" << "\n";
    PrintDims(dims);
    std::cout << "\n";

    return -1;
  }

  float *base_address = tensor->tensor_ptr;
  float *device_x = base_address + static_cast<int>(idx_at);

  cudaCheck(cudaMemcpy(device_x, scope_tensor->tensor_ptr, scope_dims_prod*sizeof(float), cudaMemcpyHostToHost));
  
  return 0;
}



extern "C" float AttrTensor(char *tensor_name, Tensor *tensor)
{
  //std::cout << "Attributing to tensor: " << tensor_name << " from " << tensor->name << "\n\n";

  Tensor *tgt_tensor = NamedTensorsT[tensor_name];
  
  
  
  if (tensor->view_of == tensor_name)
  {
    tgt_tensor->dims = tensor->dims;
    delete tensor;
  }
  else if (tensor->name=="") // Free current and point to operation result
  {
    //std::cout << "attributing op" << "\n";
    if(nn_mode==eval_mode)
    {
      cudaCheck(cudaFree(tgt_tensor->tensor_ptr));
      tgt_tensor->AttrTensor(tensor->tensor_ptr, tensor->dims, tensor->dims_prod);
      delete tensor;
    } else {
      Tensor *attr_tensor;
      attr_tensor = createBackward(tgt_tensor->scopeless_name, tensor);
      todo_backward_tensors.push_back(attr_tensor);

      std::string scopeless_name = tgt_tensor->scopeless_name;
      tgt_tensor = createTensor(tensor->tensor_ptr, tensor->dims, tensor->dims_prod, true, tensor_name);
      tgt_tensor->scopeless_name = scopeless_name;
    } 
  } else { // Copy incoming tensor to preserve it
    if(tensor->op==tensor_leaf||nn_mode==eval_mode)
    {
      if(tgt_tensor->dims != tensor->dims)
      {
        cudaCheck(cudaFree(tgt_tensor->tensor_ptr));
        cudaMalloc(&tgt_tensor->tensor_ptr, tensor->dims_prod*sizeof(float));

        tgt_tensor->dims = tensor->dims;
        tgt_tensor->dims_prod = tensor->dims_prod;
      }
      cudaCheck(cudaMemcpy(tgt_tensor->tensor_ptr, tensor->tensor_ptr, tgt_tensor->dims_prod*sizeof(float), cudaMemcpyDeviceToDevice));
      if(!tensor->leaf)
        delete tensor;
    } else {
      Tensor *attr_tensor;
      attr_tensor = createBackward(tgt_tensor->scopeless_name, tensor);
      todo_backward_tensors.push_back(attr_tensor);

      std::string scopeless_name = tgt_tensor->scopeless_name;
      tgt_tensor = createTensor(tensor->tensor_ptr, tensor->dims, tensor->dims_prod, true, tensor_name);
      tgt_tensor->scopeless_name = scopeless_name;
    }
    //OP: copy dy
  }
  

  NamedTensorsT[tensor_name] = tgt_tensor;
  return 0;
}


// Copies a pinned_tensor's reserved memory into a tensor.
extern "C" float AttrTensorNoFree(char *tensor_name, Tensor *tensor)
{
  //std::cout << "\nAttrTensorNoFree -- Attributing to tensor: " << tensor_name << "\n\n";
  

  std::vector<float> new_dims = tensor->dims;
  float dims_prod = tensor->dims_prod;

  cudaCheck(cudaGetLastError());


  float *new_tensor;
  cudaMalloc(&new_tensor, dims_prod*sizeof(float));
  cudaCheck(cudaMemcpy(new_tensor, tensor->tensor_ptr, dims_prod*sizeof(float), cudaMemcpyDeviceToDevice));

  Tensor *tgt_tensor = NamedTensorsT[tensor_name];
  cudaCheck(cudaFree(tgt_tensor->tensor_ptr));
  tgt_tensor->AttrTensor(new_tensor, new_dims, dims_prod);
  

  delete tensor;

  return 0;
}



extern "C" float AttrTensorOnIdx(char *tensor_name, Tensor *tensor, float idx_at)
{ 
  //std::cout << "AttrTensorOnIdx of" << tensor_name << " at idx " << idx_at << "\n";

  std::vector<float> dims, Rdims;
  Tensor *tgt_tensor = NamedTensorsT[tensor_name];
  dims = tgt_tensor->dims;
  int dims_prod = tgt_tensor->dims_prod;

  Rdims = tensor->dims;

  if (idx_at>(dims_prod-1))
  {
    std::string _error = "\n\t- Idexating at pos: \033[32m"+std::to_string((int)idx_at);
    _error = _error + "\033[0m on tensor \033[95m"+std::string(tensor_name);
    _error = _error + "\033[0m;\n\t- Max idx allowed:  \033[32m"+std::to_string(dims_prod)+"\033[0m.";

    LogErrorS(_error);
    std::cout << "Dimensions:" << "\n";
    PrintDims(dims);
    std::cout << "\n";

    return -1;
  }

  int R_dims_prod = tensor->dims_prod;
  if ((idx_at+R_dims_prod)>(dims_prod))
  {
    std::string _error = "\n\t- Attributing at pos: \033[32m"+std::to_string((int)idx_at)+"\033[0m with a tensor of size \033[32m"+std::to_string(R_dims_prod)+"\033[0m";
    _error = _error + "\033[0m on tensor \033[95m"+std::string(tensor_name);
    _error = _error + "\033[0m;\n\t- Max idx allowed:  \033[32m"+std::to_string(dims_prod)+"\033[0m.";

    LogErrorS(_error);
    std::cout << "Dimensions:" << "\n";
    PrintDims(dims);
    std::cout << "\n";

    return -1;
  }

  float *base_address = tgt_tensor->tensor_ptr;
  float *device_x = base_address + static_cast<int>(idx_at);

  cudaCheck(cudaMemcpy(device_x, tensor->tensor_ptr, R_dims_prod*sizeof(float), cudaMemcpyHostToHost));
    
  return 0;
}


__global__ void idx_attr_last_dim_kernel(float *tgt,
                           const float *tensor, const float *idx_tensor, 
                           int dims_prod, int last_dim_size) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int C = last_dim_size;
    
    if (i < dims_prod) {
        int b = i / C;
        int v = i % C;
        // i = b * C + v

        float *tgt_b = tgt + b;
        float idx_b = idx_tensor[b];

        if (v==idx_b)
        {
          float ix = tensor[b];
          tgt[i] = ix;
        }
    }
}

extern "C" float AttrTensorOnIdxTensor(char *tensor_name, char *idx_tensor_name, Tensor *R_tensor)
{ 
  std::cout << "ATTR Idx tensor " << tensor_name << " at index tensor " << idx_tensor_name << " with tensor " << R_tensor->name << "\n";

  //std::cout << "\n\n\nIDX " << tensor_name << "\n\n\n\n";  
  
  
  Tensor *tensor = NamedTensorsT[tensor_name];
  Tensor *idx_tensor = NamedTensorsT[idx_tensor_name];


  float *tensor_ptr, *idx_tensor_ptr, *r_tensor_ptr;
  float new_dims_prod;
  std::vector<float> dims, idx_dims, new_dims;

  tensor_ptr = tensor->tensor_ptr;
  idx_tensor_ptr = idx_tensor->tensor_ptr;
  r_tensor_ptr = R_tensor->tensor_ptr;

  dims = tensor->dims;
  idx_dims = idx_tensor->dims;


  //TODO: gather with smaller dimensions
  /*
  if (dims.size()==1)
    new_dims = {1.0f};
  else
    for (int i = 0; i < dims.size()-1; i++)
      new_dims.push_back(dims[i+1]);
  */

  if (dims.size()<=idx_dims.size())
  {
    LogErrorS("Index tensor must have less dimensions than the indexed tensor.");
    return 0;
  }

  

  std::cout << "dim size diff: " << dims.size()-idx_dims.size()  << "\n";
  if((dims.size()-idx_dims.size())==1)
  {
    new_dims_prod = idx_tensor->dims_prod;
    new_dims = idx_tensor->dims;

    std::cout << "INDEX ATTR OVER LAST DIM" << "\n";

    
    int grid_size = tensor->dims_prod;
    int block_size = 32;
    size_t shared_mem_size = 2 * block_size / 32 * sizeof(float);
    idx_attr_last_dim_kernel<<<grid_size, block_size, shared_mem_size>>>(tensor_ptr, r_tensor_ptr, idx_tensor_ptr, tensor->dims_prod, tensor->dims_prod/idx_tensor->dims_prod);
  }


  return 0;
}





Value *BinaryTensorScalarExprAST::codegen(Value *first_arg, Value *scope_str, Value *previous_scope) {
  if (not ShallCodegen)
    return ConstantFP::get(*TheContext, APFloat(0.0f));

  Value *tensor_name = Builder->CreateGlobalString(LHS->GetName());



  std::string pre_dot = LHS->GetPreDot();
  bool is_self = LHS->GetSelf();
  bool is_attr = LHS->GetIsAttribute();

  if (is_attr) { // Gets from pre_dot if it is a class attribute
    Value * object_name = Builder->CreateGlobalString(pre_dot);

    tensor_name = Builder->CreateCall(TheModule->getFunction("ConcatStr"),
                                                      {object_name, tensor_name});
  }
  if (is_self)
    tensor_name = Builder->CreateCall(TheModule->getFunction("ConcatStr"),
                                                      {first_arg, tensor_name});
  if (!(is_self||is_attr))
    tensor_name = Builder->CreateCall(TheModule->getFunction("ConcatStr"),
                                            {scope_str, tensor_name});
    



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
    
    Value *Val = RHS->codegen(first_arg, scope_str, previous_scope);
    if (!Val)
      return nullptr;

    
    
    std::cout << "1 0 attr\n";
    


    //LogErrorS("Attribution from float into tensor is not possible.");    
    
    
      
    
    seen_var_attr=false;
    return Val;
  }


  std::cout << "\n\n\nTensor scalar for LHS: " << LHS->GetName() << " RHS: " << RHS->GetName() << "\n\n\n";
  Value *LtensorPtr = LHS->codegen(first_arg, scope_str, previous_scope);
  Value *R = RHS->codegen(first_arg, scope_str, previous_scope);
  std::cout << "\n\n\nTensor scalar post codegen" << "\n\n\n";



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
                               {LtensorPtr, R}, "cudascalarmult");
  case '/':
    return Builder->CreateCall(TheModule->getFunction("CudaScalarDiv"),
                               {LtensorPtr, R}, "cudascalardiv");
  case 77:
    return Builder->CreateCall(TheModule->getFunction("CudaReverseScalarDiv"),
                               {LtensorPtr, R}, "cudareversescalardiv");
  case '+':
    return Builder->CreateCall(TheModule->getFunction("CudaScalarAdd"),
                               {LtensorPtr, R}, "cudascalaradd");
  case '-':
    return Builder->CreateCall(TheModule->getFunction("CudaScalarSub"),
                               {LtensorPtr, R}, "cudascalarsub");
  case tok_equal:
    return Builder->CreateCall(TheModule->getFunction("CudaScalarEqual"),
                               {LtensorPtr, R}, "cudascalarequal");
  case tok_diff:
    return Builder->CreateCall(TheModule->getFunction("CudaScalarDiff"),
                               {LtensorPtr, R}, "cudascalardiff");
  case '<':
    return Builder->CreateCall(TheModule->getFunction("CudaScalarMinor"),
                               {LtensorPtr, R}, "cudascalarminor");
  case '>':
    return Builder->CreateCall(TheModule->getFunction("CudaScalarHigher"),
                               {LtensorPtr, R}, "cudascalarhigher");
  case tok_minor_eq:
    return Builder->CreateCall(TheModule->getFunction("CudaScalarMinorEq"),
                               {LtensorPtr, R}, "cudascalarminoreq");
  case tok_higher_eq:
    return Builder->CreateCall(TheModule->getFunction("CudaScalarHigherEq"),
                               {LtensorPtr, R}, "cudascalarhighereq");
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



Value *BinaryPinnedScalarExprAST::codegen(Value *first_arg, Value *scope_str, Value *previous_scope) {
  if (not ShallCodegen)
    return ConstantFP::get(*TheContext, APFloat(0.0f));

  Value *tensor_name;



  

  if (Op == '=') {
    seen_var_attr=true;

    
    Value *Val = RHS->codegen(first_arg, scope_str, previous_scope);
    if (!Val)
      return nullptr;
    
    std::cout << "2 0 attr\n";
    std::cout << "is vec: " << LHS->GetIsVec()  << "\n";


    


    VecIdxExprAST   *LHSE = static_cast<VecIdxExprAST *>(LHS.get());
    tensor_name = LHSE->NameSolver->codegen(first_arg, scope_str, previous_scope);

    if (!LHSE)
      return LogErrorV("'=' destiny must be a variable.");

    

    std::vector<Value *> idx_calc_args;

    idx_calc_args.push_back(tensor_name);

    for (int i=0; i<LHSE->Idx.size(); i++)
    {
      idx_calc_args.push_back(LHSE->Idx[i]->codegen(first_arg, scope_str, previous_scope));
    }

    Value *idx_at = Builder->CreateCall(TheModule->getFunction("CalculateIdxOffset"),
                          idx_calc_args);

    Builder->CreateCall(TheModule->getFunction("AttrPinnedOnIdx"),
                          {tensor_name, Val, idx_at});


    /*
    if (LHS->GetIsVec())
    {
      VecIdxExprAST   *LHSE = static_cast<VecIdxExprAST *>(LHS.get());
      if (!LHSE)
        return LogErrorV("'=' destiny must be a variable.");
      std::cout << "is vec: " << LHS->GetIsVec()  << "\n";

      Builder->CreateCall(TheModule->getFunction("AttrPinnedOnIdx"),
                          {tensor_name, LHSE->Idx->codegen(first_arg, scope_str, previous_scope), Val});
    }
      
    else
    {
      VariableExprAST *LHSE = static_cast<VariableExprAST *>(LHS.get());
      if (!LHSE)
        return LogErrorV("'=' destiny must be a variable.");
    }
    */

    


    
    
    
      
    
    seen_var_attr=false;
    return Val;
  }


  
  Value *LtensorPtr = LHS->codegen(first_arg, scope_str, previous_scope);
  Value *R = RHS->codegen(first_arg, scope_str, previous_scope);
  



  
  
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
                               {LtensorPtr, R}, "cudascalarmult");
  case '/':
    return Builder->CreateCall(TheModule->getFunction("CudaScalarDiv"),
                               {LtensorPtr, R}, "cudascalardiv");
  case 77:
    return Builder->CreateCall(TheModule->getFunction("CudaReverseScalarDiv"),
                               {LtensorPtr, R}, "cudareversescalardiv");
  case '+':
    return Builder->CreateCall(TheModule->getFunction("CudaScalarAdd"),
                               {LtensorPtr, R}, "cudascalaradd");
  case '-':
    return Builder->CreateCall(TheModule->getFunction("CudaScalarSub"),
                               {LtensorPtr, R}, "cudascalarsub");
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


extern "C" float min(float l, float r)
{
  if (l<r)
    return l;
  return r;
}
extern "C" float max(float l, float r)
{
  if (l>r)
    return l;
  return r;
}
extern "C" float logE2f(float v) {
  return log2f(v);
}
extern "C" float roundE(float v) {
  return round(v);
}
extern "C" float floorE(float v) {
  return floor(v);
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



extern "C" Tensor *CudaMult(int is_forward_func,
                          Tensor *tensor_x, Tensor *tensor_w) {

  //std::cout << "      L " << LtensorName << "  &  R " << RtensorName << "\n";
    
  std::vector<float> Ldims, Rdims;
  Ldims = tensor_x->dims;
  Rdims = tensor_w->dims;
  float *device_x = tensor_x->tensor_ptr;
  float *device_w = tensor_w->tensor_ptr;


  std::vector<float> linear_layer_dims = format_LinearLayer_Dims(Ldims);
  int input_dims_prod = DimsProd(linear_layer_dims);
  //int resultingDimsProd = (int)linear_layer_dims[0]*Rdims[0];
  int resultingDimsProd = resultingDimsProdOnMult(linear_layer_dims, Rdims);




  float* device_y;
  cudaCheck(cudaMalloc(&device_y, resultingDimsProd * sizeof(float)));

  if (Ldims.size()<2)
    LogErrorS("Tensors multiplication requires at least 2 dimensions.");


  
  //PrintTensorF(device_w, Rdims[0], Rdims[1]);

  matmul_forward2(device_y, device_x, device_w,
                  linear_layer_dims[0], linear_layer_dims[1],
                  Rdims[0]);
                  //64
                  //);


  /*
  std::cout << "CUDA MULT" << "\n";
  PrintDims(Ldims);
  PrintDims(Rdims);
  */
  
  
  std::vector<float> new_dims = NewDimsOnMult(Ldims, Rdims);

  Tensor *new_tensor = createTensor(device_y, new_dims, resultingDimsProd, false, "");
  new_tensor->AttrNodes(tensor_x, tensor_w, mult_op);
  return new_tensor;
}




__global__ void add_forward(float *y, const float *x,
                            const float *w, int dims_prod) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < dims_prod)
        y[i] = x[i] + w[i];
}
__global__ void add_inplace(float *y, const float *x,
                                    int dims_prod) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < dims_prod)
        y[i] = y[i] + x[i];
}

__global__ void sub_forward(float *y, const float *x,
                            const float *w, int dims_prod) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < dims_prod)
        y[i] = x[i] - w[i];
}


__global__ void equal_forward(float *y, const float *x,
                            const float *w, int dims_prod) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < dims_prod)
        y[i] = (x[i]==w[i]) ? 1.0f : 0.0f;
}

extern "C" Tensor *CudaAdd(int is_forward_func,
                          Tensor *tensor_x, Tensor *tensor_w) {

  //std::cout << "Cuda add of\n      L " << tensor_x.name << "  &  R " << tensor_w.name << "\n";
    
  std::vector<float> Ldims, Rdims;
  Ldims = tensor_x->dims;
  Rdims = tensor_w->dims;
  float *device_x = tensor_x->tensor_ptr;
  float *device_w = tensor_w->tensor_ptr;


  if (Ldims!=Rdims)
  {
    LogErrorS("Tried to add tensors of different dimenstions.");
    std::cout << "   Left tensor dims " << "\n   ";
    PrintDims(Ldims);
    std::cout << "\n   Right tensor dims " << "\n   ";
    PrintDims(Rdims);
    std::cout << "\n\n";
  }

  std::vector<float> linear_layer_dims = format_LinearLayer_Dims(Ldims);
  float dims_prod = tensor_x->dims_prod;




  float* device_y;
  cudaCheck(cudaMalloc(&device_y, dims_prod * sizeof(float)));



  int grid_size = dims_prod;
  int block_size = 32;
  size_t shared_mem_size = 2 * block_size / 32 * sizeof(float);
  add_forward<<<grid_size, block_size, shared_mem_size>>>(device_y, device_x, device_w, dims_prod);
  



  Tensor *new_tensor = createTensor(device_y, Ldims, dims_prod, false, "");
  new_tensor->AttrNodes(tensor_x, tensor_w, add_op);
  return new_tensor;
}


extern "C" Tensor *CudaSub(int is_forward_func,
                          Tensor *tensor_x, Tensor *tensor_w) {

  //std::cout << "Cuda add of\n      L " << tensor_x.name << "  &  R " << tensor_w.name << "\n";
    
  std::vector<float> Ldims, Rdims;
  Ldims = tensor_x->dims;
  Rdims = tensor_w->dims;
  float *device_x = tensor_x->tensor_ptr;
  float *device_w = tensor_w->tensor_ptr;


  std::vector<float> linear_layer_dims = format_LinearLayer_Dims(Ldims);
  float dims_prod = tensor_x->dims_prod;




  float* device_y;
  cudaCheck(cudaMalloc(&device_y, dims_prod * sizeof(float)));



  int grid_size = dims_prod;
  int block_size = 32;
  size_t shared_mem_size = 2 * block_size / 32 * sizeof(float);
  sub_forward<<<grid_size, block_size, shared_mem_size>>>(device_y, device_x, device_w, dims_prod);
  
  
  Tensor *new_tensor = createTensor(device_y, Ldims, dims_prod, false, "");  
  new_tensor->AttrNodes(tensor_x, tensor_w, sub_op);
  return new_tensor;
}


extern "C" Tensor *CudaEqual(int is_forward_func,
                          Tensor *tensor_x, Tensor *tensor_w) {

  //std::cout << "Cuda add of\n      L " << tensor_x.name << "  &  R " << tensor_w.name << "\n";
    
  std::vector<float> Ldims, Rdims;
  Ldims = tensor_x->dims;
  Rdims = tensor_w->dims;
  float *device_x = tensor_x->tensor_ptr;
  float *device_w = tensor_w->tensor_ptr;


  std::vector<float> linear_layer_dims = format_LinearLayer_Dims(Ldims);
  float dims_prod = tensor_x->dims_prod;




  float* device_y;
  cudaCheck(cudaMalloc(&device_y, dims_prod * sizeof(float)));



  int grid_size = dims_prod;
  int block_size = 32;
  size_t shared_mem_size = 2 * block_size / 32 * sizeof(float);
  equal_forward<<<grid_size, block_size, shared_mem_size>>>(device_y, device_x, device_w, dims_prod);
  
  
  Tensor *new_tensor = createTensor(device_y, Ldims, dims_prod, false, "");  
  new_tensor->AttrNodes(tensor_x, tensor_w, equal_op);
  return new_tensor;
}


__global__ void hadamard_kernel(float *y, const float *x,
                            const float *w, int dims_prod) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < dims_prod)
        y[i] = x[i] * w[i];
}
extern "C" Tensor *CudaHadamard(int is_forward_func,
                          Tensor *tensor_x, Tensor *tensor_w) {

  //std::cout << "      L " << tensor_x.name << "  &  R " << tensor_w.name << "\n";
    
  std::vector<float> Ldims, Rdims;
  Ldims = tensor_x->dims;
  Rdims = tensor_w->dims;
  float *device_x = tensor_x->tensor_ptr;
  float *device_w = tensor_w->tensor_ptr;

  float dims_prod = tensor_x->dims_prod;


  if (Ldims!=Rdims) //Then broadcast
  { //TODO: change kernel instead
    bool first_iter = true;
    while (Ldims.size()>Rdims.size())
    {
      float tgt_dim_size = Ldims[Rdims.size()];
      float aux_size = DimsProd(Rdims);
      float *aux_tensor, *aux_free;
      cudaMalloc(&aux_tensor, aux_size*tgt_dim_size*sizeof(float));
      cudaMemset(aux_tensor, 0, aux_size*tgt_dim_size*sizeof(float));
      
      int grid_size = dims_prod;
      int block_size = 32;
      size_t shared_mem_size = 2 * block_size / 32 * sizeof(float);
      repeat_interleave_kernel_last_dim<<<grid_size, block_size, shared_mem_size>>>(device_w, aux_tensor, aux_size, tgt_dim_size);

      if (!first_iter)
      {
        aux_free = device_w;
        cudaCheck(cudaFree(aux_free));
      }
      device_w = aux_tensor;
      Rdims.push_back(tgt_dim_size);
      first_iter=false;
    }

    while (Ldims.size()<Rdims.size())
    {
      float tgt_dim_size = Rdims[Ldims.size()];
      float aux_size = DimsProd(Ldims);
      float *aux_tensor, *aux_free;
      cudaMalloc(&aux_tensor, aux_size*tgt_dim_size*sizeof(float));
      cudaMemset(aux_tensor, 0, aux_size*tgt_dim_size*sizeof(float));
      
      int grid_size = dims_prod;
      int block_size = 32;
      size_t shared_mem_size = 2 * block_size / 32 * sizeof(float);
      repeat_interleave_kernel_last_dim<<<grid_size, block_size, shared_mem_size>>>(device_x, aux_tensor, aux_size, tgt_dim_size);

      if (!first_iter)
      {
        aux_free = device_x;
        cudaCheck(cudaFree(aux_free));
      }
      device_x = aux_tensor;
      
      Ldims.push_back(tgt_dim_size);
      
      dims_prod = DimsProd(Ldims);
      first_iter=false;
    }
  }


  float* device_y;
  cudaCheck(cudaMalloc(&device_y, dims_prod * sizeof(float)));


  int grid_size = dims_prod;
  int block_size = 32;
  size_t shared_mem_size = 2 * block_size / 32 * sizeof(float);
  hadamard_kernel<<<grid_size, block_size, shared_mem_size>>>(device_y, device_x, device_w, dims_prod);
  //PrintTensorF(device_y, 2, 2);



  Tensor *new_tensor = createTensor(device_y, Ldims, dims_prod, false, "");
  new_tensor->AttrNodes(tensor_x, tensor_w, hadamard_op);
  return new_tensor;
}



extern "C" void *CudaDiv(int is_forward_func,
                          Tensor *tensor_x, Tensor *tensor_w) {
  
  //std::cout << "TENSOR TENSOR DIV" << "\n";
  
  std::vector<float> Ldims, Rdims;
  Ldims = tensor_x->dims;
  Rdims = tensor_w->dims;
  float *device_x = tensor_x->tensor_ptr;
  float *device_w = tensor_w->tensor_ptr;
  float dims_prod, R_dims_prod;
  dims_prod = tensor_x->dims_prod;
  R_dims_prod = tensor_w->dims_prod;


  if (Ldims!=Rdims) //Then broadcast
  { //TODO: change kernel instead
    bool first_iter = true;
    while (Ldims.size()>Rdims.size())
    {
      float tgt_dim_size = Ldims[Rdims.size()];
      float aux_size = DimsProd(Rdims);
      float *aux_tensor, *aux_free;
      cudaMalloc(&aux_tensor, aux_size*tgt_dim_size*sizeof(float));
      cudaMemset(aux_tensor, 0, aux_size*tgt_dim_size*sizeof(float));
      
      int grid_size = dims_prod;
      int block_size = 32;
      size_t shared_mem_size = 2 * block_size / 32 * sizeof(float);
      repeat_interleave_kernel_last_dim<<<grid_size, block_size, shared_mem_size>>>(device_w, aux_tensor, aux_size, tgt_dim_size);

      if (!first_iter)
      {
        aux_free = device_w;
        cudaCheck(cudaFree(aux_free));
      }
      device_w = aux_tensor;
      Rdims.push_back(tgt_dim_size);
      first_iter=false;
    }

    while (Ldims.size()<Rdims.size())
    {
      float tgt_dim_size = Rdims[Ldims.size()];
      float aux_size = DimsProd(Ldims);
      float *aux_tensor, *aux_free;
      cudaMalloc(&aux_tensor, aux_size*tgt_dim_size*sizeof(float));
      cudaMemset(aux_tensor, 0, aux_size*tgt_dim_size*sizeof(float));
      
      int grid_size = dims_prod;
      int block_size = 32;
      size_t shared_mem_size = 2 * block_size / 32 * sizeof(float);
      repeat_interleave_kernel_last_dim<<<grid_size, block_size, shared_mem_size>>>(device_x, aux_tensor, aux_size, tgt_dim_size);

      if (!first_iter)
      {
        aux_free = device_x;
        cudaCheck(cudaFree(aux_free));
      }
      device_x = aux_tensor;
      
      Ldims.push_back(tgt_dim_size);
      
      dims_prod = DimsProd(Ldims);
      first_iter=false;
    }
  }


  //if (dims_prod!=R_dims_prod)
  //  LogErrorS("Tensors division has tensors of different dimensions.");


  float* device_y;
  cudaCheck(cudaMalloc(&device_y, dims_prod * sizeof(float)));
  


  int grid_size = dims_prod;
  int block_size = 32;
  size_t shared_mem_size = 2 * block_size / 32 * sizeof(float);
  tensor_div<<<grid_size, block_size, shared_mem_size>>>(device_w, device_x, device_y, dims_prod);

  Tensor *new_tensor = createTensor(device_y, Ldims, dims_prod, false, "");  
  new_tensor->AttrNodes(tensor_x, tensor_w, div_op);
  return new_tensor;
}






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
__global__ void onehot_kernel(const float *tensor,
                           float *probs,
                           int B, int C) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    //int i = threadIdx.x;
    
    if (i < B * C) {
        int b = i / (C);
        int v = i % C;

        float *probs_b = probs + b * C;
        int ix = tensor[b];

        float indicator = (v==ix) ? 1.0f : 0.0f;
        probs_b[v] = indicator;
    }
}


extern "C" void *onehot(Tensor tensor, float num_classes)
{
  //std::cout << "ONEHOT OF " << tensor.name << "\n";

  float *tensor_ptr = tensor.tensor_ptr;
  std::vector<float> dims, new_dims;
  dims = tensor.dims;
  new_dims = tensor.dims;
  new_dims.push_back(num_classes);
  
  int B = DimsProd(dims);
  int C = (int)num_classes;

  float *probs;

  cudaMalloc(&probs, B*C*sizeof(float));
  cudaMemset(probs, 0, B*C*sizeof(float));
  
  int grid_size, block_size, shared_mem_size; 
  std::vector<int> grid_block_mem_sizes = CalculateGridAndBlockSizes(B*C);
  grid_size = grid_block_mem_sizes[0];
  block_size = grid_block_mem_sizes[1];
  shared_mem_size = grid_block_mem_sizes[2];

  onehot_kernel<<<grid_size, block_size, shared_mem_size>>>(tensor_ptr, probs, B, C);
  //grid_size = ceil_div(B*C, block_size);
  //onehot_kernel<<<grid_size, block_size>>>(tensor, probs, B, C);


  Tensor *new_tensor = createTensor(probs, new_dims, DimsProd(new_dims), false, "");
  new_tensor->op=onehot_op;
  return new_tensor;
}


extern "C" float shape(Tensor tensor)
{
  std::cout << "\nTensor \033[95m" << tensor.scopeless_name << "\033[0m:\n   ";
  PrintDims(tensor.dims);

  return 0;
}




extern "C" void *repeat_interleave(Tensor tensor, float repeats, float dim)
{
  //std::cout << "REPEAT_interleave OF " << tensor.name << " with " << repeats << " repeats.\n";

  float *tensor_ptr = tensor.tensor_ptr;
  std::vector<float> dims, new_dims;
  dims = tensor.dims;
  if (dim<0)
    dim = dims.size()+dim;
  new_dims = tensor.dims;
  new_dims[dim] = new_dims[dim]*repeats;
  
  int B = DimsProd(dims);
  int C = (int)repeats;

  float *probs;

  cudaMalloc(&probs, B*C*sizeof(float));
  cudaMemset(probs, 0, B*C*sizeof(float));
  


  int grid_size = B;
  int block_size = 32;
  size_t shared_mem_size = 2 * block_size / 32 * sizeof(float);
  if (dim==(dims.size()-1))
    repeat_interleave_kernel_last_dim<<<grid_size, block_size, shared_mem_size>>>(tensor_ptr, probs, B, C);
  //grid_size = ceil_div(B*C, block_size);
  //onehot_kernel<<<grid_size, block_size>>>(tensor, probs, B, C);


  Tensor *new_tensor = createTensor(probs, new_dims, DimsProd(new_dims), false, "");
  return new_tensor;
}

//TODO: mean over axis
extern "C" void *mean(Tensor tensor, float first_dim, ...)
{
  //std::cout << "SUM OF " << tensor.name << "\n";


  float *tensor_ptr = tensor.tensor_ptr;
  std::vector<float> dims = tensor.dims;
  float *summed;


  va_list args;
  va_start(args, first_dim);

  if (first_dim==TERMINATE_VARARG)
  {
    va_end(args);
    float *ret;
    int dims_prod = DimsProd(dims);

    summed = new float[dims_prod];
    cudaCheck(cudaMemcpy(summed, tensor_ptr, dims_prod*sizeof(float), cudaMemcpyDeviceToHost));

    cudaCheck(cudaMalloc(&ret, 1*sizeof(float)));
  
    float tensor_sum=0;
    for(int i=0; i<dims_prod; i++)
      tensor_sum += summed[i];
    tensor_sum = tensor_sum/tensor.dims_prod;
    
    delete[] summed;
  
    float *aux = new float[1];
    aux[0] = tensor_sum;
    cudaCheck(cudaMemcpy(ret, aux, 1*sizeof(float), cudaMemcpyHostToDevice));
    delete[] aux;
  
    std::vector<float> new_dims;
    new_dims.push_back(1.0f);
  
    Tensor *new_tensor = createTensor(ret, new_dims, 1.0f, false, "");
    new_tensor->op=mean_op;
    return new_tensor;
  }


  std::vector<float> sum_dims, new_dims;
  if (first_dim<0)
    first_dim = dims.size()+first_dim;
  sum_dims.push_back(first_dim);

  for (int i=0; i<10; i++)
  {
    if (i==9)
    {
      LogErrorS("A tensor with 10 dimensions???");
      return nullptr;
    }

    float dim = va_arg(args, float);
    
    if (dim==TERMINATE_VARARG)
      break;
    if (in_float_vec(dim, sum_dims))  
    {
      std::string _error = "Dim "+std::to_string(dim) + " duplicated at tensor.mean() operation.";
      LogErrorS(_error);
      return nullptr;
    }
    if (dim<0)
      dim = dims.size()+dim;
    sum_dims.push_back(dim);
  }
  va_end(args);
  
  
  float summed_dim;
  for (int i=0; i<dims.size(); i++)
    if (!in_float_vec(i, sum_dims))
      new_dims.push_back(dims[i]);
    else
      summed_dim=dims[i];


  int dims_prod = DimsProd(dims);
  int new_dims_prod = DimsProd(new_dims);

  
  cudaMalloc(&summed, new_dims_prod*sizeof(float));
  cudaMemset(summed, 0, new_dims_prod * sizeof(float));

  //std::cout << "\n\nDims prod: " << dims_prod << "\nNew dims prod: " << new_dims_prod << "\nSummed dim size: " << summed_dim << "\n\n";

  
  int grid_size = dims_prod;
  int block_size = 32;
  size_t shared_mem_size = 2 * block_size / 32 * sizeof(float);

  
  /*
  if (dims.size()==1)
  {
    sum_single_dim_kernel<<<grid_size, block_size, shared_mem_size>>>(tensor_ptr, summed, dims_prod);
    new_dims = {1.0f};
  }
  else if (sum_dims[0]==(dims.size()-1))
    sum_over_last_dim_kernel<<<grid_size, block_size, shared_mem_size>>>(tensor_ptr, summed, dims_prod, summed_dim);
  if (sum_dims[0]==(dims.size()-2))
    sum_over_semilast_dim_kernel<<<grid_size, block_size, shared_mem_size>>>(tensor_ptr, summed, dims_prod, dims[dims.size()-1], dims[dims.size()-2]);


  Tensor *new_tensor = createTensor(summed, new_dims, DimsProd(new_dims), false, "");
  return new_tensor;
  */
  LogErrorS("Mean of specific dim is not implemented yet.");
  return nullptr;
}



__global__ void sum_single_dim_kernel(const float *tensor,
                           float *summed,
                           int dims_prod) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    int C = dims_prod;
    
    if (i < dims_prod) {
        int b = i / (C); // b updates only when v reaches it's maximum value
        int v = i % C;
        // i = b*C + v


        float ix = tensor[i];

        atomicAdd(summed, ix);        
    }
}

__global__ void sum_over_last_dim_kernel(const float *tensor,
                           float *summed,
                           int dims_prod, int summed_dim_size) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    int C = summed_dim_size;
    
    if (i < dims_prod) {
        int b = i / (C); // b updates only when v reaches it's maximum value
        int v = i % C;
        // i = b*C + v

        float *summed_b = summed + b;

        float ix = tensor[i];

        atomicAdd(summed_b, ix);        
    }
}

__global__ void sum_over_semilast_dim_kernel(const float *tensor,
                           float *summed,
                           int dims_prod, int last_dim_size, int summed_dim_size) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    int C = last_dim_size;
    int D = summed_dim_size*last_dim_size;
    
    if (i < dims_prod) {
        int b = i / C; // b updates only when v reaches it's maximum value
        int d = i / D;
        int v = i % C;
        // i = b*C + v

        float *summed_b = summed + v + d*C;

        float ix = tensor[i];

        atomicAdd(summed_b, ix);        
    }
}




extern "C" void *sum(Tensor tensor, float first_dim, ...)
{
  //std::cout << "SUM OF " << tensor.name << "\n";


  float *tensor_ptr = tensor.tensor_ptr;
  std::vector<float> dims = tensor.dims;
  float *summed;


  va_list args;
  va_start(args, first_dim);

  if (first_dim==TERMINATE_VARARG)
  {
    va_end(args);
    float *ret;
    int dims_prod = DimsProd(dims);

    summed = new float[dims_prod];
    cudaCheck(cudaMemcpy(summed, tensor_ptr, dims_prod*sizeof(float), cudaMemcpyDeviceToHost));

    cudaCheck(cudaMalloc(&ret, 1*sizeof(float)));
  
    float tensor_sum=0;
    for(int i=0; i<dims_prod; i++)
      tensor_sum += summed[i];
    
    delete[] summed;
  
    float *aux = new float[1];
    aux[0] = tensor_sum;
    cudaCheck(cudaMemcpy(ret, aux, 1*sizeof(float), cudaMemcpyHostToDevice));  
    delete[] aux;
  
    std::vector<float> new_dims;
    new_dims.push_back(1.0f);
  
    Tensor *new_tensor = createTensor(ret, new_dims, 1.0f, false, "");
    new_tensor->op=sum_op;
    return new_tensor;
  }


  std::vector<float> sum_dims, new_dims;
  if (first_dim<0)
    first_dim = dims.size()+first_dim;
  sum_dims.push_back(first_dim);

  for (int i=0; i<10; i++)
  {
    if (i==9)
    {
      LogErrorS("A tensor with 10 dimensions???");
      return nullptr;
    }

    float dim = va_arg(args, float);
    
    if (dim==TERMINATE_VARARG)
      break;
    if (in_float_vec(dim, sum_dims))  
    {
      std::string _error = "Dim "+std::to_string(dim) + " duplicated at tensor.sum() operation.";
      LogErrorS(_error);
      return nullptr;
    }
    if (dim<0)
      dim = dims.size()+dim;
    sum_dims.push_back(dim);
  }
  va_end(args);
  
  
  float summed_dim;
  for (int i=0; i<dims.size(); i++)
    if (!in_float_vec(i, sum_dims))
      new_dims.push_back(dims[i]);
    else
      summed_dim=dims[i];


  int dims_prod = DimsProd(dims);
  int new_dims_prod = DimsProd(new_dims);

  
  cudaMalloc(&summed, new_dims_prod*sizeof(float));
  cudaMemset(summed, 0, new_dims_prod * sizeof(float));

  //std::cout << "\n\nDims prod: " << dims_prod << "\nNew dims prod: " << new_dims_prod << "\nSummed dim size: " << summed_dim << "\n\n";

  
  int grid_size = dims_prod;
  int block_size = 32;
  size_t shared_mem_size = 2 * block_size / 32 * sizeof(float);

  
  if (dims.size()==1)
  {
    sum_single_dim_kernel<<<grid_size, block_size, shared_mem_size>>>(tensor_ptr, summed, dims_prod);
    new_dims = {1.0f};
  }
  else if (sum_dims[0]==(dims.size()-1))
    sum_over_last_dim_kernel<<<grid_size, block_size, shared_mem_size>>>(tensor_ptr, summed, dims_prod, summed_dim);
  if (sum_dims[0]==(dims.size()-2))
    sum_over_semilast_dim_kernel<<<grid_size, block_size, shared_mem_size>>>(tensor_ptr, summed, dims_prod, dims[dims.size()-1], dims[dims.size()-2]);


  Tensor *new_tensor = createTensor(summed, new_dims, DimsProd(new_dims), false, "");
  new_tensor->op=sum_op;
  return new_tensor;
}

__device__ float atomicMul(float* address, float val) {
    int *addr_as_int = (int *)address;
    int old = *addr_as_int, assumed;
    do {
        assumed = old;
        old = atomicCAS(addr_as_int, assumed,
                        __float_as_int(val * __int_as_float(assumed)));
    } while (assumed != old);
    return __int_as_float(old);
}


__global__ void prod_single_dim_kernel(const float *tensor,
                           float *summed,
                           int dims_prod) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    int C = dims_prod;
    
    if (i < dims_prod) {
        int b = i / (C); // b updates only when v reaches it's maximum value
        int v = i % C;
        // i = b*C + v


        float ix = tensor[i];

        atomicMul(summed, ix);        
    }
}

__global__ void prod_over_last_dim_kernel(const float *tensor,
                           float *summed,
                           int dims_prod, int summed_dim_size) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    int C = summed_dim_size;
    
    if (i < dims_prod) {
        int b = i / (C); // b updates only when v reaches it's maximum value
        int v = i % C;
        // i = b*C + v

        float *summed_b = summed + b;

        float ix = tensor[i];

        atomicMul(summed_b, ix);        
    }
}

__global__ void prod_over_semilast_dim_kernel(const float *tensor,
                           float *summed,
                           int dims_prod, int last_dim_size, int summed_dim_size) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    int C = last_dim_size;
    int D = summed_dim_size*last_dim_size;
    
    if (i < dims_prod) {
        int b = i / C; // b updates only when v reaches it's maximum value
        int d = i / D;
        int v = i % C;
        // i = b*C + v

        float *summed_b = summed + v + d*C;

        float ix = tensor[i];

        atomicMul(summed_b, ix);        
    }
}

extern "C" void *prod(Tensor tensor, float first_dim, ...)
{
  std::cout << "PROD OF " << tensor.name << "\n";


  float *tensor_ptr = tensor.tensor_ptr;
  std::vector<float> dims = tensor.dims;
  float *summed;


  va_list args;
  va_start(args, first_dim);

  if (first_dim==TERMINATE_VARARG)
  {
    va_end(args);
    int dims_prod = DimsProd(dims);

    cudaMalloc(&summed, dims_prod*sizeof(float));
    cudaMemcpy(summed, tensor_ptr, dims_prod*sizeof(float), cudaMemcpyDeviceToHost);
    
    float tensor_sum=0;
    for(int i=0; i<dims_prod; i++)
      tensor_sum += summed[i];
    tensor_sum = tensor_sum;

    std::cout << "prod: " << tensor_sum << "\n";

    return summed;
  }


  std::vector<float> sum_dims, new_dims;
  if (first_dim<0)
    first_dim = dims.size()+first_dim;
  sum_dims.push_back(first_dim);

  for (int i=0; i<10; i++)
  {
    if (i==9)
    {
      LogErrorS("A tensor with 10 dimensions???");
      return nullptr;
    }

    float dim = va_arg(args, float);
    
    if (dim==TERMINATE_VARARG)
      break;
    if (in_float_vec(dim, sum_dims))  
    {
      std::string _error = "Dim "+std::to_string(dim) + " duplicated at tensor.sum() operation.";
      LogErrorS(_error);
      return nullptr;
    }
    if (dim<0)
      dim = dims.size()+dim;
    sum_dims.push_back(dim);
  }
  va_end(args);
  
  
  float summed_dim;
  for (int i=0; i<dims.size(); i++)
    if (!in_float_vec(i, sum_dims))
      new_dims.push_back(dims[i]);
    else
      summed_dim=dims[i];


  int dims_prod = DimsProd(dims);
  int new_dims_prod = DimsProd(new_dims);

  
  float *init_prod = new float[new_dims_prod];
  init_prod = make_ones_float(new_dims_prod);
  cudaMalloc(&summed, new_dims_prod*sizeof(float));
  cudaMemcpy(summed, init_prod, new_dims_prod * sizeof(float), cudaMemcpyHostToDevice);
  delete[] init_prod;

  PrintTensorF(summed, new_dims_prod,1);

  //std::cout << "\n\nDims prod: " << dims_prod << "\nNew dims prod: " << new_dims_prod << "\nSummed dim size: " << summed_dim << "\n\n";

  
  int grid_size = dims_prod;
  int block_size = 32;
  size_t shared_mem_size = 2 * block_size / 32 * sizeof(float);

  
  if (dims.size()==1)
  {
    prod_single_dim_kernel<<<grid_size, block_size, shared_mem_size>>>(tensor_ptr, summed, dims_prod);
    new_dims = {1.0f};
  }
  else if (sum_dims[0]==(dims.size()-1))
  {
    prod_over_last_dim_kernel<<<grid_size, block_size, shared_mem_size>>>(tensor_ptr, summed, dims_prod, summed_dim);
    std::cout << "prod_over_last_dim_kernel" << "\n";
  }
  if (sum_dims[0]==(dims.size()-2))
    prod_over_semilast_dim_kernel<<<grid_size, block_size, shared_mem_size>>>(tensor_ptr, summed, dims_prod, dims[dims.size()-1], dims[dims.size()-2]);


  Tensor *new_tensor = createTensor(summed, new_dims, DimsProd(new_dims), false, "");
  return new_tensor;
}


__global__ void max_over_last_dim_kernel(const float *tensor,
                           float *maxed,
                           int dims_prod, int maxed_dim_size) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    int C = maxed_dim_size;
    
    if (i < dims_prod) {
        int b = i / C;
        int v = i % C;
        // i = b * C + v

        float *max_b = maxed + b;

        float ix = tensor[i];


        unsigned int *const addr_as_ui = (unsigned int *)max_b;
        unsigned int old = *addr_as_ui, assumed;
        do {
          assumed = old;
          if (__uint_as_float(assumed) >= ix) break;
          old = atomicCAS(addr_as_ui, assumed, __float_as_uint(ix));
        } while (assumed != old);
    }
}

__global__ void max_over_semilast_dim_kernel(const float *tensor,
                           float *maxed,
                           int dims_prod, int last_dim_size, int maxed_dim_size) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    int C = last_dim_size;
    int D = maxed_dim_size*last_dim_size;
    
    
    if (i < dims_prod) {
        int b = i / C;
        int d = i / D;
        int v = i % C;
        // i = b * C + v

        float *max_b = maxed + v + d*C;

        float ix = tensor[i];

        unsigned int *const addr_as_ui = (unsigned int *)max_b;
        unsigned int old = *addr_as_ui, assumed;
        do {
          assumed = old;
          if (__uint_as_float(assumed) >= ix) break;
          old = atomicCAS(addr_as_ui, assumed, __float_as_uint(ix));
        } while (assumed != old);
    }
}


extern "C" void *tmax(Tensor tensor, float first_dim, ...) 
{ //TODO: automatic type detection for max and min (float vs tensor)
  
  //std::cout << "MAX OF " << tensor.name << "\n";
  

  float *tensor_ptr = tensor.tensor_ptr;
  std::vector<float> dims = tensor.dims;
  float *summed;


  va_list args;
  va_start(args, first_dim);

  if (first_dim==TERMINATE_VARARG)
  {
    va_end(args);
    int dims_prod = DimsProd(dims);

    cudaMalloc(&summed, dims_prod*sizeof(float));
    cudaMemcpy(summed, tensor_ptr, dims_prod*sizeof(float), cudaMemcpyDeviceToHost);
    
    float tensor_sum=0;
    for(int i=0; i<dims_prod; i++)
      tensor_sum += summed[i];
    tensor_sum = tensor_sum;

    std::cout << "Sum: " << tensor_sum << "\n";

    return summed;
  }


  std::vector<float> sum_dims, new_dims;
  if (first_dim<0)
    first_dim = dims.size()+first_dim;
  sum_dims.push_back(first_dim);

  for (int i=0; i<10; i++)
  {
    if (i==9)
    {
      LogErrorS("A tensor with 10 dimensions???");
      return nullptr;
    }

    float dim = va_arg(args, float);
    
    if (dim==TERMINATE_VARARG)
      break;
    if (in_float_vec(dim, sum_dims))  
    {
      std::string _error = "Dim "+std::to_string(dim) + " duplicated at tensor.sum() operation.";
      LogErrorS(_error);
      return nullptr;
    }
    if (dim<0)
      dim = dims.size()+dim;
    sum_dims.push_back(dim);
  }
  va_end(args);
  
  
  float summed_dim;
  for (int i=0; i<dims.size(); i++)
    if (!in_float_vec(i, sum_dims))
      new_dims.push_back(dims[i]);
    else
      summed_dim=dims[i];


  int dims_prod = DimsProd(dims);
  int new_dims_prod = DimsProd(new_dims);

  
  cudaMalloc(&summed, new_dims_prod*sizeof(float));
  cudaMemset(summed, 0, new_dims_prod * sizeof(float));


  //std::cout << "\n\nDims prod: " << dims_prod << "\nNew dims prod: " << new_dims_prod << "\nSummed dim size: " << summed_dim << "\n\n";

  
  int grid_size = dims_prod;
  int block_size = 32;
  size_t shared_mem_size = 2 * block_size / 32 * sizeof(float);

  // AtomicMax does not handle negative numbers, so gambiarra. :D (1 hour for this)
  vec_add<<<grid_size, block_size, shared_mem_size>>>(50000, tensor_ptr, tensor_ptr, dims_prod);
  if (sum_dims[0]==(dims.size()-1))
    max_over_last_dim_kernel<<<grid_size, block_size, shared_mem_size>>>(tensor_ptr, summed, dims_prod, summed_dim);
  if (sum_dims[0]==(dims.size()-2))
    max_over_semilast_dim_kernel<<<grid_size, block_size, shared_mem_size>>>(tensor_ptr, summed, dims_prod, dims[dims.size()-1], dims[dims.size()-2]);
  vec_sub<<<grid_size, block_size, shared_mem_size>>>(50000, summed, summed, new_dims_prod);
  vec_sub<<<grid_size, block_size, shared_mem_size>>>(50000, tensor_ptr, tensor_ptr, dims_prod);


  Tensor *new_tensor = createTensor(summed, new_dims, DimsProd(new_dims), false, "");
  new_tensor->op=max_op;
  return new_tensor;
}


__global__ void argmax_over_last_dim_kernel(const float *tensor,
                           float *maxed, float *argmaxed,
                           int dims_prod, int maxed_dim_size) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    int C = maxed_dim_size;
    
    if (i < dims_prod) {
        int b = i / C;
        int v = i % C;
        // i = b * C + v

        float *max_b = maxed + b;
        float *argmax_b = argmaxed + b;

        float ix = tensor[i];

        // max
        int *addr_as_int = (int *)max_b;
        int old_int = *addr_as_int, assumed_int;
        float old_val;
        do {
            assumed_int = old_int;
            old_val = __int_as_float(assumed_int);
            if (old_val >= ix) break;
            old_int = atomicCAS(addr_as_int, assumed_int, __float_as_int(ix));
        } while (assumed_int != old_int);

        // argmax
        if (__int_as_float(old_int) < ix) {
            int *addr_as_int_argmax = (int *)argmax_b;
            atomicExch(addr_as_int_argmax, __float_as_int((float)v));
        }
      }
}

extern "C" void *argmax(Tensor tensor, float first_dim, ...) 
{
  //std::cout << "ARGMAX OF " << tensor.name << "\n";

  float *tensor_ptr = tensor.tensor_ptr;
  std::vector<float> dims = tensor.dims;
  float *maxed, *argmaxed;


  va_list args;
  va_start(args, first_dim);

  if (first_dim==TERMINATE_VARARG)
  {
    va_end(args);
    LogErrorS("Argmax is only supported at dim -1.");
    return nullptr;
  }


  std::vector<float> sum_dims, new_dims;
  if (first_dim<0)
    first_dim = dims.size()+first_dim;
  sum_dims.push_back(first_dim);

  for (int i=0; i<10; i++)
  {
    if (i==9)
    {
      LogErrorS("A tensor with 10 dimensions???");
      return nullptr;
    }

    float dim = va_arg(args, float);
    
    if (dim==TERMINATE_VARARG)
      break;
    if (in_float_vec(dim, sum_dims))  
    {
      std::string _error = "Dim "+std::to_string(dim) + " duplicated at tensor.sum() operation.";
      LogErrorS(_error);
      return nullptr;
    }
    if (dim<0)
      dim = dims.size()+dim;
    sum_dims.push_back(dim);
  }
  va_end(args);
  
  
  float maxed_dim;
  for (int i=0; i<dims.size(); i++)
    if (!in_float_vec(i, sum_dims))
      new_dims.push_back(dims[i]);
    else
      maxed_dim=dims[i];


  int dims_prod = DimsProd(dims);
  int new_dims_prod = DimsProd(new_dims);

  
  cudaMalloc(&maxed, new_dims_prod*sizeof(float));
  cudaMalloc(&argmaxed, new_dims_prod*sizeof(float));
  cudaMemset(maxed, 0, new_dims_prod * sizeof(float));
  cudaMemset(argmaxed, 0, new_dims_prod * sizeof(float));


  //std::cout << "\n\nDims prod: " << dims_prod << "\nNew dims prod: " << new_dims_prod << "\nMaxed dim size: " << summed_dim << "\n\n";

  
  int grid_size = dims_prod;
  int block_size = 32;
  size_t shared_mem_size = 2 * block_size / 32 * sizeof(float);

  
  vec_add<<<grid_size, block_size, shared_mem_size>>>(50000, tensor_ptr, tensor_ptr, dims_prod);
  if (sum_dims[0]==(dims.size()-1))
    argmax_over_last_dim_kernel<<<grid_size, block_size, shared_mem_size>>>(tensor_ptr, maxed, argmaxed, dims_prod, maxed_dim);
  //if (sum_dims[0]==(dims.size()-2))
  //  max_over_semilast_dim_kernel<<<grid_size, block_size, shared_mem_size>>>(tensor, maxed, dims_prod, dims[dims.size()-1], dims[dims.size()-2]);
  vec_sub<<<grid_size, block_size, shared_mem_size>>>(50000, tensor_ptr, tensor_ptr, dims_prod);

  cudaCheck(cudaFree(maxed));

  Tensor *new_tensor = createTensor(argmaxed, new_dims, DimsProd(new_dims), false, "");
  new_tensor->op=argmax_op;
  return new_tensor;
}

__global__ void topk_kernel(const float *tensor, float *topk,
                           float *maxed, float *argmaxed,
                           int dims_prod, int maxed_dim_size,
                           int j, int k) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    int C = maxed_dim_size;
    
    if (i < dims_prod) {
        int b = i / C;
        int v = i % C;
        // i = b * C + v

        float *max_b = maxed + b;
        float *argmax_b = argmaxed + b;
        float *topk_b = topk + b*k + j;

        float ix = tensor[i];

        // max
        int *addr_as_int = (int *)max_b;
        int old_int = *addr_as_int, assumed_int;
        float old_val;
        do {
            assumed_int = old_int;
            old_val = __int_as_float(assumed_int);
            if (old_val >= ix) break;
            old_int = atomicCAS(addr_as_int, assumed_int, __float_as_int(ix));
        } while (assumed_int != old_int);

        // argmax & topk
        if (__int_as_float(old_int) < ix) {
            int *addr_as_int_argmax = (int *)argmax_b;
            atomicExch(addr_as_int_argmax, __float_as_int((float)v));

            int *addr_as_int_topk = (int *)topk_b;
            atomicExch(addr_as_int_topk, __float_as_int((float)v));
        }
      }
}

__global__ void topk_erase_argmax_aux_kernel(float *tensor,
                           float *argmaxed, int dims_prod, int maxed_dim_size) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    int C = maxed_dim_size;
    
    if (i < dims_prod) {
        int b = i / C;
        int v = i % C;
        // i = b * C + v

        float *tensor_b = tensor + b * C;

        float ix = argmaxed[b];

        float indicator = (v==ix) ? 0 : ix;
        if (v==ix)
          tensor_b[v] = 0;
      }
}

extern "C" void *topk(Tensor tensor, float k) 
{
  std::cout << "TOPK OF " << tensor.name << "\n";

  float *tensor_ptr = tensor.tensor_ptr;
  std::vector<float> dims = tensor.dims;
  float *maxed, *argmaxed, *topk, *tensor_copy;


  std::vector<float> new_dims = RemoveLastDim(dims);
  std::vector<float> topk_dims = RemoveLastDim(dims);
  float new_dims_prod = DimsProd(new_dims);
  int dims_prod = DimsProd(dims);
  topk_dims.push_back(k);
  float topk_dims_prod = DimsProd(topk_dims);

  float maxed_dim = dims[dims.size()-1];

  
  cudaMalloc(&maxed, new_dims_prod*sizeof(float));
  cudaMalloc(&argmaxed, new_dims_prod*sizeof(float));
  cudaMalloc(&topk, topk_dims_prod * sizeof(float));
  cudaMalloc(&tensor_copy, dims_prod*sizeof(float));
  cudaMemset(maxed, 0, new_dims_prod*sizeof(float));
  cudaMemset(argmaxed, 0, new_dims_prod*sizeof(float));
  cudaMemset(topk, 0, topk_dims_prod * sizeof(float));
  cudaMemcpy(tensor_copy, tensor_ptr, dims_prod*sizeof(float), cudaMemcpyDeviceToDevice);


  //std::cout << "\n\nDims prod: " << dims_prod << "\nNew dims prod: " << new_dims_prod << "\nMaxed dim size: " << summed_dim << "\n\n";

  
  int grid_size = dims_prod;
  int block_size = 32;
  size_t shared_mem_size = 2 * block_size / 32 * sizeof(float);

  
  vec_add<<<grid_size, block_size, shared_mem_size>>>(50000, tensor_copy, tensor_copy, dims_prod);
  
  for (int i=0; i<k; i++)
  {
    //std::cout << "Top k at iter:" << i << "\n";
    topk_kernel<<<grid_size, block_size, shared_mem_size>>>(tensor_copy, topk, maxed, argmaxed, dims_prod, maxed_dim, i, k);
    //PrintTensorF(maxed, 3, 1);
    //PrintTensorF(argmaxed, 3, 1);
    //std::cout << "Topk" << "\n";
    //PrintTensorF(topk, 3, k);
    topk_erase_argmax_aux_kernel<<<grid_size, block_size, shared_mem_size>>>(tensor_copy, argmaxed, dims_prod, maxed_dim);
    cudaMemset(maxed, 0, new_dims_prod*sizeof(float));
    //std::cout << "\n\n\n\n";
  }
  cudaCheck(cudaFree(tensor_copy));
  cudaCheck(cudaFree(maxed));
  cudaCheck(cudaFree(argmaxed));

  Tensor *new_tensor = createTensor(topk, topk_dims, topk_dims_prod, false, "");
  new_tensor->op=topk_op;
  return new_tensor;
}





extern "C" void *clip(Tensor tensor, float _min, float _max)
{
  float *tensor_ptr = tensor.tensor_ptr;
  std::vector<float> dims = tensor.dims;
  
  int B = DimsProd(dims);

  float* device_y;
  cudaCheck(cudaMalloc(&device_y, B*sizeof(float)));


  int grid_size = B;
  int block_size = 32;
  size_t shared_mem_size = 2 * block_size / 32 * sizeof(float);
  tensor_clip<<<grid_size, block_size, shared_mem_size>>>(tensor_ptr, device_y, _min, _max, B);

  
  Tensor *new_tensor = createTensor(device_y, dims, tensor.dims_prod, false, "");
  new_tensor->op=clip_op; //TODO: what is the grad of clip?
  return new_tensor;
}


__global__ void gelu_forward_kernel1(const float* inp, float* out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        float xi = inp[i];
        float cube = 0.044715f * xi * xi * xi;
        out[i] = 0.5f * xi * (1.0f + tanhf(GELU_SCALING_FACTOR * (xi + cube)));
    }
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
  
  int grid_size, block_size, shared_mem_size; 
  std::vector<int> grid_block_mem_sizes = CalculateGridAndBlockSizes(N);
  grid_size = grid_block_mem_sizes[0];
  block_size = grid_block_mem_sizes[1];
  shared_mem_size = grid_block_mem_sizes[2];

  gelu_backward1<<<grid_size, block_size>>>(dinp, inp, dout, N);
  cudaCheck(cudaGetLastError());
}

extern "C" void *gelu(Tensor *tensor)
{
  float *tensor_ptr = tensor->tensor_ptr;
  std::vector<float> dims = tensor->dims;
  

  float dims_prod = DimsProd(dims);

  int grid_size, block_size, shared_mem_size; 
  std::vector<int> grid_block_mem_sizes = CalculateGridAndBlockSizes(dims_prod);
  grid_size = grid_block_mem_sizes[0];
  block_size = grid_block_mem_sizes[1];
  shared_mem_size = grid_block_mem_sizes[2];

  std::vector<float> linear_layer_dims = format_LinearLayer_Dims(dims);
  

  float *y;
  cudaMalloc(&y, dims_prod*sizeof(float));
  cudaMemset(y, 0, dims_prod * sizeof(float));


  gelu_forward_kernel1<<<grid_size, block_size>>>(tensor_ptr, y, dims_prod);
  cudaCheck(cudaGetLastError());

  
  int is_forward_func=1;
  

  Tensor *new_tensor = createTensor(y, dims, DimsProd(dims), false, "");
  new_tensor->AttrLNode(tensor, gelu_op);
  return new_tensor;
}



__global__ void sigmoid_forward_kernel(const float* inp, float* out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        float x = inp[i];
        out[i] = 1/(1+exp(-x));
    }
}
__global__ void sigmoid_backward_kernel(float* dinp, const float* out, const float* dout, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        float x = (float)out[i];
        float local_grad = x * (1 - x);
        dinp[i] = (float)(local_grad * (float)dout[i]);
    }
}

void sigmoid_backward(const float* out, float B, float C, float* dinp, const float* dout) {
  float N = B * C;
  
  int grid_size, block_size, shared_mem_size; 
  std::vector<int> grid_block_mem_sizes = CalculateGridAndBlockSizes(N);
  grid_size = grid_block_mem_sizes[0];
  block_size = grid_block_mem_sizes[1];
  shared_mem_size = grid_block_mem_sizes[2];

  sigmoid_backward_kernel<<<grid_size, block_size>>>(dinp, out, dout, N);
  cudaCheck(cudaGetLastError());
}

extern "C" void *sigmoid(Tensor *tensor)
{
  float *tensor_ptr = tensor->tensor_ptr;
  std::vector<float> dims = tensor->dims;
  

  float dims_prod = DimsProd(dims);

  int grid_size, block_size, shared_mem_size; 
  std::vector<int> grid_block_mem_sizes = CalculateGridAndBlockSizes(dims_prod);
  grid_size = grid_block_mem_sizes[0];
  block_size = grid_block_mem_sizes[1];
  shared_mem_size = grid_block_mem_sizes[2];

  std::vector<float> linear_layer_dims = format_LinearLayer_Dims(dims);
  

  float *y;
  cudaMalloc(&y, dims_prod*sizeof(float));
  cudaMemset(y, 0, dims_prod * sizeof(float));

  
  sigmoid_forward_kernel<<<grid_size, block_size>>>(tensor_ptr, y, dims_prod);
  cudaCheck(cudaGetLastError());

  
  int is_forward_func=1;


  Tensor *new_tensor = createTensor(y, dims, DimsProd(dims), false, "");
  new_tensor->AttrLNode(tensor, sigmoid_op);
  return new_tensor;
}



__global__ void tanh_forward_kernel(const float* inp, float* out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
        out[i] = tanhf(inp[i]);
}
__global__ void tanh_backward_kernel(float* dinp, const float* out, const float* dout, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        float x = (float)out[i];
        float local_grad = 1 - x*x;
        dinp[i] = (float)(local_grad * (float)dout[i]);
    }
}

void tanh_backward(const float* out, float B, float C, float* dinp, const float* dout) {
  float N = B * C;
  
  int grid_size, block_size, shared_mem_size; 
  std::vector<int> grid_block_mem_sizes = CalculateGridAndBlockSizes(N);
  grid_size = grid_block_mem_sizes[0];
  block_size = grid_block_mem_sizes[1];
  shared_mem_size = grid_block_mem_sizes[2];

  tanh_backward_kernel<<<grid_size, block_size>>>(dinp, out, dout, N);
  cudaCheck(cudaGetLastError());
}

extern "C" void *_tanh(Tensor *tensor)
{
  float *tensor_ptr = tensor->tensor_ptr;
  std::vector<float> dims = tensor->dims;
  

  float dims_prod = DimsProd(dims);

  int grid_size, block_size, shared_mem_size; 
  std::vector<int> grid_block_mem_sizes = CalculateGridAndBlockSizes(dims_prod);
  grid_size = grid_block_mem_sizes[0];
  block_size = grid_block_mem_sizes[1];
  shared_mem_size = grid_block_mem_sizes[2];

  std::vector<float> linear_layer_dims = format_LinearLayer_Dims(dims);
  

  float *y;
  cudaMalloc(&y, dims_prod*sizeof(float));
  cudaMemset(y, 0, dims_prod * sizeof(float));

  
  tanh_forward_kernel<<<grid_size, block_size>>>(tensor_ptr, y, dims_prod);
  cudaCheck(cudaGetLastError());

  
  int is_forward_func=1;
  
  

  Tensor *new_tensor = createTensor(y, dims, DimsProd(dims), false, "");
  new_tensor->AttrLNode(tensor, tanh_op);
  return new_tensor;
}

__global__ void relu_forward(float* Z, float* A,
                                      float N) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < N) {
        A[index] = fmaxf(Z[index], 0);
    }
}

extern "C" void *relu(Tensor *tensor)
{
  float *tensor_ptr = tensor->tensor_ptr;
  std::vector<float> dims = tensor->dims;
  std::vector<float> linear_layer_dims = format_LinearLayer_Dims(dims);

  float N = DimsProd(dims);
  float block_size = 32;

  float *y;
  cudaMalloc(&y, N*sizeof(float));
  cudaMemset(y, 0, N * sizeof(float));

  const int grid_size = ceil_div(N, block_size);
  relu_forward<<<grid_size, block_size>>>(tensor_ptr, y, N);



  Tensor *new_tensor = createTensor(y, dims, DimsProd(dims), false, "");
  new_tensor->AttrLNode(tensor, relu_op);
  return new_tensor;
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
    int warpId = threadIdx.x / 32; // warp index within a block //starred
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


extern "C" void *softmax(Tensor tensor)
{
  float *tensor_ptr = tensor.tensor_ptr;
  std::vector<float> dims = tensor.dims;
  
  dims =  format_LinearLayer_Dims(dims);

  int B = dims[0];
  int C = dims[1];

  float *probs;
  cudaMalloc(&probs, B*C*sizeof(float));
  cudaMemset(probs, 0, B*C*sizeof(float));

  int grid_size = B;
  int block_size = 32;
  size_t shared_mem_size = 2 * block_size / 32 * sizeof(float);
  softmax_forward_kernel4<<<grid_size, block_size, shared_mem_size>>>(tensor_ptr, probs, B, C);

  
  Tensor *new_tensor = createTensor(probs, dims, tensor.dims_prod, false, "");
  new_tensor->op=softmax_op;
  return new_tensor;
}




class BatchNorm2d
{
  cudnnTensorDescriptor_t input_desc, output_desc, scale_bias_mean_var_desc;

  public:
    float* scale=nullptr;
    float* bias=nullptr;
    float* running_mean=nullptr;
    float* running_var=nullptr;
    float* saved_mean=nullptr;
    float* saved_var=nullptr;
    float* dscale, dbias;
    int B = 0;
    int C;
    int H = 0;
    int W = 0;
    std::string Name;

    BatchNorm2d(int C, std::string Name)
        : C(C), Name(Name) {
      NamedTensorsT[Name] = new Tensor();
      NamedTensorsT[Name+"_bias"] = new Tensor();
    }

  
  void SetDescriptors(int, int, int);
  void InitMovingAverages();
  float *Forward(float *, int, int, int, int);
  void Backward(float *, float *, float *, float *, float *);

};

//global
static std::map<std::string, std::unique_ptr<BatchNorm2d>> NamedBatchNorm2d;


void BatchNorm2d::SetDescriptors(int H, int W, int B)
{
  cudnnTensorDescriptor_t input_desc;
  checkCUDNN(cudnnCreateTensorDescriptor(&input_desc));
  checkCUDNN(cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, B, C, H, W));
  this->input_desc = input_desc;

  cudnnTensorDescriptor_t output_desc;
  checkCUDNN(cudnnCreateTensorDescriptor(&output_desc));
  checkCUDNN(cudnnSetTensor4dDescriptor(output_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, B, C, H, W));
  this->output_desc = output_desc;

  cudnnTensorDescriptor_t scale_bias_mean_var_desc;
  checkCUDNN(cudnnCreateTensorDescriptor(&scale_bias_mean_var_desc));
  //checkCUDNN(cudnnSetTensor4dDescriptor(scale_bias_mean_var_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, C, 1, 1));
  checkCUDNN(cudnnDeriveBNTensorDescriptor(scale_bias_mean_var_desc, input_desc, CUDNN_BATCHNORM_SPATIAL_PERSISTENT));
  this->scale_bias_mean_var_desc = scale_bias_mean_var_desc;
}

void BatchNorm2d::InitMovingAverages()
{
  float *aux;

  aux = make_ones_float(C);
  cudaCheck(cudaMalloc(&scale, C*sizeof(float)));
  cudaCheck(cudaMemcpy(scale, aux, C*sizeof(float), cudaMemcpyHostToDevice));
  delete[] aux;
  
  aux = make_zeros_float(C);
  cudaCheck(cudaMalloc(&bias, C*sizeof(float)));
  cudaCheck(cudaMemcpy(bias, aux, C*sizeof(float), cudaMemcpyHostToDevice));
  delete[] aux;
  
  aux = make_zeros_float(C);
  cudaCheck(cudaMalloc(&running_mean, C*sizeof(float)));
  cudaCheck(cudaMemcpy(running_mean, aux, C*sizeof(float), cudaMemcpyHostToDevice));
  delete[] aux;
  
  aux = make_ones_float(C);
  cudaCheck(cudaMalloc(&running_var, C*sizeof(float)));
  cudaCheck(cudaMemcpy(running_var, aux, C*sizeof(float), cudaMemcpyHostToDevice));
  delete[] aux;
  
  aux = make_zeros_float(C);
  cudaCheck(cudaMalloc(&saved_mean, C*sizeof(float)));
  cudaCheck(cudaMemcpy(saved_mean, aux, C*sizeof(float), cudaMemcpyHostToDevice));
  delete[] aux;
  
  aux = make_ones_float(C);
  cudaCheck(cudaMalloc(&saved_var, C*sizeof(float)));
  cudaCheck(cudaMemcpy(saved_var, aux, C*sizeof(float), cudaMemcpyHostToDevice));
  delete[] aux;
}

float *BatchNorm2d::Forward(float *tensor, int H, int W, int B, int C)
{

  if (H != this->H || W != this->W || B != this->B)
    this->SetDescriptors(H, W, B);

  // Initialize weights.
  if (scale==nullptr)
    this->InitMovingAverages();


  // Forward
  float *output;
  float *aux = make_ones_float(B * H * W * C);
  cudaCheck(cudaMalloc(&output, B * H * W * C * sizeof(float)));
  cudaCheck(cudaMemcpy(output, aux, B * H * W * C * sizeof(float), cudaMemcpyHostToDevice));
  delete[] aux;
  
  constexpr float one = 1.0f;
  constexpr float zero = 0.0f;
  float gamma = 0.1f;
  float eps = 0.00001f;

  

  if(nn_mode==training_mode)
  {
    checkCUDNN(cudnnBatchNormalizationForwardTraining(
      cudnn,
      CUDNN_BATCHNORM_SPATIAL_PERSISTENT,
      &one,
      &zero,
      input_desc,
      tensor,
      output_desc,
      output,
      scale_bias_mean_var_desc,
      scale,
      bias,
      gamma,
      running_mean,
      running_var,
      eps,
      saved_mean,
      saved_var
    ));
  }
  else
  {
    checkCUDNN(cudnnBatchNormalizationForwardInference(
      cudnn,
      CUDNN_BATCHNORM_SPATIAL,
      &one,
      &zero,
      input_desc,
      tensor,
      output_desc,
      output,
      scale_bias_mean_var_desc,
      scale,
      bias,
      running_mean,
      running_var,
      eps
    ));
  }
  
  return output;
}


extern "C" void *BatchNormForward2d(char *self, Tensor *tensor, char *conv_namec, int is_obj_attr_or_self)
{
  //TODO: remove self arg and concatenate it instead during the function call
  //std::cout << "\nBatchNormForward2d " << conv_namec << " and tensor " << tensor.name << "\n";
  
  std::string _self = self;
  std::string conv_name = conv_namec;
  if (is_obj_attr_or_self)
    conv_name = _self + conv_name;

  //std::cout << "Conv forward for  conv: " << conv_name <<"\n";
  

  float *tensor_ptr, *output;
  tensor_ptr = tensor->tensor_ptr;
  std::vector<float> dims = tensor->dims;
  float input_dims_prod = DimsProd(dims);

  float B = dims[0];
  float C = dims[dims.size()-3];
  float H = dims[dims.size()-2];
  float W = dims[dims.size()-1];



  std::unique_ptr<BatchNorm2d> conv = std::move(NamedBatchNorm2d[conv_name]);

  if ((int)C!=(int)conv->C)
  {
    std::string error = "Input tensor channels are: " + std::to_string((int)C) + ", while the expected input channels of the BatchNorm2d are: " + std::to_string(conv->C);
    LogError(error);
    
    NamedBatchNorm2d[conv_name] = std::move(conv);
    return nullptr;
  }


  output = conv->Forward(tensor_ptr, H, W, B, C);

  float resultingDimsProd = B * (float)C * (float)H * (float)W;

  
  
  std::vector<float> bn_dims = {(float)C};
  std::string bias_name = conv_name+"_bias";

  Tensor *scale_bias_tensor, *scale_tensor, *bias_tensor;

  // for the backprop
  scale_bias_tensor = createTensor(conv->scale, bn_dims, C, true, conv_name);
  scale_bias_tensor->SetBias(conv->bias, C);
  scale_bias_tensor->weight=true;


  // for the optimizer only
  scale_tensor = NamedTensorsT[conv_name];
  scale_tensor->NewTensor(conv->scale, bn_dims, C, true, conv_name);
  scale_tensor->weight=true;
  
  bias_tensor = NamedTensorsT[bias_name];
  bias_tensor->NewTensor(conv->bias, bn_dims, C, true, conv_name);
  bias_tensor->weight=true;



  NamedBatchNorm2d[conv_name] = std::move(conv);

  std::vector<float> new_dims = {(float)B, (float)C, (float)H, (float)W};
  Tensor *new_tensor = createTensor(output, new_dims, DimsProd(new_dims), false, conv_name);
  new_tensor->AttrNodes(tensor, scale_bias_tensor, batchnorm2d);
  return new_tensor;
}



void BatchNorm2d::Backward(float *tensor, float *dx, float *dw, float *db, float *dy)
{
  constexpr float one = 1.0f;
  constexpr float zero = 0.0f;
  float eps = 0.00001f;
  
  
  checkCUDNN(cudnnBatchNormalizationBackward(
    cudnn,
    CUDNN_BATCHNORM_SPATIAL_PERSISTENT,
    &one,
    &zero,
    &one,
    &one,
    input_desc,
    tensor,
    output_desc,
    dy,
    input_desc,
    dx,
    scale_bias_mean_var_desc,
    scale,
    dw,
    db,
    eps,
    saved_mean,
    saved_var
  ));
}

void batchnormd2d_backward(float *inp, 
                     float *dinp, float *dw, float *db,
                     float *dout, std::string conv_name)
{

  //std::cout << "batchnorm2d_backward for " << conv_name << "\n";
  std::unique_ptr<BatchNorm2d> conv = std::move(NamedBatchNorm2d[conv_name]);

  conv->Backward(inp, dinp, dw, db, dout);

  NamedBatchNorm2d[conv_name] = std::move(conv);

  cudaCheck(cudaGetLastError());
}




class Conv2d
{
  // Forward
  cudnnTensorDescriptor_t input_desc, output_desc;
  cudnnFilterDescriptor_t filter_desc;
  cudnnConvolutionDescriptor_t conv_desc;
  cudnnConvolutionFwdAlgo_t fwd_algo;  

  // Weight backward grad
  cudnnConvolutionBwdFilterAlgo_t w_bwd_algo;
  // Input backward grad
  cudnnConvolutionBwdDataAlgo_t y_bwd_algo;

  std::size_t workspace_size, workspace_size_w_back, workspace_size_y_back;
  void *d_workspace, *d_workspace_w_back, *d_workspace_y_back;




  public:
    float* d_filter=nullptr;
    float* d_filter_g=nullptr;
    int C, OC, ks, stride, padding, out_H, out_W;
    int B = 0;
    int H = 0;
    int W = 0;
    std::string Init, Name;

    Conv2d(int C, int OC, int ks, int stride, int padding, std::string Init, std::string Name) 
        : C(C), OC(OC), ks(ks), stride(stride), padding(padding), Init(Init), Name(Name) {
      NamedTensorsT[Name] = new Tensor();
    }

  


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
        output_desc,
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
        output_desc,
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
        output_desc,
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
        output_desc,
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
      filter = make_random_float_uniform(ks*ks);


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
    output_desc, // output grad tensor descriptor
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
    output_desc, // output grad tensor descriptor
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




extern "C" void *ConvForward2d(char *self, Tensor *tensor, char *conv_namec, int is_obj_attr_or_self)
{
  //TODO: remove self arg and concatenate it instead during the function call
  //std::cout << "Conv forward of " << conv_namec << " and tensor " << tensor.name << "\n";
  
  std::string _self = self;
  std::string conv_name = conv_namec;
  if (is_obj_attr_or_self)
    conv_name = _self + conv_name;

  //std::cout << "Conv forward for  conv: " << conv_name <<"\n";
  

  float *tensor_ptr, *output, *d_filter;
  tensor_ptr = tensor->tensor_ptr;
  std::vector<float> dims = tensor->dims;
  float input_dims_prod = DimsProd(dims);

  float B = dims[0];
  float C = dims[dims.size()-3];
  float H = dims[dims.size()-2];
  float W = dims[dims.size()-1];



  std::unique_ptr<Conv2d> conv = std::move(NamedConv2d[conv_name]);

  if ((int)C!=(int)conv->C)
  {
    std::string error = "Input tensor channels are: " + std::to_string((int)C) + ", while the expected input channels of the convolution are: " + std::to_string(conv->C);
    LogError(error);
    
    NamedConv2d[conv_name] = std::move(conv);
    return nullptr;
  }

  output = conv->Forward(tensor_ptr, H, W, B);

  int ks_H = conv->ks;
  int ks_W = conv->ks;


  
  
  float resultingDimsProd = B * (float)conv->OC * (float)conv->out_H * (float)conv->out_W;

  int is_forward_func = 1;
  


  std::vector<float> new_dims = {(float)conv->B, (float)conv->OC, (float)conv->out_H, (float)conv->out_W};
  

  //for backprop:
  std::vector<float> kernel_dims = {(float)conv->OC, (float)C, (float)conv->ks, (float)conv->ks}; 




  Tensor *conv_tensor = NamedTensorsT[conv_name];
  conv_tensor->NewTensor(conv->d_filter, kernel_dims, DimsProd(kernel_dims), true, conv_name);
  conv_tensor->weight=true;
  

  //PrintTensorF(device_y, 2, 2);
  NamedConv2d[conv_name] = std::move(conv);

  Tensor *new_tensor = createTensor(output, new_dims, DimsProd(new_dims), false, "");
  new_tensor->AttrNodes(tensor, conv_tensor, conv2d);
  return new_tensor;
}





extern "C" float CreateConv2dOnDemand(char *tensor_name, char *init,
                                      float C, float OC, float ks, float stride, float padding)
{
  std::cout << "\nCreate conv on demand:\n   C: " << C << " OC " << OC << " ks " << ks << " stride " << stride << " padding " << padding << "\n";

  auto conv = std::make_unique<Conv2d>((int)C, (int)OC, (int)ks, (int)stride, (int)padding, init, tensor_name);

  std::cout << "Adding " << tensor_name << " to NamedConv2d dict\n";
  NamedConv2d[tensor_name] = std::move(conv);
  return 0;
}




extern "C" float CreateBatchNorm2dOnDemand(char *tensor_name, float C)
{
  std::cout << "\nCreate BatchNorm2d on demand:\n   C: " << C  << "\n";

  auto conv = std::make_unique<BatchNorm2d>((int)C, tensor_name);

  NamedBatchNorm2d[tensor_name] = std::move(conv);
  return 0;
}



class MaxPool2d
{
  // Forward
  cudnnTensorDescriptor_t input_desc;
  cudnnPoolingDescriptor_t pooling_desc;
  cudnnTensorDescriptor_t output_desc;





  public:
    std::string Type;
    int ks, stride, padding, out_H, out_W;
    int B = 0;
    int C = 0;
    int H = 0;
    int W = 0;

    MaxPool2d(int ks, int stride, int padding, std::string Type)
        : ks(ks), stride(stride), padding(padding), Type(Type) {}

  


  void SetDescriptors(int, int, int, int);
  float *Forward(float *, int, int, int, int);
  void Backward(float *, float *, float *, float *);

};
static std::map<std::string, std::unique_ptr<MaxPool2d>> NamedMaxPool2d;



void MaxPool2d::SetDescriptors(int H, int W, int B, int C)
{
  this->H = H;
  this->W = W;
  this->B = B;




  out_H = std::floor((H - ks + 2 * padding) / stride) + 1;
  out_W = std::floor((W - ks + 2 * padding) / stride) + 1;
  //std::cout << "Out H: " << out_H << " out W: " << out_W << "\n";



  // Initialize input tensor descriptor
  cudnnTensorDescriptor_t input_desc;
  checkCUDNN(cudnnCreateTensorDescriptor(&input_desc));
  checkCUDNN(cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, B, C, H, W));
  this->input_desc = input_desc;

  // Initialize pooling descriptor
  checkCUDNN(cudnnCreatePoolingDescriptor(&pooling_desc));
  checkCUDNN(cudnnSetPooling2dDescriptor(pooling_desc,            //descriptor handle
                                         (Type=="max") ? CUDNN_POOLING_MAX : CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING,       //mode - max pooling
                                         CUDNN_NOT_PROPAGATE_NAN, //NaN propagation mode
                                         ks,                       //window height
                                         ks,                       //window width
                                         padding,                       //vertical padding
                                         padding,                       //horizontal padding
                                         stride,                       //vertical stride
                                         stride));                     //horizontal stride
  
  // Initialize output tensor descriptor
  cudnnTensorDescriptor_t output_desc;
  checkCUDNN(cudnnCreateTensorDescriptor(&output_desc));
  checkCUDNN(cudnnSetTensor4dDescriptor(output_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, B, C, out_H, out_W));
  this->output_desc = output_desc;

}



float *MaxPool2d::Forward(float *tensor, int H, int W, int B, int C)
{
  // Initialize descriptors.
  //std::cout << "\nPool2d Forward with H: " << H << " W: " << W << "\n";
  //std::cout << "Type: " << Type << "\n";


  if (H != this->H || W != this->W || B != this->B || C != this->C)
    this->SetDescriptors(H, W, B, C);


  
  // Forward
  float *d_output;
  cudaCheck(cudaMalloc(&d_output, B * out_H * out_W * C * sizeof(float)));

  constexpr float one = 1.0f;
  constexpr float zero = 0.0f;

  checkCUDNN(cudnnPoolingForward(
        cudnn,
        pooling_desc,
        &one,
        input_desc,
        tensor,
        &zero,
        output_desc,
        d_output
    ));
  

  return d_output;
}


extern "C" void *MaxPoolForward2d(char *self, Tensor *tensor, char *conv_namec, int is_obj_attr_or_self)
{
  //std::cout << "MaxPoolForward2d of " << conv_namec << " and tensor " << tensor.name << "\n";
  
  std::string _self = self;
  std::string conv_name = conv_namec;
  if (is_obj_attr_or_self)
    conv_name = _self + conv_name;

  //std::cout << "Conv forward for  conv: " << conv_name <<"\n";
  

  float *tensor_ptr, *output, *d_filter;
  tensor_ptr = tensor->tensor_ptr;
  std::vector<float> dims = tensor->dims;
  float input_dims_prod = DimsProd(dims);

  float B = dims[0];
  float C = dims[dims.size()-3];
  float H = dims[dims.size()-2];
  float W = dims[dims.size()-1];
  float OC = C;


  std::unique_ptr<MaxPool2d> conv = std::move(NamedMaxPool2d[conv_name]);


  output = conv->Forward(tensor_ptr, H, W, B, C);

  int ks_H = conv->ks;
  int ks_W = conv->ks;
  

  
  
  float resultingDimsProd = B * (float)OC * (float)conv->out_W * (float)conv->out_W;



  std::vector<float> new_dims = {(float)B, (float)OC, (float)conv->out_H, (float)conv->out_W};
  

  NamedMaxPool2d[conv_name] = std::move(conv);

  Tensor *new_tensor = createTensor(output, new_dims, DimsProd(new_dims), false, conv_name);
  new_tensor->AttrLNode(tensor, maxpool2d);
  return new_tensor;
}


void MaxPool2d::Backward(float *tensor, float *out, float *dx, float *dy)
{
  //std::cout << "\nMaxPool2d Backward with H: " << H << " W: " << W << "\n";


  constexpr float one = 1.0f;
  constexpr float zero = 0.0f;


  // Backward to input
  checkCUDNN(cudnnPoolingBackward(
    cudnn,
    pooling_desc,
    &one,
    output_desc,
    out,
    output_desc, // output grad tensor descriptor
    dy,
    input_desc,
    tensor,
    &zero,
    input_desc, // input descriptor
    dx
  ));
  
}


void maxpool2d_backward(float *inp,  float *out,
                     float *dinp,
                     float *dout, std::string conv_name)
{
  //std::cout << "maxpool2d_backward of " << conv_name << "\n";
  std::unique_ptr<MaxPool2d> conv = std::move(NamedMaxPool2d[conv_name]);

  conv->Backward(inp, out, dinp, dout);

  NamedMaxPool2d[conv_name] = std::move(conv);

  cudaCheck(cudaGetLastError());
}




extern "C" float CreateMaxPool2dOnDemand(char *tensor_name, char *type, float ks, float stride, float padding)
{
  std::cout << "\nCreate maxpool2d on demand:\n" << "   ks " << ks << " stride " << stride << " padding " << padding << "\n";

  auto maxpool = std::make_unique<MaxPool2d>((int)ks, (int)stride, (int)padding, type);

  NamedMaxPool2d[tensor_name] = std::move(maxpool);
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
                          int B, int C, 
                          float *dloss)
{
  
  int grid_size = B;
  int block_size = 32;
  size_t shared_mem_size = 2 * block_size / 32 * sizeof(float);
  

  float *probs;
  cudaMalloc(&probs, B*C*sizeof(float));

  //int grid_size, block_size;
  //size_t shared_mem_size;
  std::vector<int> grid_block_mem_sizes = CalculateGridAndBlockSizes(B, 32);
  /*
  grid_size = grid_block_mem_sizes[0];
  block_size = grid_block_mem_sizes[1];
  shared_mem_size = grid_block_mem_sizes[2];
  */

  softmax_forward_kernel4<<<grid_size, block_size, shared_mem_size>>>(y_hat, probs, B, C);
  grid_block_mem_sizes = CalculateGridAndBlockSizes(B*C, 32);
  grid_size = grid_block_mem_sizes[0];
  block_size = grid_block_mem_sizes[1];
  shared_mem_size = grid_block_mem_sizes[2];

  
  crossentropy_softmax_backward_kernel1<<<grid_size, block_size>>>(dloss, probs, y, B, C);
  cudaFree(probs);

  cudaCheck(cudaGetLastError());
}



extern "C" float cross_entropy(Tensor *y_hat, Tensor *y)
{
  
  Tensor *loss_tensor = new Tensor();

  //std::cout << "cross_entropy y_hat leaf " << y_hat->leaf << " y leaf " << y->leaf << "\n";

  loss_tensor->AttrNodes(y_hat, y, cross_entropy_op);

  todo_backward_tensors.push_back(loss_tensor);

  

  return 0;
}



//




//




//





void safeCudaFree(void** ptr)
{
    if (ptr != nullptr && *ptr != nullptr)
    {
        cudaCheck(cudaFree(*ptr));
        *ptr = nullptr;
    }
}



std::map<std::string, float *> var_to_grad;
std::vector<float *> backprop_tensors_to_free;
std::vector<Tensor *> backprop_Tensors_to_free;

void to_free(float *tensor_ptr)
{
  if(!in_float_ptr_vec(tensor_ptr, backprop_tensors_to_free))
    backprop_tensors_to_free.push_back(tensor_ptr);
}
void to_free_tensor(Tensor *tensor_ptr)
{
  if(!in_tensor_ptr_vec(tensor_ptr, backprop_Tensors_to_free))
    backprop_Tensors_to_free.push_back(tensor_ptr);
}

void TraversePreOrder(Tensor *back_node, float *device_dy, bool from_gradless)
{
  if(back_node==nullptr)
    return;

  int op=back_node->op;
  std::string tensor_name, param_name, bias_name;
  float *w;
  float *device_dx, *device_dw;
  device_dx=nullptr;
  device_dw=nullptr;

  

  if(!in_int(op, gradless_ops) && !from_gradless)
  {

    if(device_dy==nullptr&&!in_int(op, loss_ops))
      LogErrorS("dy derivate is null at the backward mode.");

    //std::cout << "\nTraversing: " << back_node->name << ", op: " << back_node->op << ", leaf: " << back_node->leaf << ", weight: " << back_node->weight << "\n";


    if (back_node->weight) // dw is updated by pointer
      return;
    


    tensor_name = back_node->scopeless_name;
    if (back_node->leaf)
    {

      float dims_prod = back_node->dims_prod;
      
      //std::cout << "Accumulating grad of: " << tensor_name << "\n";

      if(var_to_grad.count(tensor_name)>0)
      {
        
        
        
        float *acc_y = var_to_grad[tensor_name];
        
        int grid_size, block_size, shared_mem_size;
        std::vector<int> grid_block_mem_sizes = CalculateGridAndBlockSizes(dims_prod);
        grid_size = grid_block_mem_sizes[0];
        block_size = grid_block_mem_sizes[1];

        
        add_inplace<<<grid_size, block_size>>>(acc_y, device_dy, dims_prod);
        to_free(device_dy);

      } else 
        var_to_grad[tensor_name] = device_dy;
      

      to_free(back_node->tensor_ptr);
      
      

      to_free_tensor(back_node);
      return;
    }

    int B, C, OC;
    float x_size, w_size, b_size;

    float *inp, *b, *out, *last_inp;
    float *dinp, *dw, *db, *device_db;
    device_dw=nullptr;
    device_db=nullptr;
    w=nullptr;
    b=nullptr;
    

    
    //std::cout << "Acquire info"  << "\n";

    tensor_name = back_node->L_Node->scopeless_name;

    inp = back_node->L_Node->tensor_ptr;
    x_size = back_node->L_Node->dims_prod;

    out = back_node->tensor_ptr;

    //std::cout << "Check null" << "\n";
    if(back_node->R_Node!=nullptr)
    {
      //std::cout << "not null " << "\n";
      param_name  = back_node->R_Node->name;
      w = back_node->R_Node->tensor_ptr;
      w_size = back_node->R_Node->dims_prod;

      b = back_node->R_Node->b;
      b_size = back_node->R_Node->b_size;
    }


    //std::cout << "malloc device w" << "\n";

    // weight gradient
    if(!in_int(op, loss_ops)&&back_node->R_Node!=nullptr)
    {
      //std::cout << "Is weight: " << back_node->R_Node->weight << "\n";
      if(back_node->R_Node->weight)
      {
        float *new_grad_ptr;
        if (w!=nullptr&&op!=hadamard_op&&op!=add_op)
        {
          dw = make_zeros_float(w_size);
          //std::cout << "weight of size " << w_size << "\n";
          if (NamedParamGrads[param_name]==nullptr)
          {
            cudaCheck(cudaMalloc(&new_grad_ptr, w_size*sizeof(float)));
            NamedParamGrads[param_name] = new_grad_ptr;
          } 
        
          device_dw = NamedParamGrads[param_name];
          cudaCheck(cudaMemcpy(device_dw, dw, w_size*sizeof(float), cudaMemcpyHostToDevice));
          delete[] dw;
        }

        if (b!=nullptr&&op!=hadamard_op&&op!=add_op)
        {
          db = make_zeros_float(b_size);
          bias_name = param_name+"_bias";

          if (NamedParamGrads[bias_name]==nullptr)
          {
            cudaCheck(cudaMalloc(&new_grad_ptr, b_size*sizeof(float)));
            NamedParamGrads[bias_name] = new_grad_ptr;
          }
          
          device_db = NamedParamGrads[bias_name];
          cudaCheck(cudaMemcpy(device_db, db, b_size*sizeof(float), cudaMemcpyHostToDevice));
          delete[] db;
        }
      } else {
        if(op!=add_op)
        {
          dw = make_zeros_float(w_size);
          cudaCheck(cudaMalloc(&device_dw, w_size*sizeof(float)));
          cudaCheck(cudaMemcpy(device_dw, dw, w_size*sizeof(float), cudaMemcpyHostToDevice));
          delete[] dw;
        }
      }
    }
    

    //std::cout << "malloc device_dx " << "\n";

    // input gradient
    if(op!=add_op)
    {   
      dinp = make_zeros_float(x_size);
      cudaCheck(cudaMalloc(&device_dx, x_size*sizeof(float)));
      cudaCheck(cudaMemcpy(device_dx, dinp, x_size*sizeof(float), cudaMemcpyHostToDevice));
      delete[] dinp;
    }

    //std::cout << "malloc done"  << "\n";


    B=0;
    C=0;
    OC=0;
    if(back_node->L_Node->dims.size()>0)
    {
      std::vector<float> BC = format_LinearLayer_Dims(back_node->L_Node->dims);
      B  = BC[0];
      C  = BC[1];
    }
    if (w!=nullptr)
      OC = back_node->R_Node->dims[0];
    


    //std::cout << "EXECUTING OP  " << op << "\n";
    switch (op)
    {
      case mult_op:
        matmul_backward(inp, w, B, C, OC, device_dx, device_dw, device_dy);
        break;
      case conv2d:
        conv2d_backward(inp, w, device_dx, device_dw, device_dy, param_name);
        break;
      case maxpool2d:
        maxpool2d_backward(inp, out, device_dx, device_dy, back_node->name);
        break;
      case batchnorm2d:
        batchnormd2d_backward(inp, device_dx, device_dw, device_db, device_dy, back_node->name);
        break;
      case relu_op:
        relu_backward(inp, B, C, device_dx, device_dy);
        break;
      case gelu_op:
        gelu_backward(inp, B, C, device_dx, device_dy);
        break;
      case sigmoid_op:
        sigmoid_backward(out, B, C, device_dx, device_dy);
        break;
      case tanh_op:
        tanh_backward(out, B, C, device_dx, device_dy);
        break;
      case cross_entropy_op:
        CrossEntropyBackward(inp, w, B, C, device_dx);
        break;
      case add_op:
        device_dx = device_dy;
        device_dw = device_dy;
        //cudaCheck(cudaMemcpy(device_dx, device_dy, x_size*sizeof(float), cudaMemcpyDeviceToDevice));
        //cudaCheck(cudaMemcpy(device_dw, device_dy, w_size*sizeof(float), cudaMemcpyDeviceToDevice));
        break;
      default:
        std::string _error = "The operation "+std::to_string(op)+" does not yet have the backward implementation";
        LogErrorS(_error);
        break;
    }
    
    cudaCheck(cudaGetLastError());
  } else
  {
    //std::cout << "\n\nFROM A GRADLESS OP" << "\n\n\n";
    from_gradless = true;
  }

  if (in_int(op, loss_ops))
  {
    //to_free(back_node->R_Node->tensor_ptr);
    delete back_node->R_Node;
    back_node->R_Node = nullptr;
  }


  // Garbage Collector on all lines below
  TraversePreOrder(back_node->L_Node, device_dx, from_gradless);
  TraversePreOrder(back_node->R_Node, device_dw, from_gradless);
  

  
  if(!in_int(op, loss_ops))
    to_free(back_node->tensor_ptr);

  if(!from_gradless)
    to_free(device_dy);

  to_free_tensor(back_node);
}


extern "C" float backprop()
{
  
  
  int op;
  

  std::string tensor_name;
  
  
  float *device_dy=nullptr;

  

  while(todo_backward_tensors.size()>0)
  {
    //std::cout << "\n\nbackprop:\n\n\n";
    Tensor *back_node = todo_backward_tensors.back();
    todo_backward_tensors.pop_back();

    op = back_node->op;
    
    
    if (op==attribution)
    {
      tensor_name = back_node->name;
      //std::cout << "backward attribution of " << tensor_name << "\n";
      device_dy = var_to_grad[tensor_name];
      //if (device_dy==nullptr)
      //  std::cout << "propagating null device_dy"  << "\n";
      var_to_grad.erase(tensor_name);
      
      back_node = back_node->R_Node;
    }

    
    TraversePreOrder(back_node, device_dy, false);
  }



  for(auto &pair : var_to_grad)
    cudaCheck(cudaFree(pair.second));

  for(float *tensor_ptr : backprop_tensors_to_free)
    cudaCheck(cudaFree(tensor_ptr));

  for(Tensor *tensor : backprop_Tensors_to_free)
    delete tensor;

  backprop_tensors_to_free.clear();
  backprop_Tensors_to_free.clear();
  var_to_grad.clear();
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
  //std::cout << "AdamW stepping: " << param_name << "\n";
  //PrintDims(dims);
  

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
      //if (!ends_with(param_name, "_bias"))
      //  PrintTensorF(pair.second, 2, 2);
      Tensor *tensor = NamedTensorsT[param_name];
      optimizer->init_states(param_name, tensor->dims_prod);
      optimizer->step(tensor->tensor_ptr, pair.second, tensor->dims, param_name);
      //std::cout << "DELETING " << param_name << "\n";
      //delete tensor;
    }
  }
  optimizer->count_step();



  return 0;
}




Value *BinaryTensorTensorExprAST::codegen(Value *first_arg, Value *scope_str, Value *previous_scope) {
  if (not ShallCodegen)
    return ConstantFP::get(*TheContext, APFloat(0.0f));

  Value *LtensorName = Builder->CreateGlobalString(LHS->GetName());
  Value *RtensorName = Builder->CreateGlobalString(RHS->GetName());
  Value *object_name;


  
  // if is attribution
  if (Op == '=') {
  
    seen_var_attr=true;

    Value *RtensorPtr = RHS->codegen(first_arg, scope_str, previous_scope);
    

    if (!LHS->GetIsVec())
    {
      VariableExprAST *LHSE = static_cast<VariableExprAST *>(LHS.get());
      LtensorName = LHSE->NameSolver->codegen(first_arg, scope_str, previous_scope);

      if (!LHSE)
        return LogErrorV("'=' left side expression must be a var.");
      
      std::cout << "1 1 attr\n";


      Builder->CreateCall(TheModule->getFunction("AttrTensor"),
                          {LtensorName, RtensorPtr});
      std::cout << "Post attr call\n\n";
    } else
    {
      std::cout << "1 1 INDEXED attr\n";

      VecIdxExprAST *LHSE = static_cast<VecIdxExprAST *>(LHS.get());
      LtensorName = LHSE->NameSolver->codegen(first_arg, scope_str, previous_scope);
      if (!LHSE)
        return LogErrorV("'=' left side expression must be a var.");

      if(LHSE->Idx[0]->GetType()!="tensor")
      {
        std::vector<Value *> idx_calc_args;
        idx_calc_args.push_back(LtensorName);
        for (int i=0; i<LHSE->Idx.size(); i++)
          idx_calc_args.push_back(LHSE->Idx[i]->codegen(first_arg, scope_str, previous_scope));
        Value *idx_at = Builder->CreateCall(TheModule->getFunction("CalculateIdxOffset"),
                              idx_calc_args);

        Builder->CreateCall(TheModule->getFunction("AttrTensorOnIdx"),
                            {LtensorName, RtensorPtr,
                            idx_at});
      } else {
        VariableExprAST *idx = static_cast<VariableExprAST *>(LHSE->Idx[0].get());
        Value *idx_tensor_name = idx->NameSolver->codegen(first_arg, scope_str, previous_scope);
        
        Builder->CreateCall(TheModule->getFunction("AttrTensorOnIdxTensor"), {LtensorName, idx_tensor_name, RtensorPtr});

      }
    }

    seen_var_attr=false;
    return ConstantFP::get(*TheContext, APFloat(0.0f));
  }




  std::string functionName = Builder->GetInsertBlock()->getParent()->getName().str();
  std::cout << "\nTensor Tensor for function: " << functionName << "\n";
  int forward_func = 0;
  if(ends_with(functionName, "forward"))
    forward_func = 1;
  forward_func = 1; // TODO: RemoveLastDim this line



  
  Value *LtensorPtr = LHS->codegen(first_arg, scope_str, previous_scope);
  Value *RtensorPtr = RHS->codegen(first_arg, scope_str, previous_scope);



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
    return Builder->CreateCall(TheModule->getFunction("CudaMult"),
                                    {is_forward_func,
                                     LtensorPtr, RtensorPtr});
  case '/':
  {
    return Builder->CreateCall(TheModule->getFunction("CudaDiv"),
                                    {is_forward_func,
                                     LtensorPtr, RtensorPtr});
  }
  case '+':
    return Builder->CreateCall(TheModule->getFunction("CudaAdd"),
                                    {is_forward_func,
                                     LtensorPtr, RtensorPtr});
  case '*':
    return Builder->CreateCall(TheModule->getFunction("CudaHadamard"),
                                    {is_forward_func,
                                     LtensorPtr, RtensorPtr});
  case '-':
    return Builder->CreateCall(TheModule->getFunction("CudaSub"),
                                    {is_forward_func,
                                     LtensorPtr, RtensorPtr});
  case tok_equal:
    return Builder->CreateCall(TheModule->getFunction("CudaEqual"),
                               {is_forward_func, LtensorPtr, RtensorPtr}, "cudaequal");
  case ':':
    return LtensorPtr;
  default:
    break;
  }
  

  std::string _error = "The operator " + ReverseToken(Op) + " is not implemented for operations between tensors";
  LogErrorS(_error);
  
  Function *F = getFunction(std::string("binary") + Op);
  assert(F && "Operator not found.");

  Value *Ops[] = {LtensorName, RtensorName};
  return Builder->CreateCall(F, Ops, "binop");
}




Value *BinaryTensorPinnedExprAST::codegen(Value *first_arg, Value *scope_str, Value *previous_scope) {
  std::cout << "Binary Tensor Pinned codegen" << "\n";

  if (not ShallCodegen)
    return ConstantFP::get(*TheContext, APFloat(0.0f));

  std::cout << "Binary Tensor Pinned codegen" << "\n";

  Value *LtensorName = Builder->CreateGlobalString(LHS->GetName());
  Value *object_name;



  // if is attribution
  if (Op == '=') {
  
    seen_var_attr=true;

    Value *RtensorPtr = RHS->codegen(first_arg, scope_str, previous_scope);
    

    if (!LHS->GetIsVec())
    {
      VariableExprAST *LHSE = static_cast<VariableExprAST *>(LHS.get());
      LtensorName = LHSE->NameSolver->codegen(first_arg, scope_str, previous_scope);

      if (!LHSE)
        return LogErrorV("'=' left side expression must be a var.");
      std::cout << "1 2 attr\n";
      
      

      Builder->CreateCall(TheModule->getFunction("AttrTensorNoFree"),
                          {LtensorName, RtensorPtr});
      std::cout << "Post attr call\n";
    } else
    {
      std::cout << "1 2 INDEXED attr\n";

      VecIdxExprAST *LHSE = static_cast<VecIdxExprAST *>(LHS.get());
      LtensorName = LHSE->NameSolver->codegen(first_arg, scope_str, previous_scope);
      if (!LHSE)
        return LogErrorV("'=' left side expression must be a var.");


      std::vector<Value *> idx_calc_args;
      idx_calc_args.push_back(LtensorName);
      for (int i=0; i<LHSE->Idx.size(); i++)
        idx_calc_args.push_back(LHSE->Idx[i]->codegen(first_arg, scope_str, previous_scope));
      Value *idx_at = Builder->CreateCall(TheModule->getFunction("CalculateIdxOffset"),
                            idx_calc_args);

      
      Builder->CreateCall(TheModule->getFunction("AttrTensorOnIdx"),
                          {LtensorName, RtensorPtr,
                           idx_at});
      
    }
    seen_var_attr=false;
    return ConstantFP::get(*TheContext, APFloat(0.0f));
  }
  
}





Value *BinaryObjExprAST::codegen(Value *first_arg, Value *scope_str, Value *previous_scope) {
  if (not ShallCodegen)
    return ConstantFP::get(*TheContext, APFloat(0.0f));


  Value *LName = Builder->CreateGlobalString(LHS->GetName());
  Value *RName = Builder->CreateGlobalString(RHS->GetName());
  Value *object_name;

  


  // if is attribution
  if (Op == '=') {
  
    seen_var_attr=true;


    if (!LHS->GetIsVec())
    {
      std::cout << "\n\n3 3 attr\n";
      VariableExprAST *LHSE = static_cast<VariableExprAST *>(LHS.get());
      if (!LHSE)
        return LogErrorV("'=' object attribution destiny must be an object variable.");
      LName = LHSE->NameSolver->codegen(first_arg, scope_str, previous_scope);
      
      if (RHS->GetIsVec())
      {
        std::cout << "3 3 other INDEXED of RHS->GetIsVec() && RHS->GetType()==object" << "\n";
        VecIdxExprAST *RHSE = static_cast<VecIdxExprAST *>(RHS.get());
        RName = RHSE->NameSolver->codegen(first_arg, scope_str, previous_scope);
        
        Builder->CreateCall(TheModule->getFunction("objAttr_var_from_vec"),
                                                        {LName, RName});
      } else {
        VariableExprAST *RHSE = static_cast<VariableExprAST *>(RHS.get());
        RName = RHSE->NameSolver->codegen(first_arg, scope_str, previous_scope);
        
        Builder->CreateCall(TheModule->getFunction("objAttr_var_from_var"),
                                                        {LName, RName});

      }
    
    } else {
      std::cout << "\n\n3 3 other INDEXED attr\n";
      VecIdxExprAST *LHSE = static_cast<VecIdxExprAST *>(LHS.get());
      if (!LHSE)
        return LogErrorV("'=' object attribution destiny must be an object variable.");
      LName = LHSE->NameSolver->codegen(first_arg, scope_str, previous_scope);


      std::cout << "ok" << "\n";
      
      if (RHS->GetIsVec())
      {
        std::cout << "3 3 other INDEXED of RHS->GetIsVec() && RHS->GetType()==object" << "\n";
        VecIdxExprAST *RHSE = static_cast<VecIdxExprAST *>(RHS.get());
        RName = RHSE->NameSolver->codegen(first_arg, scope_str, previous_scope);
        
        Builder->CreateCall(TheModule->getFunction("objAttr_vec_from_vec"),
                                                        {LName, RName});
      } else {
        std::cout << "3 3 VEC FROM VAR" << "\n";
        VariableExprAST *RHSE = static_cast<VariableExprAST *>(RHS.get());
        RName = RHSE->NameSolver->codegen(first_arg, scope_str, previous_scope);
        
        Builder->CreateCall(TheModule->getFunction("objAttr_vec_from_var"),
                                                        {LName, RName});

      }


    }
    seen_var_attr=false;
    return ConstantFP::get(*TheContext, APFloat(0.0f));
  }
  
}







Value *BinaryExprAST::codegen(Value *first_arg, Value *scope_str, Value *previous_scope) {
  if (not ShallCodegen)
    return ConstantFP::get(*TheContext, APFloat(0.0f));
  // Special case '=' because we don't want to emit the LHS as an expression.
  if (Op == '=') {

    //std::cout << "\n0 0 ATTRIBUTION" << "\n\n\n";

    seen_var_attr=true;
    // Assignment requires the LHS to be an identifier.
    // This assume we're building without RTTI because LLVM builds that way by
    // default.  If you build LLVM with RTTI this can be changed to a
    // dynamic_cast for automatic error checking.
    VariableExprAST *LHSE = static_cast<VariableExprAST *>(LHS.get());
    Value *Lvar_name = LHSE->NameSolver->codegen(first_arg, scope_str, previous_scope);


    NameSolverAST *name_solver = static_cast<NameSolverAST *>(LHSE->NameSolver.get());
    std::string Lname = std::get<0>(name_solver->Names[0]);
    std::string LType = LHS->GetType();


    if (!LHSE)
      return LogErrorV("'=' destiny must be a variable.");
    // Codegen the RHS.
    
    Value *Val = RHS->codegen(first_arg, scope_str, previous_scope);

    if (!Val)
    {
      seen_var_attr=false;
      return nullptr;
    }

    // Look up the name.
    if (LType=="float") {
      Builder->CreateCall(TheModule->getFunction("StoreOnDemand"),
                                                  {Lvar_name,
                                                   Val});

    } else if (LType=="str") {


      Builder->CreateCall(TheModule->getFunction("StoreStrOnDemand"),
                                                  {Lvar_name,
                                                   Val});
                                                   

    } else if (LType=="str_vec") {

      //std::cout << "ATTRIBUTING TO STRING VEC: " << Lname << "\n";
      Value *Variable = NamedStrVecs[Lname];
      
      if(LHS->GetSelf())
        Builder->CreateCall(TheModule->getFunction("StoreStrVecOnDemand"),
                                                  {Lvar_name,
                                                   Val});
      else
        Builder->CreateStore(Val, Variable);

    } else if (LType=="float_vec") {

      //std::cout << "ATTRIBUTING TO FLOAT VEC: " << Lname << ", type: " << Type << ", is vec: " << LHS->GetIsVec() << "\n";

      Value *Variable = NamedFloatVecs[Lname];
      

      if(LHS->GetSelf())
      {
        if(LHS->GetIsVec())
        {
          VecIdxExprAST *LHSV = static_cast<VecIdxExprAST *>(LHS.get());
          

          Builder->CreateCall(TheModule->getFunction("StoreFloatVecOnDemandOnIdx"),
                                                  {Lvar_name,
                                                   LHSV->Idx[0]->codegen(first_arg, scope_str, previous_scope),
                                                   Val});

        } else
          Builder->CreateCall(TheModule->getFunction("StoreFloatVecOnDemand"),
                                                  {Lvar_name,
                                                   Val});
      }
      else
      {
        if(LHS->GetIsVec())
        {
          // TODO: Implement non-object-attr float vector index attribution.
        } else
          Builder->CreateStore(Val, Variable);
      }
        

    } else {
      
      seen_var_attr=false;
      
      
      Builder->CreateCall(TheModule->getFunction("StoreOnDemand"),
                                                  {Lvar_name,
                                                   Val});
      

      //std::string _error = "Could not find variable " + Lname + ".";
      //return LogErrorV(_error);
    }

    seen_var_attr=false;
    return Val;
  }


  

  Value *L = LHS->codegen(first_arg, scope_str, previous_scope);
  Value *R = RHS->codegen(first_arg, scope_str, previous_scope);
  
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
    case tok_equal:
      L = Builder->CreateFCmpUEQ(L, R, "cmptmp");
      // Convert bool 0/1 to float 0.0 or 1.0
      return Builder->CreateUIToFP(L, Type::getFloatTy(*TheContext), "booltmp");
    case tok_diff:
      L = Builder->CreateFCmpUNE(L, R, "cmptmp");
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


Value *UnaryExprAST::codegen(Value *first_arg, Value *scope_str, Value *previous_scope) {
  if (not ShallCodegen)
    return ConstantFP::get(*TheContext, APFloat(0.0f));
  Value *OperandV = Operand->codegen(first_arg, scope_str, previous_scope);
  if (!OperandV)
    return nullptr;
  
  
  
  //std::cout << "Operand type: " << Operand->GetType();
  if (Opcode=='-')
  {
    //std::cout << "\n\n\n\n\n\nIT'S A MINUS " << Operand->GetType() << "\n\n\n\n\n\n\n";
    if (Operand->GetType()=="tensor")
    {
      Value *tensor_name = Builder->CreateGlobalString(Operand->GetName());

      std::string pre_dot = Operand->GetPreDot();
      bool is_self = Operand->GetSelf();
      bool is_attr = Operand->GetIsAttribute();

      if (is_attr) { // Gets from pre_dot if it is a class attribute
        Value * object_name = Builder->CreateGlobalString(pre_dot);

        tensor_name = Builder->CreateCall(TheModule->getFunction("ConcatStr"),
                                                          {object_name, tensor_name});
      }
      if (is_self)
        tensor_name = Builder->CreateCall(TheModule->getFunction("ConcatStr"),
                                                          {first_arg, tensor_name});
      if (!(is_self||is_attr))
        tensor_name = Builder->CreateCall(TheModule->getFunction("ConcatStr"),
                                                {scope_str, tensor_name});
        

      Value *tensorPtr = Builder->CreateCall(TheModule->getFunction("LoadTensor"),
                                              {tensor_name});
      Value *R = ConstantFP::get(Type::getFloatTy(*TheContext), -1);

      return Builder->CreateCall(TheModule->getFunction("CudaScalarMult"),
                                {tensorPtr, R}, "cudascalarmult");
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


Value *IfExprAST::codegen(Value *first_arg, Value *scope_str, Value *previous_scope) {
  if (not ShallCodegen)
    return ConstantFP::get(*TheContext, APFloat(0.0f));


  Value *CondV = Cond->codegen(first_arg, scope_str, previous_scope);
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
    ThenV = then_body->codegen(first_arg, scope_str, previous_scope);
  

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
    ElseV = else_body->codegen(first_arg, scope_str, previous_scope);

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

Value *ForExprAST::codegen(Value *first_arg, Value *scope_str, Value *previous_scope) {
  if (not ShallCodegen)
    return ConstantFP::get(*TheContext, APFloat(0.0f));
  Function *TheFunction = Builder->GetInsertBlock()->getParent();

  // Create an alloca for the variable in the entry block.
  AllocaInst *Alloca = CreateEntryBlockAlloca(TheFunction, VarName);

  // Emit the start code first, without 'variable' in scope.
  Value *StartVal = Start->codegen(first_arg, scope_str, previous_scope);
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
    StepVal = Step->codegen(first_arg, scope_str, previous_scope);
    if (!StepVal)
      return nullptr;
  } 


  // Compute the end condition.
  Value *EndCond = End->codegen(first_arg, scope_str, previous_scope);
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
    body->codegen(first_arg, scope_str, previous_scope);

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



Value *WhileExprAST::codegen(Value *first_arg, Value *scope_str, Value *previous_scope) {
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
  Value* condVal = Cond->codegen(first_arg, scope_str, previous_scope);
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
    body->codegen(first_arg, scope_str, previous_scope);

  // After the loop body, go back to the condition check
  Builder->CreateBr(CondBB);

  // Insert the after loop block
  Builder->SetInsertPoint(AfterBB);

  return Constant::getNullValue(Type::getFloatTy(*TheContext));
}



Function *codegenAsyncFunction(std::vector<std::unique_ptr<ExprAST>> &asyncBody, Value *first_arg, Value *scope_str, Value *previous_scope) {
  

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
  std::cout << "\n\nfunction * get basic block for function: " << functionName << "\n";
  BasicBlock *BB = BasicBlock::Create(*TheContext, "async_bb", asyncFun);
  Builder->SetInsertPoint(BB);
  

  // define body of function
  Value *V;

  for (auto &body : asyncBody)
    V = body->codegen(first_arg, scope_str, previous_scope);



  if (V)
  {
    
    std::cout << "create return" << "\n";
    Builder->CreateRet(Constant::getNullValue(int8PtrTy));
    

    std::string functionError;
    llvm::raw_string_ostream functionErrorStream(functionError);

    if (verifyFunction(*asyncFun, &functionErrorStream)) {
      functionErrorStream.flush();
      llvm::errs() << "Function verification failed:\n" << functionError << "\n";
    } 

    verifyModule(*TheModule);
    return asyncFun;
  }
  
  std::cout << "ERASING ASYNC FROM PARENT" << "\n";
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
  
  std::cout << "Creating thread" << "\n";
  pthread_create(thread, attr, start_routine, arg);
  std::cout << "Created" << "\n";

  //pthread_create(&t, attr, start_routine, arg);

  //pthread_join(t, nullptr);

  //std::cout << "Join\n";
}


extern "C" void pthread_join_aux(pthread_t thread)
{
  std::cout << "Joining " << thread <<  "\n";
  void **value_ptr;
  value_ptr = nullptr;

  pthread_join(thread, value_ptr);
  std::cout << "Joined: " << thread << "\n";
}



std::vector<Value *> thread_pointers;


Value *AsyncExprAST::codegen(Value *first_arg, Value *scope_str, Value *previous_scope) {
  if (not ShallCodegen)
    return ConstantFP::get(*TheContext, APFloat(0.0f));

  
  // Create/Spawn Threads

  BasicBlock *CurrentBB = Builder->GetInsertBlock();

  /* 
  Function *TheFunction = Builder->GetInsertBlock()->getParent();
  std::string functionName = TheFunction->getName().str();
  BasicBlock *CurrentBB = BasicBlock::Create(*TheContext, "loop", TheFunction);
  Builder->CreateBr(CurrentBB);
  */
  
  
  
  //std::cout << "\nAsync get insert block for function: " << functionName << "\n\n";


  Function *asyncFun = codegenAsyncFunction(std::ref(Body), first_arg, scope_str, previous_scope);


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
  
  std::cout << "Created join call" << "\n";
  
  /*
  BasicBlock *PostAsyncBB = BasicBlock::Create(*TheContext, "postasync", TheFunction);
  Builder->CreateBr(PostAsyncBB); // Branch from the async block to the post async block
  Builder->SetInsertPoint(PostAsyncBB);
  */

  thread_pointers.push_back(pthreadPtr);

  return pthreadPtr;
}



Value *FinishExprAST::codegen(Value *first_arg, Value *scope_str, Value *previous_scope) {
  if (not ShallCodegen)
    return ConstantFP::get(*TheContext, APFloat(0.0f));

  
  Function *TheFunction = Builder->GetInsertBlock()->getParent();
  std::string functionName = TheFunction->getName().str();


  //BasicBlock *FinishBB = BasicBlock::Create(*TheContext, "loop", TheFunction);
  //BasicBlock *AfterBB = BasicBlock::Create(*TheContext, "afterbb", TheFunction);



  //Builder->CreateBr(FinishBB);
  //Builder->SetInsertPoint(FinishBB);

  std::cout << "\n\nFinish codegen for: " << functionName <<  "\n";


  //std::vector<Value *> thread_pointers;
  

  for (int i=0; i < Bodies.size(); i++)
  {

    /*
    if (IsAsync[i])
      thread_pointers.push_back(Bodies[i]->codegen(first_arg, scope_str, previous_scope));
    else
      Bodies[i]->codegen(first_arg, scope_str, previous_scope);
    */
    Bodies[i]->codegen(first_arg, scope_str, previous_scope);
  }

  
  //Builder->CreateBr(AfterBB);
  //Builder->SetInsertPoint(AfterBB);


  PointerType *pthreadTy = Type::getInt8Ty(*GlobalContext)->getPointerTo();

  Function *pthread_join = TheModule->getFunction("pthread_join_aux");


  std::cout << "\n\n\n\nFINISH HAS " << thread_pointers.size() << " ASYNC EXPRESSIONS "  << "\n\n\n\n\n";


  for (Value *pthreadPtr : thread_pointers)
  {
    Value *pthread = Builder->CreateLoad(pthreadTy, pthreadPtr);

    Builder->CreateCall(pthread_join,
                        {pthread});
    
  }
  
  thread_pointers.clear();
  
  return ConstantFP::get(*TheContext, APFloat(0.0f));
}


Value *LockExprAST::codegen(Value *first_arg, Value *scope_str, Value *previous_scope){
  
  Builder->CreateCall(TheModule->getFunction("LockMutex"), {Builder->CreateGlobalString(Name)});

  Value *V;
  for (auto &body : Bodies)
    V = body->codegen(first_arg, scope_str, previous_scope);

  return V;
}

Value *UnlockExprAST::codegen(Value *first_arg, Value *scope_str, Value *previous_scope){
  Builder->CreateCall(TheModule->getFunction("UnlockMutex"), {Builder->CreateGlobalString(Name)});
  return ConstantFP::get(*TheContext, APFloat(0.0f));
}




Value *ReturnExprAST::codegen(Value *first_arg, Value *scope_str, Value *previous_scope) {

  for (int i=0; i<Destiny.size(); i++)
  {
    //TODO: add self and attr to return
    
    std::string name, type, l_name, l_type;
    bool is_vec, l_is_vec;

    name   = Destiny[i]->GetName();
    type   = Destiny[i]->GetType();
    is_vec = Destiny[i]->GetIsVec();

    Value *_name = Builder->CreateGlobalString(name);

    std::cout << "\nRETURNING: " << name << ", type: " << type << ", is vec: " << is_vec <<  "\n\n";

    if (!IsAs[i])
    {
      if(type=="tensor")
      {
        VariableExprAST *destiny = static_cast<VariableExprAST *>(Destiny[i].get());
        destiny->NameSolver->SetSolverIncludeScope(false);
        _name = destiny->NameSolver->codegen(first_arg, scope_str, previous_scope);

        Builder->CreateCall(TheModule->getFunction("RemoveTensorScope"),
                                            {_name, scope_str,
                                             _name, previous_scope});
      }
    } else {
      l_name   = Vars[i]->GetName();
      l_type   = Vars[i]->GetType();
      l_is_vec = Vars[i]->GetIsVec();

      std::cout << "l_name: " << l_name << " l_type: " << l_type << ", l_is_vec: " << l_is_vec << "\n";

      if (!is_vec)
      {
        VariableExprAST *destiny = static_cast<VariableExprAST *>(Destiny[i].get());
        destiny->NameSolver->SetSolverIncludeScope(false);
        _name = destiny->NameSolver->codegen(first_arg, scope_str, previous_scope);

        
        VariableExprAST *var = static_cast<VariableExprAST *>(Vars[i].get());
        var->NameSolver->SetSolverIncludeScope(false);
        Value *_l_name = var->NameSolver->codegen(first_arg, scope_str, previous_scope);

        if (l_type=="tensor"||type=="tensor")
        {
          Builder->CreateCall(TheModule->getFunction("RemoveTensorScope"),
                                              {_l_name, scope_str,
                                               _name,   previous_scope});
        }
      } else {

        VecIdxExprAST *destiny = static_cast<VecIdxExprAST *>(Destiny[i].get());
        if (!destiny)
          return LogErrorV("Could not deal with return expression");
        destiny->NameSolver->SetSolverIncludeScope(false);
        _name = destiny->NameSolver->codegen(first_arg, scope_str, previous_scope);
        

        std::vector<Value *> idx_calc_args;
        idx_calc_args.push_back(Builder->CreateCall(TheModule->getFunction("ConcatStr"),
                                                      {previous_scope, _name}));
        for (int i=0; i<destiny->Idx.size(); i++)
          idx_calc_args.push_back(destiny->Idx[i]->codegen(first_arg, scope_str, previous_scope));
        Value *idx_at = Builder->CreateCall(TheModule->getFunction("CalculateIdxOffset"),
                              idx_calc_args);

        
        Value *_l_name = Builder->CreateGlobalString(l_name);
        Builder->CreateCall(TheModule->getFunction("RemoveTensorScopeAttrOnIndex"),
                                              {_l_name, scope_str,
                                               _name, previous_scope,
                                               idx_at});
      }
    }
  }

  return ConstantFP::get(*TheContext, APFloat(0.0));
}




// Create Var
Value *VarExprAST::codegen(Value *first_arg, Value *scope_str, Value *previous_scope) {
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
      InitVal = Init->codegen(first_arg, scope_str, previous_scope);
      if (!InitVal)
        return nullptr;
    } else { // If not specified, use 0.0.
      InitVal = ConstantFP::get(*TheContext, APFloat(0.0));
    }



    Value *var_name = Builder->CreateGlobalString(VarName);
      var_name = Builder->CreateCall(TheModule->getFunction("ConcatStr"),
                                        {scope_str, var_name});

    Builder->CreateCall(TheModule->getFunction("StoreOnDemand"),
                                                  {var_name, InitVal});
    
  }


  return ConstantFP::get(*TheContext, APFloat(0.0));
}





Value *StrExprAST::codegen(Value *first_arg, Value *scope_str, Value *previous_scope) {
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
      InitVal = Init->codegen(first_arg, scope_str, previous_scope);
      if (!InitVal)
        return nullptr;
    } else { // If not specified, use 0.0.
      InitVal = Builder->CreateGlobalString("");
    }



      
    // Remember the old variable binding so that we can restore the binding when
    // we unrecurse.
    
    Value *var_name, *scopeless_name;
    var_name = Builder->CreateCall(TheModule->getFunction("CopyString"),
                                            {Builder->CreateGlobalString(VarName)});
    

    bool is_self = GetSelf();
    bool is_attr = GetIsAttribute();

    if (is_self||is_attr)
      var_name = Builder->CreateCall(TheModule->getFunction("ConcatStrFreeRight"),
                                            {first_arg, var_name});
    //scopeless_name = Builder->CreateCall(TheModule->getFunction("CopyString"),
    //                                        {var_name});
    if (!(is_self||is_attr))
      var_name = Builder->CreateCall(TheModule->getFunction("ConcatStrFreeRight"),
                                            {scope_str, var_name});
    

    
    Builder->CreateCall(TheModule->getFunction("AddToScopeCleanList"),
                        {scope_str,
                         Builder->CreateCall(TheModule->getFunction("CopyString"), {var_name}),
                         Builder->CreateGlobalString("str") //stack?
                        });

                        
    Builder->CreateCall(TheModule->getFunction("StoreStrOnDemand"),
                                                  {var_name,
                                                   InitVal});
  }

  
  return ConstantFP::get(*TheContext, APFloat(0.0f));
}


Value *StrVecExprAST::codegen(Value *first_arg, Value *scope_str, Value *previous_scope) {
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
      InitVal = Init->codegen(first_arg, scope_str, previous_scope);
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


extern "C" float InitObjectVecWithNull(char *name, float vec_size)
{
  //std::cout << "InitObjectVecWithNull of " << name << " with vec_size " << vec_size << "\n\n\n\n";

  for (int i=0; i<vec_size; i++)
  {
    std::string indexed_name = name + std::to_string(i);
    objectVecs[indexed_name] = "nullptr";
  }
    
  return 0;
}

extern "C" float is_null(char *name)
{
  //std::cout << "\n\nIS NULL OF: " << name << "\n\n\n";

  if (objectVecs[name]=="nullptr")
    return 1;
  return 0;
}


Value *ObjectExprAST::codegen(Value *first_arg, Value *scope_str, Value *previous_scope) {
  if (not ShallCodegen)
    return ConstantFP::get(*TheContext, APFloat(0.0f));


  Function *TheFunction = Builder->GetInsertBlock()->getParent();

  Value *init;
  if (Init)
    init = Init->codegen(first_arg, scope_str, previous_scope);

  // Register all variables and emit their initializer.

  for (unsigned i = 0, e = VarNames.size(); i != e; ++i) {
    const std::string &VarName = VarNames[i].first;

    Value *var_name = Builder->CreateGlobalString(VarName);
    
    if (!GetIsVec())
    {
      Builder->CreateCall(TheModule->getFunction("InstantiateObject"),
                                              {scope_str, var_name});
    }
    else if (Init) // init of vec[size]
    {
      std::string pre_dot = GetPreDot();
      bool is_self = GetSelf();
      bool is_attr = GetIsAttribute();

      if (is_self||is_attr)
        var_name = Builder->CreateCall(TheModule->getFunction("ConcatStr"),
                                              {first_arg, var_name});
      if (!(is_self||is_attr))
        var_name = Builder->CreateCall(TheModule->getFunction("ConcatStr"),
                                              {scope_str, var_name});

      //Value *object_hash = Builder->CreateCall(TheModule->getFunction("RandomStrOnDemand"), {});
      //var_name = Builder->CreateCall(TheModule->getFunction("ConcatStr"),
      //                                        {object_hash, var_name});


      Builder->CreateCall(TheModule->getFunction("InitObjectVecWithNull"),
                                                {var_name, init});
    } else
    {}
  }


  return ConstantFP::get(*TheContext, APFloat(0.0));
}




extern "C" void *rand_like(Tensor tensor)
{
  float dims_prod = tensor.dims_prod;

  float *tensor_ptr, *tensor_cpu;

  tensor_cpu = make_random_float_uniform(dims_prod);

  cudaMalloc(&tensor_ptr, dims_prod*sizeof(float));
  cudaMemcpy(tensor_ptr, tensor_cpu, dims_prod*sizeof(float), cudaMemcpyHostToDevice);
  delete[] tensor_cpu;

  Tensor *new_tensor = createTensor(tensor_ptr, tensor.dims, dims_prod, false, "");
  return new_tensor;
}





std::vector<float> cur_dim;

extern "C" float StoreDimsOnDemand(char *tensor_name, float d)
{
  std::vector<float> dims;
  
  if (NamedDims.count(tensor_name)>0)
    dims = NamedDims[tensor_name];

  dims.push_back(d);

  NamedDims[tensor_name] = dims;
  return 0;
}



extern "C" void CreatePinnedTensorOnDemand(char *tensor_name, char *init)
{
  std::vector<float> dims = NamedDims[tensor_name];
  NamedDims[tensor_name].clear();
  Tensor *tensor;

  int product = DimsProd(dims);
  float *tensor_ptr;
  float *tensor_cpu;


  cudaMallocHost(&tensor_cpu, product*sizeof(float));
  //tensor_cpu = new float[product];

  for (int i = 0; i < product; ++i) {
    tensor_cpu[i] = 0.0f;
  }
  

  cudaMalloc(&tensor_ptr, product*sizeof(float));
  //cudaCheck(cudaMemcpy(tensor, tensor_cpu, product*sizeof(float), cudaMemcpyHostToDevice));
  
  tensor = createPinned(tensor_ptr, tensor_cpu, dims, product, tensor_name);

  NamedTensorsT[tensor_name] = tensor;
  
}

extern "C" float CreateTensorOnDemand(char *tensor_name, char *scopeless_name, char *init, int is_weight)
{
  //std::cout << "CREATING TENSOR " << tensor_name << "\n";

  Tensor *tensor;

  std::vector<float> dims = NamedDims[tensor_name];
  NamedDims[tensor_name].clear(); //TODO: Global vars are bad with threads.

  int product = DimsProd(dims);

  float *tensor_ptr;
  float *tensor_cpu;

  if (std::strcmp(init, "randu") == 0)
    tensor_cpu = make_random_float_uniform(product);
  else if (std::strcmp(init, "zeros") == 0)
    tensor_cpu = make_zeros_float(product);
  else if (std::strcmp(init, "ones") == 0)
    tensor_cpu = make_ones_float(product);
  else if (std::strcmp(init, "xavu") == 0)
    tensor_cpu = make_xavier_uniform_float(product, dims[dims.size()-1], dims[dims.size()-2]);
  else if (std::strcmp(init, "xavu_relu") == 0)
    tensor_cpu = make_xavier_uniform_float_relu(product, dims[dims.size()-1], dims[dims.size()-2]);
  else if (std::strcmp(init, "int") == 0)
    tensor_cpu = make_random_int(product, 10);
  else if (std::strcmp(init, "binary") == 0)
    tensor_cpu = make_random_int(product, 1);
    

  cudaCheck(cudaMalloc(&tensor_ptr, product*sizeof(float)));
  cudaCheck(cudaMemcpy(tensor_ptr, tensor_cpu, product*sizeof(float), cudaMemcpyHostToDevice));
  delete[] tensor_cpu;


  /*
  if(NamedTensorsT.count(tensor_name)>0)
  {
    tensor = NamedTensorsT[tensor_name];
    delete tensor;
  }


  
  if (NamedTensorsT.count(tensor_name) > 0)
  {
    float *aux_ptr = NamedTensorsT[tensor_name].tensor_ptr;
    if (aux_ptr!=nullptr)
      cudaCheck(cudaFree(aux_ptr));
  }
  */


  tensor = createTensor(tensor_ptr, dims, product, true, tensor_name);
  tensor->scopeless_name = scopeless_name;
  tensor->weight = (bool)is_weight;
  tensor->op = create_tensor_op;

  
  NamedTensorsT[tensor_name] = tensor;

  return 0;
}


Value *TensorExprAST::codegen(Value *first_arg, Value *scope_str, Value *previous_scope) {
  if (not ShallCodegen)
    return ConstantFP::get(*TheContext, APFloat(0.0f));


  Function *TheFunction = Builder->GetInsertBlock()->getParent();

  // Register all variables and emit their initializer.
  for (unsigned i = 0, e = VarNames.size(); i != e; ++i) {
    const std::string &VarName = VarNames[i].first;
    
    
    Value *var_name, *scopeless_name;
    var_name = Builder->CreateGlobalString(VarName);

    bool is_self = GetSelf();
    bool is_attr = GetIsAttribute();

    if (is_self||is_attr)
      var_name = Builder->CreateCall(TheModule->getFunction("ConcatStr"),
                                            {first_arg, var_name});
    scopeless_name = Builder->CreateCall(TheModule->getFunction("CopyString"),
                                            {var_name});
    if (!(is_self||is_attr))
      var_name = Builder->CreateCall(TheModule->getFunction("ConcatStr"),
                                            {scope_str, var_name});

    Value *aux;
    for (int j=0; j<V_Dims.size(); j++)
    {
      aux = V_Dims[j]->codegen(first_arg, scope_str, previous_scope);
      Builder->CreateCall(TheModule->getFunction("StoreDimsOnDemand"),
                                                  {var_name, aux});
    }

    Builder->CreateCall(TheModule->getFunction("CreateTensorOnDemand"),
                                              {var_name, scopeless_name, Builder->CreateGlobalString(TensorInit),
                                               ConstantInt::get(Type::getInt32Ty(*TheContext), IsWeight)});
  }


  return ConstantFP::get(*TheContext, APFloat(0.0));
}




Value *PinnedTensorExprAST::codegen(Value *first_arg, Value *scope_str, Value *previous_scope) {
  if (not ShallCodegen)
    return ConstantFP::get(*TheContext, APFloat(0.0f));


  std::cout << "\n\nPinned tensor type: " << Type << "\n\n\n";

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
      InitVal = Init->codegen(first_arg, scope_str, previous_scope);
      if (!InitVal)
        return nullptr;
    } else { // If not specified, use 0.0.
      InitVal = ConstantFP::get(*TheContext, APFloat(0.0));
    }


   Value *var_name = Builder->CreateGlobalString(VarName);

    std::string pre_dot = GetPreDot();
    bool is_self = GetSelf();
    bool is_attr = GetIsAttribute();

    if (is_self||is_attr)
      var_name = Builder->CreateCall(TheModule->getFunction("ConcatStr"),
                                            {first_arg, var_name});
    if (!(is_self||is_attr))
      var_name = Builder->CreateCall(TheModule->getFunction("ConcatStr"),
                                            {scope_str, var_name});

    Value *aux;
    for (int j=0; j<V_Dims.size(); j++)
    {
      aux = V_Dims[j]->codegen(first_arg, scope_str, previous_scope);
      Builder->CreateCall(TheModule->getFunction("StoreDimsOnDemand"),
                                                  {var_name, aux});
    }
    
    Builder->CreateCall(TheModule->getFunction("CreatePinnedTensorOnDemand"),
                                              {var_name, Builder->CreateGlobalString(TensorInit)});

 
  }


  return ConstantFP::get(*TheContext, APFloat(0.0));
}





Value *Conv2dExprAST::codegen(Value *first_arg, Value *scope_str, Value *previous_scope) {
  if (not ShallCodegen)
    return ConstantFP::get(*TheContext, APFloat(0.0f));



  Function *TheFunction = Builder->GetInsertBlock()->getParent();

  // Register all variables and emit their initializer.
  for (unsigned i = 0, e = VarNames.size(); i != e; ++i) {
    const std::string &VarName = VarNames[i].first;
    ExprAST *Init = VarNames[i].second.get();


    Value *var_name, *scopeless_name;
    var_name = Builder->CreateGlobalString(VarName);

    bool is_self = GetSelf();
    bool is_attr = GetIsAttribute();

    if (is_self||is_attr)
      var_name = Builder->CreateCall(TheModule->getFunction("ConcatStr"),
                                            {first_arg, var_name});
    scopeless_name = Builder->CreateCall(TheModule->getFunction("CopyString"),
                                            {var_name});
    if (!(is_self||is_attr))
      var_name = Builder->CreateCall(TheModule->getFunction("ConcatStr"),
                                            {scope_str, var_name});
    
    


    std::cout << "Parsing Conv2d var for: " << VarName << "\n";

    Builder->CreateCall(TheModule->getFunction("CreateConv2dOnDemand"),
                                              {var_name, Builder->CreateGlobalString(TensorInit),
                                               C->codegen(first_arg, scope_str, previous_scope), OC->codegen(first_arg, scope_str, previous_scope), Ks->codegen(first_arg, scope_str, previous_scope), Stride->codegen(first_arg, scope_str, previous_scope),
                                               Padding->codegen(first_arg, scope_str, previous_scope)});
  }
  return ConstantFP::get(*TheContext, APFloat(0.0));
}



Value *MaxPool2dExprAST::codegen(Value *first_arg, Value *scope_str, Value *previous_scope) {
  if (not ShallCodegen)
    return ConstantFP::get(*TheContext, APFloat(0.0f));



  Function *TheFunction = Builder->GetInsertBlock()->getParent();

  // Register all variables and emit their initializer.
  for (unsigned i = 0, e = VarNames.size(); i != e; ++i) {
    const std::string &VarName = VarNames[i].first;
    
    Value *var_name, *scopeless_name, *type;
    var_name = Builder->CreateGlobalString(VarName);
    type = Builder->CreateGlobalString(Type);

    bool is_self = GetSelf();
    bool is_attr = GetIsAttribute();

    if (is_self||is_attr)
      var_name = Builder->CreateCall(TheModule->getFunction("ConcatStr"),
                                            {first_arg, var_name});
    scopeless_name = Builder->CreateCall(TheModule->getFunction("CopyString"),
                                            {var_name});
    if (!(is_self||is_attr))
      var_name = Builder->CreateCall(TheModule->getFunction("ConcatStr"),
                                            {scope_str, var_name});
    

    
    std::cout << "Parsing Conv2d var for: " << VarName << "\n";

    Builder->CreateCall(TheModule->getFunction("CreateMaxPool2dOnDemand"),
                                              {var_name, type,
                                               Ks->codegen(first_arg, scope_str, previous_scope),
                                               Stride->codegen(first_arg, scope_str, previous_scope),
                                               Padding->codegen(first_arg, scope_str, previous_scope)});
  }
  return ConstantFP::get(*TheContext, APFloat(0.0));
}



Value *BatchNorm2dExprAST::codegen(Value *first_arg, Value *scope_str, Value *previous_scope) {
  if (not ShallCodegen)
    return ConstantFP::get(*TheContext, APFloat(0.0f));


  Function *TheFunction = Builder->GetInsertBlock()->getParent();

  // Register all variables and emit their initializer.
  for (unsigned i = 0, e = VarNames.size(); i != e; ++i) {
    const std::string &VarName = VarNames[i].first;
    
    Value *var_name, *scopeless_name, *type;
    var_name = Builder->CreateGlobalString(VarName);
    type = Builder->CreateGlobalString(Type);

    bool is_self = GetSelf();
    bool is_attr = GetIsAttribute();

    if (is_self||is_attr)
      var_name = Builder->CreateCall(TheModule->getFunction("ConcatStr"),
                                            {first_arg, var_name});
    scopeless_name = Builder->CreateCall(TheModule->getFunction("CopyString"),
                                            {var_name});
    if (!(is_self||is_attr))
      var_name = Builder->CreateCall(TheModule->getFunction("ConcatStr"),
                                            {scope_str, var_name});
    

    
    std::cout << "Parsing Conv2d var for: " << VarName << "\n";

    Builder->CreateCall(TheModule->getFunction("CreateBatchNorm2dOnDemand"),
                                              {var_name, type,
                                               C->codegen(first_arg, scope_str, previous_scope)});
  }
  return ConstantFP::get(*TheContext, APFloat(0.0));
}




Value *CallExprAST::codegen(Value *first_arg, Value *scope_str, Value *previous_scope) {
  if (not ShallCodegen)
    return ConstantFP::get(*TheContext, APFloat(0.0f));
  // Look up the name in the global module table.
  std::string tgt_function = Callee;
  

  Function *TheFunction = Builder->GetInsertBlock()->getParent();
  std::string functionName = TheFunction->getName().str();
  std::string tgt_function_name;

  std::cout << "\n\nFunction: " << tgt_function << "\n";

  int nested_function;
  if (functionName=="__anon_expr" || starts_with(functionName.c_str(), "__async_"))
  {
    nested_function=0;
  }
  else
    nested_function=1;


  

  //TODO: Solve scope_str discontinuity on async functions
  if (starts_with(functionName.c_str(), "__async_"))
    scope_str = Builder->CreateCall(TheModule->getFunction("GetEmptyChar"), {});


  //Builder->CreateCall(TheModule->getFunction("FreeChar"), {previous_scope});
  
  previous_scope = Builder->CreateCall(TheModule->getFunction("CopyString"),
                                        {scope_str});


  Value *_pre_dot_str = Builder->CreateGlobalString(_pre_dot);
  Value *first_arg_copy;



  if (isAttribute && !isSelf && !in_str(tgt_function, native_methods))
  { // e.g: model.forward()
    if (nested_function)
      first_arg_copy = Builder->CreateCall(TheModule->getFunction("CopyString"),
                                                    {first_arg});

    first_arg = Builder->CreateCall(TheModule->getFunction("CopyString"),
                                                    {_pre_dot_str});
                                                    
    first_arg = Builder->CreateCall(TheModule->getFunction("ConcatStr"),
                {previous_scope, first_arg});
  }
  
  
  



  int target_args_size = Args.size();
  std::vector<Value *> ArgsV;

  bool is_self_of_nested_function = (nested_function==1 && isSelf);
  
  // Handle self or object attribute expressions
  if(isSelf || isAttribute)
  {
    bool not_coding_language_method = (!in_str(tgt_function, native_methods));    

    if (not_coding_language_method)
      tgt_function = Class+tgt_function;



    if (!is_self_of_nested_function && not_coding_language_method)
    {
      if (nested_function)
        _pre_dot_str = Builder->CreateCall(TheModule->getFunction("ConcatStr"),
                {scope_str, _pre_dot_str});
      first_arg = Builder->CreateCall(TheModule->getFunction("FirstArgOnDemand"),
                                                    {first_arg,
                                                     _pre_dot_str,
                                                     Builder->CreateGlobalString(Class),
                                                     Builder->CreateGlobalString(Callee),
                                                     ConstantInt::get(Type::getInt32Ty(*TheContext), nested_function),
                                                     ConstantInt::get(Type::getInt32Ty(*TheContext), isSelf),
                                                     ConstantInt::get(Type::getInt32Ty(*TheContext), isAttribute)});
    }
    if (is_self_of_nested_function && not_coding_language_method)
    {
      first_arg_copy = Builder->CreateCall(TheModule->getFunction("CopyString"), {first_arg});
      //first_arg = Builder->CreateCall(TheModule->getFunction("CopyString"), {first_arg});
      first_arg = Builder->CreateCall(TheModule->getFunction("ConcatStr"),
                                                    {first_arg_copy,
                                                     _pre_dot_str});
    }
    
    
    if (CalleeOverride!="none"||in_str(Callee, native_methods))
    { // e.g: x.view()
    
      
      
      if (isSelf&&!isAttribute)
        ArgsV.push_back(first_arg);
      if (!isSelf&&isAttribute)
      {
        Value *arg = Builder->CreateCall(TheModule->getFunction("ConcatStr"),
                        {previous_scope, _pre_dot_str});
        ArgsV.push_back(arg);
      }
      
      if (isSelf && isAttribute)
      { // e.g: self.can_load_.first_nonzero()
        // Extend first arg
        ArgsV.push_back(first_arg);
        ArgsV[0] = Builder->CreateCall(TheModule->getFunction("ConcatStr"),
                                        {ArgsV[0], _pre_dot_str});
        //ArgsV.push_back(ConstantInt::get(Type::getInt32Ty(*TheContext), (int)(isSelf)));
        //target_args_size+=1;
      }

      if (in_str(Callee, return_tensor_methods))
        ArgsV[0] = Builder->CreateCall(TheModule->getFunction("LoadTensor"), {ArgsV[0]});

    }
    else
    { // Pass first_arg's reference for the derived AST nodes.
      std::cout << "Adding first arg and scope  for " << tgt_function << "\n";
      ArgsV.push_back(first_arg);
    }
    target_args_size+=1;
  }


  bool has_scope = false;
  if (!(CalleeOverride!="none" || in_str(Callee, native_fn)))
  {
    has_scope = true;
    //if (starts_with(functionName.c_str(), "__async_"))
    //  scope_name = Builder->CreateGlobalString("threaded_");
    //else 
    scope_str = Builder->CreateCall(TheModule->getFunction("RandomStrOnDemand"), {});
    
    
    ArgsV.push_back(scope_str); // Pass scope's reference for the derived AST nodes.
    ArgsV.push_back(previous_scope);
    target_args_size+=2;
  }
  


  
  

  // Detect function errors
  Function *CalleeF;
  if (!IsVarForward)
  {
    CalleeF = getFunction(tgt_function);
    if (!CalleeF)
    {
      std::string _error = "The referenced function "+ tgt_function +" was not yet declared.";
      return LogErrorV(_error);
    }

    tgt_function_name = CalleeF->getName().str();

    // If argument mismatch error.
    if ((CalleeF->arg_size()) != target_args_size && !in_str(tgt_function_name, vararg_methods))
    {
      //std::cout << "CalleeF->arg_size() " << CalleeF->arg_size() << " target_args_size " << target_args_size << "\n";
      std::string _error = "Incorrect parameters used on function " + tgt_function + " call.";
      return LogErrorV(_error);
    }
  }
  //std::cout << "\n\n\nCalling function: " << tgt_function <<"\n";









  // Get Arguments
  for (unsigned i = 0, e = Args.size(); i != e; ++i) {
    //std::cout << "\nCall codegen for argument n°: " << i << ".\n";

    // deal with firstarg on self.mcts(self.actions)
    Value *fa = (isAttribute && !isSelf && !in_str(tgt_function, native_methods) && nested_function) ? first_arg_copy : first_arg;

    //deal with scope on model.forward()
    Value *_scope = (!in_str(tgt_function, native_methods)) ? previous_scope : scope_str;
    

    Value * arg;
    std::cout << "ARG: " << Args[i]->GetName() << " has self: " << Args[i]->GetSelf() << " and type: " << Args[i]->GetType() <<  "\n\n";
    if ((Args[i]->GetType()=="tensor" || Args[i]->GetType()=="pinned_tensor") && Args[i]->GetIsVarLoad())
    {
      //if (starts_with(functionName.c_str(), "__async_"))
      //  Builder->CreateStore(Builder->CreateGlobalString("threaded_"), _scope);
      VariableExprAST *Arg = static_cast<VariableExprAST *>(Args[i].get());
      arg = Arg->NameSolver->codegen(first_arg, _scope, previous_scope);

      arg = Builder->CreateCall(TheModule->getFunction("LoadTensor"), {arg});
    }
    else
      arg = Args[i]->codegen(fa, _scope, previous_scope);

  
    ArgsV.push_back(arg);


    if (!ArgsV.back())
      return nullptr;
  }







  
  Value * ret = ConstantFP::get(*TheContext, APFloat(0.0f));
  std::cout << "\n\nCreate call: "  << tgt_function_name << " from parent: " << functionName << ", with override: " << CalleeOverride << "\n\n";
  if (CalleeOverride=="none")
    ret = Builder->CreateCall(CalleeF, ArgsV, "calltmp");
  else
  {
    //std::cout << "Override: " << CalleeOverride << "\n";
    if (CalleeOverride=="ConvForward2d"||CalleeOverride=="MaxPoolForward2d"||CalleeOverride=="BatchNormForward2d")
    {
      CalleeF = getFunction(CalleeOverride);
      Value *conv_name = Builder->CreateGlobalString(tgt_function);
      Value *is_attr = ConstantInt::get(Type::getInt32Ty(*GlobalContext), (int)(isSelf));
      ArgsV.push_back(conv_name);
      ArgsV.push_back(is_attr);
      ret = Builder->CreateCall(CalleeF, ArgsV, "calltmp");
    }
    else if (CalleeOverride=="SplitString")
    {
      Value *V = Builder->CreateCall(TheModule->getFunction("LoadStrOnDemand"), 
                                     {Builder->CreateGlobalString(PreDot)});
      
      ret = Builder->CreateCall(getFunction("SplitString"), 
                          {V, ArgsV[1]});

    }
    else if (CalleeOverride=="ToFloat")
    {
      std::cout << "\n\nTO FLOAT HAS TYPE " << Args[0]->GetType() << "\n";
      if (Args[0]->GetType()=="str")
        ret = Builder->CreateCall(getFunction("StrToFloat"), 
                          {ArgsV[0]});

    } else
      ret = Builder->CreateCall(getFunction(CalleeOverride), ArgsV, "calltmp");
  }

  
  Builder->CreateCall(TheModule->getFunction("FreeChar"), {previous_scope});
  //if (has_scope)
  //  Builder->CreateCall(TheModule->getFunction("FreeChar"), {scope_str});
  
  return ret;
}







Function *PrototypeAST::codegen() {
  if (not ShallCodegen)
    return nullptr;
  // Make the function type:  float(float,float) etc.

  std::vector<Type *> types;


  for (auto &type : Types)
  {
    if (type=="s"||type=="t"||type=="c")
      types.push_back(int8PtrTy);
    else
      types.push_back(Type::getFloatTy(*TheContext));
  }


  FunctionType *FT = FunctionType::get(Type::getFloatTy(*TheContext), types, false);
  

  Function *F =
      Function::Create(FT, Function::ExternalLinkage, Name, TheModule.get());

  // Set names for all arguments.
  unsigned Idx = 0;
  for (auto &Arg : F->args())
    Arg.setName(Args[Idx++]);
  

  return F;
}



extern "C" void *SplitString(char *self, char *pattern)
{

  //std::cout << "\n\nSPLITTING: " << self << ", with pattern: " << pattern << "\n";


  std::vector<char *> result;
  char *input = strdup(self); // Duplicate the input string to avoid modifying the original
  char *token = strtok(input, pattern); // Get the first token

  while (token != nullptr) {
    result.push_back(token);
    token = strtok(nullptr, pattern); // Get the next token
  }

  std::string random_str = RandomString(15);
  StrVecAuxHash[random_str] = result;
  AuxRandomStrs[random_str] = "str_vec";
    
  return &StrVecAuxHash[random_str];
    
}




// INDEX METHODS

extern "C" char *SplitStringIndexate(char *name, char *pattern, float idx)
{
  char *self = NamedStrs[name];
  //std::cout << "splitting: " << self << ", with pattern: " << pattern << "\n";

  
  std::vector<char *> splits;
  char *input = (char*)malloc(strlen(self) + 1);
  strcpy(input, self);

  char *saveptr;
  char *token = strtok_r(input, pattern, &saveptr); // Get the first token

  while (token != nullptr) {
    splits.push_back(token);
    token = strtok_r(nullptr, pattern, &saveptr); // Get the next token
  }

  if (idx < 0) 
    idx = splits.size() + idx;
  

  //std::cout << "Spltting " << self << " with " << pattern <<" at ["<<idx<<"]:  " << splits[idx] << "\n";
 
  delete[] name;
  return splits[idx];
}




extern "C" char *IndexStrVec(std::vector<char*> vec, float _idx)
{

  int idx = (int) _idx;

  //std::cout << "Str vec indexed at [" << idx << "]: " << vec[idx] << "\n";
  
  
  return vec[idx];
}


extern "C" char * IndexClassStrVec(char *vec_name, float _idx)
{
  int idx = (int) _idx;


  std::vector<char*> vec = ClassStrVecs[vec_name];

  //std::cout << "Class object Str Vec " << vec_name << "indexed at [" << idx << "]: " << vec[idx] << "\n";
  delete[] vec_name;

  return vec[idx];
}


extern "C" float IndexClassFloatVec(char *vec_name, float _idx)
{
  int idx = (int) _idx;

  float ret = ClassFloatVecs[vec_name][idx];
  delete[] vec_name;
  return ret;
}



extern "C" float StrToFloat(char *in_str)
{
  //std::cout << "\n\nstr to float of " << in_str << "\n\n\n";

  char *copied = (char*)malloc(strlen(in_str) + 1);
  strcpy(copied, in_str);
  char *end;

  float ret = std::strtof(copied, &end);
  delete[] copied;
  return ret;
}




extern "C" void objAttr_var_from_var(char *LName, char *RName)
{
  //std::cout << "objAttr_var_from_var of " << LName << " from " << RName << "\n";

  //std::cout << "Loading: " << NamedObjects[RName] << "\n";
  //std::cout << "Replacing: " << NamedObjects[LName] << "\n";

  NamedObjects[LName] = NamedObjects[RName];
  
  
}

extern "C" void objAttr_var_from_vec(char *LName, char *RName)
{
  //std::cout << "objAttr_var_from_vec of " << LName << " from " << RName << "\n";

  //std::cout << "Loading: " << objectVecs[RName] << "\n";
  //std::cout << "Replacing: " << NamedObjects[LName] << "\n";

  NamedObjects[LName] = objectVecs[RName];

  
}

extern "C" void objAttr_vec_from_var(char *LName, char *RName)
{
  //std::cout << "objAttr_vec_from_var of " << LName << " from " << RName << "\n";

  //std::cout << "Loading: " << NamedObjects[RName] << "\n";
  //std::cout << "Replacing: " << objectVecs[LName] << "\n";

  objectVecs[LName] = NamedObjects[RName];

  
}


extern "C" void objAttr_vec_from_vec(char *LName, char *RName)
{
  //std::cout << "objAttr_vec_from_vec of " << LName << " from " << RName << "\n";

  //std::cout << "Loading: " << objectVecs[RName] << "\n";
  //std::cout << "Replacing: " << objectVecs[LName] << "\n";

  objectVecs[LName] = objectVecs[RName];

  
}

extern "C" float append(char *self, char *obj_name)
{
  //char* copied = (char*)malloc(strlen(in_str) + 1);
  //strcpy(copied, in_str);

  std::cout << "\n\nAPPEND OF " << obj_name << " into: " << self << "\n";
  
  std::string obj_name_str = obj_name;


  


  int obj_vec_last_id = 0;
  if (objectVecsLastId.count(self)>0)
  {
    obj_vec_last_id = objectVecsLastId[self];
    obj_vec_last_id+=1;
  }
  objectVecsLastId[self] = obj_vec_last_id;

  std::string indexed_self = self + std::to_string(obj_vec_last_id);
  objectVecs[indexed_self] = NamedObjects[obj_name];

  
  

  return 0;
}

extern "C" char *LoadObjectScopeName(char *self)
{
  if (objectVecs.count(self)==0)
  {
    std::string _self = self;
    std::string _error = "Object "+_self+" does not exist";
    LogErrorS(_error);
    return "";
  }
  std::string ret = objectVecs[self];
  delete[] self;
  //std::cout << "LoadObjectScopeName is: " << ret << ", from self: " << self << "\n";

  return str_to_char(ret);
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
  std::string function_name = P.getName();

  Function *TheFunction = getFunction(function_name);
  if (!TheFunction)
    return nullptr;

  // If this is an operator, install it.
  if (P.isBinaryOp())
    BinopPrecedence[P.getOperatorName()] = P.getBinaryPrecedence();

  // Create a new basic block to start insertion into.
  BasicBlock *BB = BasicBlock::Create(*TheContext, "entry", TheFunction);
  Builder->SetInsertPoint(BB);

  
  // Record the function arguments in the NamedValues map.

  Value *first_arg, *scope_str, *previous_scope;
  /*
  first_arg = Builder->CreateAlloca(int8PtrTy);
  scope_str = Builder->CreateAlloca(int8PtrTy);
  previous_scope = Builder->CreateAlloca(int8PtrTy);
  */

  if (function_name=="__anon_expr")
  {
    first_arg = Builder->CreateCall(TheModule->getFunction("GetEmptyChar"), {});
    scope_str = Builder->CreateCall(TheModule->getFunction("GetEmptyChar"), {});
    previous_scope = Builder->CreateCall(TheModule->getFunction("GetEmptyChar"), {});
  }
  


  std::cout << "\033[32mExecuting function: " << function_name << " \033[0m\n";

  NamedValues.clear();

  bool has_self, has_scope, has_previous_scope;
  has_self = false;
  has_scope = false;
  has_previous_scope = false;

  float val;
  int i = 0;
  for (auto &Arg : TheFunction->args()) {
    // Create an alloca for this variable.
    // TODO: solve bugged shifted arguments when using tensors
    
    std::string arg_name = Arg.getName().str();
    //std::cout << "FUNCTION ARG IS: " << arg_name  << "\n";

    std::string __print = "FUNCTION ALLOCA OF " + std::string(Arg.getName()) + " ";

    if (arg_name == "self")
    {
      first_arg = Builder->CreateCall(TheModule->getFunction("CopyString"), {&Arg});
      
      has_self = true;
    }
    else if (arg_name == "scope_str")
    {
      scope_str = Builder->CreateCall(TheModule->getFunction("CopyString"), {&Arg});
      
      has_scope = true;
    }
    else if (arg_name == "previous_scope")
    {
      previous_scope = Builder->CreateCall(TheModule->getFunction("CopyString"), {&Arg});
      
      has_previous_scope = true;
    } else if (in_str(arg_name, floatVars))
    {
      Value *var_name = Builder->CreateGlobalString(arg_name);
      var_name = Builder->CreateCall(TheModule->getFunction("ConcatStr"),
                                    {scope_str, var_name});

      Builder->CreateCall(TheModule->getFunction("StoreArgOnDemand"),
                                                  {var_name, &Arg});
    } else if (in_str(arg_name, strVars))
    {
      Value *var_name = Builder->CreateGlobalString(arg_name);
      var_name = Builder->CreateCall(TheModule->getFunction("ConcatStr"),
                                    {scope_str, var_name});


      Builder->CreateCall(TheModule->getFunction("StoreStrOnDemand"),
                                                  {var_name, &Arg});
    }
    else if (!in_str(arg_name, tensorVars))
    {
      AllocaInst *Alloca = CreateEntryBlockAlloca(TheFunction, Arg.getName());
      Builder->CreateStore(&Arg, Alloca);
      //Builder->CreateStore(Builder->CreateLoad(Type::getFloatTy(*TheContext), &Arg), Alloca);



      NamedValues[std::string(Arg.getName())] = Alloca;

    }
    else
    {
      if (in_str(arg_name, tensorVars))
      {
        //Builder->CreateCall(TheModule->getFunction("print"),
        //  {Builder->CreateGlobalString(__print), &Arg});

        Builder->CreateCall(TheModule->getFunction("CopyArgTensor"),
                          {&Arg,
                           Builder->CreateGlobalString(arg_name),
                           previous_scope,
                           scope_str});
      }
    }
  }
  


  Value *RetVal;
  for (auto &body : Body)
    RetVal = body->codegen(first_arg, scope_str, previous_scope);



  Value *aux = Builder->CreateGlobalString(function_name);


  
  

  //if(has_self)
  //  Builder->CreateCall(TheModule->getFunction("FreeCharFromFunc"), {first_arg, aux});
  
  if(has_scope)
    Builder->CreateCall(TheModule->getFunction("CleanScopeVars"), {scope_str});
  
  //if(has_previous_scope)
  //  Builder->CreateCall(TheModule->getFunction("FreeCharFromFunc"), {previous_scope, aux});
  
  

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
  //std::cout << "\nINITIALIZING A NEW MODULE"  << "\n\n";

  // Open a new context and module.
  TheContext = std::make_unique<LLVMContext>();
  TheModule = std::make_unique<Module>("my cool jit", *TheContext);
  TheModule->setDataLayout(TheJIT->getDataLayout());

  //std::cout << "Initialize Module\n";
  

  // Create a new builder for the module.
  Builder = std::make_unique<IRBuilder<>>(*TheContext);

  floatPtrTy = Type::getFloatTy(*TheContext)->getPointerTo();
  int8PtrTy = Type::getInt8Ty(*TheContext)->getPointerTo();
  ShallCodegen = true;
  seen_var_attr = false;



  //===----------------------------------------------------------------------===//
  // Scalar   Operations
  //===----------------------------------------------------------------------===//

  // 
  FunctionType *fmaxTy = FunctionType::get( //TODO: automatic type detection for max and min
      Type::getFloatTy(*TheContext),
      {Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext)},
      false
  );
  TheModule->getOrInsertFunction("max", fmaxTy);


  // 
  FunctionType *fminTy = FunctionType::get(
      Type::getFloatTy(*TheContext),
      {Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext)},
      false
  );
  TheModule->getOrInsertFunction("min", fminTy);


  // 
  FunctionType *flog2Ty = FunctionType::get(
      Type::getFloatTy(*TheContext),
      {Type::getFloatTy(*TheContext)},
      false
  );
  TheModule->getOrInsertFunction("logE2f", flog2Ty);


  // 
  FunctionType *roundTy = FunctionType::get(
      Type::getFloatTy(*TheContext),
      {Type::getFloatTy(*TheContext)},
      false
  );
  TheModule->getOrInsertFunction("roundE", roundTy);


  // 
  FunctionType *floorTy = FunctionType::get(
      Type::getFloatTy(*TheContext),
      {Type::getFloatTy(*TheContext)},
      false
  );
  TheModule->getOrInsertFunction("floorE", floorTy);


  //===----------------------------------------------------------------------===//
  // Tensor -- Scalar   Operations
  //===----------------------------------------------------------------------===//

  //
  FunctionType *CudaScalarMultTy = FunctionType::get(
      int8PtrTy,
      {int8PtrTy, Type::getFloatTy(*TheContext)}, 
      false 
  );
  TheModule->getOrInsertFunction("CudaScalarMult", CudaScalarMultTy);


  //
  FunctionType *CudaScalarDivTy = FunctionType::get(
      int8PtrTy,
      {int8PtrTy, Type::getFloatTy(*TheContext)}, 
      false
  );
  TheModule->getOrInsertFunction("CudaScalarDiv", CudaScalarDivTy);


  //
  FunctionType *CudaReverseScalarDivTy = FunctionType::get(
      int8PtrTy,
      {int8PtrTy, Type::getFloatTy(*TheContext)}, 
      false
  );
  TheModule->getOrInsertFunction("CudaReverseScalarDiv", CudaReverseScalarDivTy);


  //
  FunctionType *CudaScalarAddTy = FunctionType::get(
      int8PtrTy,
      {int8PtrTy, Type::getFloatTy(*TheContext)}, 
      false
  );
  TheModule->getOrInsertFunction("CudaScalarAdd", CudaScalarAddTy);


  //
  FunctionType *CudaScalarSubTy = FunctionType::get(
      int8PtrTy,
      {int8PtrTy, Type::getFloatTy(*TheContext)}, 
      false
  );
  TheModule->getOrInsertFunction("CudaScalarSub", CudaScalarSubTy);


  //
  FunctionType *CudaScalarEqualTy = FunctionType::get(
      int8PtrTy,
      {int8PtrTy, Type::getFloatTy(*TheContext)}, 
      false
  );
  TheModule->getOrInsertFunction("CudaScalarEqual", CudaScalarEqualTy);


  //
  FunctionType *CudaScalarDiffTy = FunctionType::get(
      int8PtrTy,
      {int8PtrTy, Type::getFloatTy(*TheContext)}, 
      false
  );
  TheModule->getOrInsertFunction("CudaScalarDiff", CudaScalarDiffTy);


  //
  FunctionType *CudaScalarMinorTy = FunctionType::get(
      int8PtrTy,
      {int8PtrTy, Type::getFloatTy(*TheContext)}, 
      false
  );
  TheModule->getOrInsertFunction("CudaScalarMinor", CudaScalarMinorTy);


  //
  FunctionType *CudaScalarHigherTy = FunctionType::get(
      int8PtrTy,
      {int8PtrTy, Type::getFloatTy(*TheContext)}, 
      false
  );
  TheModule->getOrInsertFunction("CudaScalarHigher", CudaScalarHigherTy);

  
  //
  FunctionType *CudaScalarHigherEqTy = FunctionType::get(
      int8PtrTy,
      {int8PtrTy, Type::getFloatTy(*TheContext)}, 
      false
  );
  TheModule->getOrInsertFunction("CudaScalarHigherEq", CudaScalarHigherEqTy);


  //
  FunctionType *CudaScalarMinorEqTy = FunctionType::get(
      int8PtrTy,
      {int8PtrTy, Type::getFloatTy(*TheContext)}, 
      false
  );
  TheModule->getOrInsertFunction("CudaScalarMinorEq", CudaScalarMinorEqTy);

  

  //===----------------------------------------------------------------------===//
  // Tensor Tensor CUDA Ops
  //===----------------------------------------------------------------------===//

  //
  FunctionType *CudaMultTy = FunctionType::get(
      int8PtrTy,
      {Type::getInt32Ty(*TheContext),
       int8PtrTy,
       int8PtrTy}, 
      false
  );
  TheModule->getOrInsertFunction("CudaMult", CudaMultTy);


  //
  FunctionType *CudaAddTy = FunctionType::get(
      int8PtrTy,
      {Type::getInt32Ty(*TheContext),
       int8PtrTy,
       int8PtrTy}, 
      false
  );
  TheModule->getOrInsertFunction("CudaAdd", CudaAddTy);


  //
  FunctionType *CudaSubTy = FunctionType::get(
      int8PtrTy,
      {Type::getInt32Ty(*TheContext),
       int8PtrTy,
       int8PtrTy}, 
      false
  );
  TheModule->getOrInsertFunction("CudaSub", CudaSubTy);


  //
  FunctionType *CudaEqualTy = FunctionType::get(
      int8PtrTy,
      {Type::getInt32Ty(*TheContext),
       int8PtrTy,
       int8PtrTy}, 
      false
  );
  TheModule->getOrInsertFunction("CudaEqual", CudaEqualTy);


  //
  FunctionType *CudaHadamardTy = FunctionType::get(
      int8PtrTy,
      {Type::getInt32Ty(*TheContext),
       int8PtrTy,
       int8PtrTy}, 
      false
  );
  TheModule->getOrInsertFunction("CudaHadamard", CudaHadamardTy);


  //
  FunctionType *CudaDivTy = FunctionType::get(
      int8PtrTy,
      {Type::getInt32Ty(*TheContext),
       int8PtrTy,
       int8PtrTy}, 
      false
  );
  TheModule->getOrInsertFunction("CudaDiv", CudaDivTy);


  //
  FunctionType *LoadTensorTy = FunctionType::get(
      int8PtrTy,
      {int8PtrTy}, 
      false
  );
  TheModule->getOrInsertFunction("LoadTensor", LoadTensorTy);


  //
  FunctionType *IdxTensorTy = FunctionType::get(
      floatPtrTy,
      {int8PtrTy, Type::getFloatTy(*TheContext)}, 
      false
  );
  TheModule->getOrInsertFunction("IdxTensor", IdxTensorTy);


  //
  FunctionType *IdxTensorWithTensorTy = FunctionType::get(
      floatPtrTy,
      {int8PtrTy, int8PtrTy}, 
      false
  );
  TheModule->getOrInsertFunction("IdxTensorWithTensor", IdxTensorWithTensorTy);


  //
  FunctionType *PrintTensorFTy = FunctionType::get(
      Type::getFloatTy(*TheContext),
      {floatPtrTy,
       Type::getInt32Ty(*TheContext),
       Type::getInt32Ty(*TheContext),}, 
      false
  );
  TheModule->getOrInsertFunction("PrintTensorF", PrintTensorFTy);


  //
  FunctionType *print_tensorTy = FunctionType::get(
      Type::getFloatTy(*TheContext),
      {int8PtrTy}, 
      false
  );
  TheModule->getOrInsertFunction("print_tensor", print_tensorTy);


  //
  FunctionType *LoadDimsTy = FunctionType::get(
      int8PtrTy,
      {int8PtrTy},
      false
  );
  TheModule->getOrInsertFunction("LoadDims", LoadDimsTy);


  //
  FunctionType *PrintDimsTy = FunctionType::get(
      Type::getVoidTy(*TheContext),
      {int8PtrTy}, 
      false 
  );
  TheModule->getOrInsertFunction("PrintDims", PrintDimsTy);


  //
  FunctionType *NewDimsOnIdxTy = FunctionType::get(
      int8PtrTy,
      {int8PtrTy}, 
      false
  );
  TheModule->getOrInsertFunction("NewDimsOnIdx", NewDimsOnIdxTy);
  

  //
  FunctionType *clipTy = FunctionType::get(
      floatPtrTy,
      {int8PtrTy,
       Type::getInt32Ty(*TheContext),
       Type::getInt32Ty(*TheContext)}, 
      false
  );
  TheModule->getOrInsertFunction("clip", clipTy);

  //===----------------------------------------------------------------------===//
  // Backward and Optimizers CUDA Ops
  //===----------------------------------------------------------------------===//

  //
  FunctionType *BackpropagationTy = FunctionType::get(
      Type::getFloatTy(*TheContext),
      {}, 
      false
  );
  TheModule->getOrInsertFunction("backprop", BackpropagationTy);
  
  
  //
  FunctionType *AdamWTy = FunctionType::get(
      Type::getFloatTy(*TheContext),
      {Type::getFloatTy(*TheContext),
       Type::getFloatTy(*TheContext),
       Type::getFloatTy(*TheContext),
       Type::getFloatTy(*TheContext)}, 
      false
  );
  TheModule->getOrInsertFunction("AdamW", AdamWTy);


  //===----------------------------------------------------------------------===//
  // Unary CUDA Ops
  //===----------------------------------------------------------------------===//

  //
  FunctionType *CudaLogTy = FunctionType::get(
      int8PtrTy,
      {int8PtrTy},
      false 
  );
  TheModule->getOrInsertFunction("logE", CudaLogTy);
  

  // 
  FunctionType *log2Ty = FunctionType::get(
      int8PtrTy,
      {int8PtrTy},
      false
  );
  TheModule->getOrInsertFunction("logE2", log2Ty);


  // 
  FunctionType *softmaxTy = FunctionType::get(
      int8PtrTy,
      {int8PtrTy},
      false
  );
  TheModule->getOrInsertFunction("softmax", softmaxTy);
  

  //
  FunctionType *reluTy = FunctionType::get(
      int8PtrTy,
      {int8PtrTy},
      false
  );
  TheModule->getOrInsertFunction("relu", reluTy);
  

  //
  FunctionType *geluTy = FunctionType::get(
      int8PtrTy,
      {int8PtrTy},
      false
  );
  TheModule->getOrInsertFunction("gelu", geluTy);
  

  //
  FunctionType *sigmoidTy = FunctionType::get(
      int8PtrTy,
      {int8PtrTy},
      false
  );
  TheModule->getOrInsertFunction("sigmoid", sigmoidTy);
  

  //
  FunctionType *tanhTy = FunctionType::get(
      int8PtrTy,
      {int8PtrTy},
      false
  );
  TheModule->getOrInsertFunction("_tanh", tanhTy);


  //
  FunctionType *conv2dForwardTy = FunctionType::get(
      int8PtrTy,
      {int8PtrTy, int8PtrTy, int8PtrTy, Type::getInt32Ty(*TheContext)},
      false
  );
  TheModule->getOrInsertFunction("ConvForward2d", conv2dForwardTy);


  //
  FunctionType *MaxPoolForward2dTy = FunctionType::get(
      int8PtrTy,
      {int8PtrTy, int8PtrTy, int8PtrTy, Type::getInt32Ty(*TheContext)},
      false
  );
  TheModule->getOrInsertFunction("MaxPoolForward2d", MaxPoolForward2dTy);


  //
  FunctionType *BatchNormForward2dTy = FunctionType::get(
      int8PtrTy,
      {int8PtrTy, int8PtrTy, int8PtrTy, Type::getInt32Ty(*TheContext)},
      false
  );
  TheModule->getOrInsertFunction("BatchNormForward2d", BatchNormForward2dTy);
  

  //
  FunctionType *onehotTy = FunctionType::get(
      int8PtrTy,
      {int8PtrTy, Type::getFloatTy(*TheContext)},
      false
  );
  TheModule->getOrInsertFunction("onehot", onehotTy);
  

  //
  FunctionType *shapeTy = FunctionType::get(
      Type::getFloatTy(*TheContext),
      {int8PtrTy},
      false
  );
  TheModule->getOrInsertFunction("shape", shapeTy);
  

  //
  FunctionType *evalTy = FunctionType::get(
      Type::getFloatTy(*TheContext),
      {},
      false
  );
  TheModule->getOrInsertFunction("eval", evalTy);
  

  //
  FunctionType *trainTy = FunctionType::get(
      Type::getFloatTy(*TheContext),
      {},
      false
  );
  TheModule->getOrInsertFunction("train", trainTy);

  
  //
  FunctionType *repeat_interleaveTy = FunctionType::get(
      int8PtrTy,
      {int8PtrTy, Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext)},
      false
  );
  TheModule->getOrInsertFunction("repeat_interleave", repeat_interleaveTy);
  

  // 
  FunctionType *sumTy = FunctionType::get(
      int8PtrTy,
      {int8PtrTy, Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext)},
      true // vararg
  );
  TheModule->getOrInsertFunction("sum", sumTy);
  

  // 
  FunctionType *prodTy = FunctionType::get(
      int8PtrTy,
      {int8PtrTy, Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext)},
      true // vararg
  );
  TheModule->getOrInsertFunction("prod", prodTy);


  // 
  FunctionType *meanTy = FunctionType::get(
      int8PtrTy,
      {int8PtrTy, Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext)},
      false
  );
  TheModule->getOrInsertFunction("mean", meanTy);
  

  // 
  FunctionType *maxTy = FunctionType::get(
      int8PtrTy,
      {int8PtrTy, Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext)},
      true
  );
  TheModule->getOrInsertFunction("tmax", maxTy);


  //
  FunctionType *argmaxTy = FunctionType::get(
      int8PtrTy,
      {int8PtrTy, Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext)},
      true
  );
  TheModule->getOrInsertFunction("argmax", argmaxTy);
  

  //
  FunctionType *topkTy = FunctionType::get(
      int8PtrTy,
      {int8PtrTy, Type::getFloatTy(*TheContext)},
      true
  );
  TheModule->getOrInsertFunction("topk", topkTy);


  //
  FunctionType *viewTy = FunctionType::get(
      int8PtrTy,
      {int8PtrTy,int8PtrTy,Type::getFloatTy(*TheContext),Type::getFloatTy(*TheContext),Type::getFloatTy(*TheContext),Type::getFloatTy(*TheContext),Type::getFloatTy(*TheContext),Type::getFloatTy(*TheContext),Type::getFloatTy(*TheContext),Type::getFloatTy(*TheContext)},
      true // Vararg
  );
  TheModule->getOrInsertFunction("view", viewTy);


  //
  FunctionType *CalculateIdxOffsetTy = FunctionType::get(
      Type::getFloatTy(*TheContext),
      {int8PtrTy, Type::getFloatTy(*TheContext),Type::getFloatTy(*TheContext),Type::getFloatTy(*TheContext),Type::getFloatTy(*TheContext),Type::getFloatTy(*TheContext),Type::getFloatTy(*TheContext),Type::getFloatTy(*TheContext),Type::getFloatTy(*TheContext)},
      true // Vararg
  );
  TheModule->getOrInsertFunction("CalculateIdxOffset", CalculateIdxOffsetTy);
  

  //===----------------------------------------------------------------------===//
  // Loss CUDA Ops
  //===----------------------------------------------------------------------===//

  //
  FunctionType *cross_entropyTy = FunctionType::get(
      Type::getFloatTy(*TheContext),
      {Type::getInt8Ty(*TheContext)->getPointerTo(), Type::getInt8Ty(*TheContext)->getPointerTo()}, 
      false
  );
  TheModule->getOrInsertFunction("cross_entropy", cross_entropyTy);
  

  //===----------------------------------------------------------------------===//
  // File Handling Ops
  //===----------------------------------------------------------------------===//
  
  //
  FunctionType *load_imgTy = FunctionType::get(
      PointerType::get(Type::getFloatTy(*GlobalContext), 0),
      {PointerType::get(Type::getInt8Ty(*GlobalContext), 0)},
      false 
  );
  TheModule->getOrInsertFunction("load_img", load_imgTy);
  

  //
  FunctionType *load_preprocess_imgTy = FunctionType::get(
      Type::getFloatTy(*TheContext),
      {int8PtrTy, int8PtrTy},
      false
  );
  TheModule->getOrInsertFunction("load_preprocess_img", load_preprocess_imgTy);
  

  //===----------------------------------------------------------------------===//
  // Pinned Tensor Ops
  //===----------------------------------------------------------------------===//

  //
  FunctionType *gload_imgTy = FunctionType::get(
      floatPtrTy,
      {int8PtrTy, int8PtrTy, Type::getFloatTy(*TheContext)},
      false 
  );
  TheModule->getOrInsertFunction("gload_img", gload_imgTy);


  //
  FunctionType *wload_imgTy = FunctionType::get(
      floatPtrTy,
      {int8PtrTy, int8PtrTy, Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext)},
      false
  );
  TheModule->getOrInsertFunction("wload_img", wload_imgTy);
  

  //
  FunctionType *AttrPinnedOnIdxTy = FunctionType::get(
      Type::getVoidTy(*TheContext),
      {int8PtrTy, Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext)},
      false
  );
  TheModule->getOrInsertFunction("AttrPinnedOnIdx", AttrPinnedOnIdxTy);


  //
  FunctionType *gpuTy = FunctionType::get(
      int8PtrTy,
      {int8PtrTy},
      false 
  );
  TheModule->getOrInsertFunction("gpu", gpuTy);


  //  
  FunctionType *gpuwTy = FunctionType::get(
      int8PtrTy,
      {int8PtrTy, Type::getFloatTy(*TheContext)},
      false 
  );
  TheModule->getOrInsertFunction("gpuw", gpuwTy);


  //===----------------------------------------------------------------------===//
  // Parallel Ops
  //===----------------------------------------------------------------------===//

  //  
  FunctionType *sleepTy = FunctionType::get(
      Type::getVoidTy(*TheContext),
      {Type::getFloatTy(*TheContext)},
      false
  );
  TheModule->getOrInsertFunction("sleep", sleepTy);

  
  FunctionType *silent_sleepTy = FunctionType::get(
      Type::getFloatTy(*TheContext),
      {Type::getFloatTy(*TheContext)},
      false
  );
  TheModule->getOrInsertFunction("silent_sleep", silent_sleepTy);
  


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
  TheModule->getOrInsertFunction("pthread_create_aux", pthreadCreateTy);


  // int pthread_join(pthread_t thread, void **value_ptr)
  FunctionType *pthreadJoinTy = FunctionType::get(
    Type::getVoidTy(*TheContext),
    {pthreadPtr},
    false);
  TheModule->getOrInsertFunction("pthread_join_aux", pthreadJoinTy);

  

  FunctionType *LockTy = FunctionType::get(
    Type::getVoidTy(*TheContext),
    {int8PtrTy},
    false);
  TheModule->getOrInsertFunction("LockMutex", LockTy);

  FunctionType *UnlockMutexTy = FunctionType::get(
    Type::getVoidTy(*TheContext),
    {int8PtrTy},
    false);
  TheModule->getOrInsertFunction("UnlockMutex", UnlockMutexTy);


  //===----------------------------------------------------------------------===//
  // Str Ops
  //===----------------------------------------------------------------------===//


  // char *
  FunctionType *globTy = FunctionType::get(
      int8PtrTy,
      {int8PtrTy},
      false
  );
  TheModule->getOrInsertFunction("_glob_b_", globTy);


  //
  FunctionType *zeros_vecTy = FunctionType::get(
      int8PtrTy,
      {Type::getFloatTy(*TheContext)},
      false
  );
  TheModule->getOrInsertFunction("zeros_vec", zeros_vecTy);


  //
  FunctionType *ones_vecTy = FunctionType::get(
      int8PtrTy,
      {Type::getFloatTy(*TheContext)},
      false
  );
  TheModule->getOrInsertFunction("ones_vec", ones_vecTy);
  

  FunctionType *PrintFloatTy = FunctionType::get(
      Type::getVoidTy(*TheContext),
      {Type::getFloatTy(*TheContext)},
      false 
  );
  TheModule->getOrInsertFunction("PrintFloat", PrintFloatTy);

  FunctionType *UnbugFloatTy = FunctionType::get(
      Type::getFloatTy(*TheContext),
      {Type::getFloatTy(*TheContext)},
      false 
  );
  TheModule->getOrInsertFunction("UnbugFloat", UnbugFloatTy);


FunctionType *unbugTy = FunctionType::get(
      Type::getFloatTy(*TheContext),
      {},
      false 
  );
  TheModule->getOrInsertFunction("unbug", unbugTy);
  
  
  FunctionType *SplitStringTy = FunctionType::get(
      int8PtrTy,
      {int8PtrTy, int8PtrTy},
      false 
  );
  TheModule->getOrInsertFunction("SplitString", SplitStringTy);


  //
  FunctionType *SplitStringIndexateTy = FunctionType::get(
      int8PtrTy,
      {int8PtrTy, int8PtrTy, Type::getFloatTy(*TheContext)},
      false 
  );
  TheModule->getOrInsertFunction("SplitStringIndexate", SplitStringIndexateTy);


  //
  FunctionType *StrToFloatTy = FunctionType::get(
      Type::getFloatTy(*TheContext),
      {int8PtrTy},
      false 
  );
  TheModule->getOrInsertFunction("StrToFloat", StrToFloatTy);


  //
  FunctionType *CopyStringTy = FunctionType::get(
      int8PtrTy,
      {int8PtrTy},
      false 
  );
  TheModule->getOrInsertFunction("CopyString", CopyStringTy);


  //
  FunctionType *appendTy = FunctionType::get(
      Type::getFloatTy(*TheContext),
      {int8PtrTy, int8PtrTy},
      false 
  );
  TheModule->getOrInsertFunction("append", appendTy);


  //
  FunctionType *objAttr_var_from_varTy = FunctionType::get(
      Type::getVoidTy(*TheContext),
      {int8PtrTy, int8PtrTy},
      false 
  );
  TheModule->getOrInsertFunction("objAttr_var_from_var", objAttr_var_from_varTy);


  //
  FunctionType *objAttr_var_from_vecTy = FunctionType::get(
      Type::getVoidTy(*TheContext),
      {int8PtrTy, int8PtrTy},
      false 
  );
  TheModule->getOrInsertFunction("objAttr_var_from_vec", objAttr_var_from_vecTy);


  //
  FunctionType *objAttr_vec_from_varTy = FunctionType::get(
      Type::getVoidTy(*TheContext),
      {int8PtrTy, int8PtrTy},
      false 
  );
  TheModule->getOrInsertFunction("objAttr_vec_from_var", objAttr_vec_from_varTy);


  //
  FunctionType *objAttr_vec_from_vecTy = FunctionType::get(
      Type::getVoidTy(*TheContext),
      {int8PtrTy, int8PtrTy},
      false 
  );
  TheModule->getOrInsertFunction("objAttr_vec_from_vec", objAttr_vec_from_vecTy);


  //
  FunctionType *LoadObjectScopeNameTy = FunctionType::get(
      int8PtrTy,
      {int8PtrTy},
      false 
  );
  TheModule->getOrInsertFunction("LoadObjectScopeName", LoadObjectScopeNameTy);



  //
  FunctionType *PrintStrTy = FunctionType::get(
      Type::getFloatTy(*TheContext),
      {int8PtrTy}, 
      false 
  );
  TheModule->getOrInsertFunction("PrintStr", PrintStrTy);


  //
  FunctionType *PrintStrVecTy = FunctionType::get(
      Type::getFloatTy(*TheContext),
      {int8PtrTy}, 
      false 
  );
  TheModule->getOrInsertFunction("PrintStrVec", PrintStrVecTy);


  //
  FunctionType *PrintFloatVecTy = FunctionType::get(
      Type::getFloatTy(*TheContext),
      {int8PtrTy}, 
      false 
  );
  TheModule->getOrInsertFunction("PrintFloatVec", PrintFloatVecTy);
  

  //
  FunctionType *first_nonzeroTy = FunctionType::get(
      Type::getFloatTy(*TheContext),
      {int8PtrTy}, 
      false 
  );
  TheModule->getOrInsertFunction("first_nonzero", first_nonzeroTy);


  //
  FunctionType *LoadStrVecOnDemandTy = FunctionType::get(
      int8PtrTy,
      {int8PtrTy},
      false
  );
  TheModule->getOrInsertFunction("LoadStrVecOnDemand", LoadStrVecOnDemandTy);


  //
  FunctionType *LoadFloatVecOnDemandTy = FunctionType::get(
      int8PtrTy,
      {int8PtrTy},
      false
  );
  TheModule->getOrInsertFunction("LoadFloatVecOnDemand", LoadFloatVecOnDemandTy);

  
  //
  FunctionType *StoreStrOnDemandTy = FunctionType::get(
      Type::getFloatTy(*TheContext),
      {int8PtrTy, int8PtrTy},
      false
  );
  TheModule->getOrInsertFunction("StoreStrOnDemand", StoreStrOnDemandTy);

  
  //
  FunctionType *LoadStrOnDemandTy = FunctionType::get(
      int8PtrTy,
      {int8PtrTy},
      false
  );
  TheModule->getOrInsertFunction("LoadStrOnDemand", LoadStrOnDemandTy);


  //
  FunctionType *StoreStrVecOnDemandTy = FunctionType::get(
      Type::getFloatTy(*TheContext),
      {int8PtrTy, int8PtrTy},
      false
  );
  TheModule->getOrInsertFunction("StoreStrVecOnDemand", StoreStrVecOnDemandTy);
  
  
  //
  FunctionType *StoreFloatVecOnDemandTy = FunctionType::get(
      Type::getFloatTy(*TheContext),
      {int8PtrTy, int8PtrTy},
      false
  );
  TheModule->getOrInsertFunction("StoreFloatVecOnDemand", StoreFloatVecOnDemandTy);

  FunctionType *StoreFloatVecOnDemandOnIdxTy = FunctionType::get(
      Type::getFloatTy(*TheContext),
      {int8PtrTy, Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext)},
      false
  );
  TheModule->getOrInsertFunction("StoreFloatVecOnDemandOnIdx", StoreFloatVecOnDemandOnIdxTy);


  //
  FunctionType *LenStrVecTy = FunctionType::get(
      Type::getFloatTy(*TheContext),
      {int8PtrTy},
      false 
  );
  TheModule->getOrInsertFunction("LenStrVec", LenStrVecTy);


  //
  FunctionType *ShuffleStrVecTy = FunctionType::get(
      int8PtrTy,
      {int8PtrTy},
      false 
  );
  TheModule->getOrInsertFunction("ShuffleStrVec", ShuffleStrVecTy);


  //
  FunctionType *IndexStrVecTy = FunctionType::get(
      int8PtrTy,
      {int8PtrTy, Type::getFloatTy(*TheContext)}, 
      false 
  );
  TheModule->getOrInsertFunction("IndexStrVec", IndexStrVecTy);


  //
  FunctionType *IndexClassStrVecTy = FunctionType::get(
      int8PtrTy,
      {int8PtrTy, Type::getFloatTy(*TheContext)}, 
      false 
  );
  TheModule->getOrInsertFunction("IndexClassStrVec", IndexClassStrVecTy);

  //
  FunctionType *IndexClassFloatVecTy = FunctionType::get(
      Type::getFloatTy(*TheContext),
      {int8PtrTy, Type::getFloatTy(*TheContext)}, 
      false 
  );
  TheModule->getOrInsertFunction("IndexClassFloatVec", IndexClassFloatVecTy);


  //
  FunctionType *AuxFnTy = FunctionType::get(
      int8PtrTy,
      {int8PtrTy}, 
      false 
  );
  TheModule->getOrInsertFunction("AuxFn", AuxFnTy);


  // char *
  FunctionType *shuffle_strTy = FunctionType::get(
      int8PtrTy,
      {int8PtrTy},
      false 
  );
  TheModule->getOrInsertFunction("shuffle_str", shuffle_strTy);


  //
  FunctionType *InitObjectVecWithNullTy = FunctionType::get(
      Type::getFloatTy(*TheContext),
      {int8PtrTy, Type::getFloatTy(*TheContext)},
      false 
  );
  TheModule->getOrInsertFunction("InitObjectVecWithNull", InitObjectVecWithNullTy);


  //
  FunctionType *is_nullTy = FunctionType::get(
      Type::getFloatTy(*TheContext),
      {int8PtrTy},
      false 
  );
  TheModule->getOrInsertFunction("is_null", is_nullTy);


  //===----------------------------------------------------------------------===//
  // Other Ops
  //===----------------------------------------------------------------------===//


  // 
  FunctionType *FirstArgOnDemandTy = FunctionType::get(
      int8PtrTy,
      {int8PtrTy, int8PtrTy, int8PtrTy, int8PtrTy, Type::getInt32Ty(*TheContext), Type::getInt32Ty(*TheContext), Type::getInt32Ty(*TheContext)},
      false 
  );
  TheModule->getOrInsertFunction("FirstArgOnDemand", FirstArgOnDemandTy);
  

  // 
  FunctionType *objHashTy = FunctionType::get(
      int8PtrTy,
      {int8PtrTy, int8PtrTy},
      false 
  );
  TheModule->getOrInsertFunction("objHash", objHashTy);
  

  // 
  FunctionType *LoadObjectTy = FunctionType::get(
      int8PtrTy,
      {int8PtrTy},
      false 
  );
  TheModule->getOrInsertFunction("LoadObject", LoadObjectTy);
  

  //
  FunctionType *InstantiateObjectTy = FunctionType::get(
      Type::getVoidTy(*TheContext),
      {int8PtrTy, int8PtrTy},
      false 
  );
  TheModule->getOrInsertFunction("InstantiateObject", InstantiateObjectTy);
  

  //
  FunctionType * GetEmptyCharTy = FunctionType::get(
      int8PtrTy,
      {},
      false 
  );
  TheModule->getOrInsertFunction("GetEmptyChar", GetEmptyCharTy);
  

  //
  FunctionType *FreeCharTy = FunctionType::get(
      Type::getVoidTy(*TheContext),
      {int8PtrTy},
      false
  );
  TheModule->getOrInsertFunction("FreeChar", FreeCharTy);
  

  //
  FunctionType *FreeCharFromFuncTy = FunctionType::get(
      Type::getVoidTy(*TheContext),
      {int8PtrTy},
      false
  );
  TheModule->getOrInsertFunction("FreeCharFromFunc", FreeCharFromFuncTy);
  

  //
  FunctionType * ConcatStrTy = FunctionType::get(
      int8PtrTy,
      {int8PtrTy, int8PtrTy},
      false 
  );
  TheModule->getOrInsertFunction("ConcatStr", ConcatStrTy);
  

  //
  FunctionType * ConcatStrFreeLeftTy = FunctionType::get(
      int8PtrTy,
      {int8PtrTy, int8PtrTy},
      false 
  );
  TheModule->getOrInsertFunction("ConcatStrFreeLeft", ConcatStrFreeLeftTy);
  

  //
  FunctionType * ConcatStrFreeRightTy = FunctionType::get(
      int8PtrTy,
      {int8PtrTy, int8PtrTy},
      false 
  );
  TheModule->getOrInsertFunction("ConcatStrFreeRight", ConcatStrFreeRightTy);
  

  //
  FunctionType * ConcatStrFreeTy = FunctionType::get(
      int8PtrTy,
      {int8PtrTy, int8PtrTy},
      false 
  );
  TheModule->getOrInsertFunction("ConcatStrFree", ConcatStrFreeTy);

  
  //
  FunctionType * ConcatNumToStrTy = FunctionType::get(
      int8PtrTy,
      {int8PtrTy, Type::getInt32Ty(*TheContext)},
      false 
  );
  TheModule->getOrInsertFunction("ConcatNumToStr", ConcatNumToStrTy);

  
  //
  FunctionType * ConcatNumToStrFreeTy = FunctionType::get(
      int8PtrTy,
      {int8PtrTy, Type::getInt32Ty(*TheContext)},
      false 
  );
  TheModule->getOrInsertFunction("ConcatNumToStrFree", ConcatNumToStrFreeTy);

  
  //
  FunctionType * AddToScopeCleanListTy = FunctionType::get(
      Type::getVoidTy(*TheContext),
      {int8PtrTy, int8PtrTy, int8PtrTy},
      false 
  );
  TheModule->getOrInsertFunction("AddToScopeCleanList", AddToScopeCleanListTy);

  
  //
  FunctionType *CleanScopeVarsTy = FunctionType::get(
      Type::getVoidTy(*TheContext),
      {int8PtrTy},
      false 
  );
  TheModule->getOrInsertFunction("CleanScopeVars", CleanScopeVarsTy);
  

  //
  FunctionType * RandomStrOnDemandTy = FunctionType::get(
      int8PtrTy,
      {},
      false
  );
  TheModule->getOrInsertFunction("RandomStrOnDemand", RandomStrOnDemandTy);

  
  // 
  FunctionType *StoreOnDemandTy = FunctionType::get(
      Type::getVoidTy(*TheContext),
      {int8PtrTy, int8PtrTy, Type::getFloatTy(*TheContext)},
      false //
  );
  TheModule->getOrInsertFunction("StoreOnDemand", StoreOnDemandTy);

  
  // 
  FunctionType *StoreOnDemandNoFreeTy = FunctionType::get(
      Type::getVoidTy(*TheContext),
      {int8PtrTy, int8PtrTy, Type::getFloatTy(*TheContext)},
      false //
  );
  TheModule->getOrInsertFunction("StoreOnDemandNoFree", StoreOnDemandNoFreeTy);

  
  // 
  FunctionType *StoreArgOnDemandTy = FunctionType::get(
      Type::getVoidTy(*TheContext),
      {int8PtrTy, int8PtrTy, Type::getFloatTy(*TheContext)},
      false //
  );
  TheModule->getOrInsertFunction("StoreArgOnDemand", StoreArgOnDemandTy);
  

  //
  FunctionType *LoadOnDemandTy = FunctionType::get(
      Type::getFloatTy(*TheContext),
      {Type::getInt8Ty(*TheContext)->getPointerTo()},
      false
  );
  TheModule->getOrInsertFunction("LoadOnDemand", LoadOnDemandTy);
  

  //
  FunctionType *LoadOnDemandNoFreeTy = FunctionType::get(
      Type::getFloatTy(*TheContext),
      {Type::getInt8Ty(*TheContext)->getPointerTo()},
      false
  );
  TheModule->getOrInsertFunction("LoadOnDemandNoFree", LoadOnDemandNoFreeTy);
  

  //
  FunctionType *StoreDimsOnDemandTy = FunctionType::get(
      Type::getFloatTy(*TheContext),
      {int8PtrTy, Type::getFloatTy(*TheContext)},
      false
  );
  TheModule->getOrInsertFunction("StoreDimsOnDemand", StoreDimsOnDemandTy);
  

  //
  FunctionType *CreatePinnedTensorOnDemandTy = FunctionType::get(
      Type::getVoidTy(*TheContext),
      {int8PtrTy, int8PtrTy},
      false 
  );
  TheModule->getOrInsertFunction("CreatePinnedTensorOnDemand", CreatePinnedTensorOnDemandTy);


  //
  FunctionType *CreateTensorOnDemandTy = FunctionType::get(
      Type::getFloatTy(*TheContext),
      {int8PtrTy, int8PtrTy, int8PtrTy, Type::getInt32Ty(*TheContext)},
      false 
  );
  TheModule->getOrInsertFunction("CreateTensorOnDemand", CreateTensorOnDemandTy);


  //
  FunctionType *CreateConv2dOnDemandTy = FunctionType::get(
      Type::getFloatTy(*TheContext),
      {int8PtrTy,
       int8PtrTy,
       Type::getFloatTy(*TheContext),
       Type::getFloatTy(*TheContext),
       Type::getFloatTy(*TheContext),
       Type::getFloatTy(*TheContext),
       Type::getFloatTy(*TheContext)},
      false
  );
  TheModule->getOrInsertFunction("CreateConv2dOnDemand", CreateConv2dOnDemandTy);


  //
  FunctionType *CreateBatchNorm2dOnDemandTy = FunctionType::get(
      Type::getFloatTy(*TheContext),
      {int8PtrTy,
       Type::getFloatTy(*TheContext)},
      false
  );
  TheModule->getOrInsertFunction("CreateBatchNorm2dOnDemand", CreateBatchNorm2dOnDemandTy);


  //
  FunctionType *CreateMaxPool2dOnDemandTy = FunctionType::get(
      Type::getFloatTy(*TheContext),
      {int8PtrTy, int8PtrTy,
       Type::getFloatTy(*TheContext),
       Type::getFloatTy(*TheContext),
       Type::getFloatTy(*TheContext)},
      false
  );
  TheModule->getOrInsertFunction("CreateMaxPool2dOnDemand", CreateMaxPool2dOnDemandTy);


  //
  FunctionType *CopyArgTensorTy = FunctionType::get(
      Type::getFloatTy(*TheContext),
      {int8PtrTy, int8PtrTy, int8PtrTy, int8PtrTy}, 
      false 
  );
  TheModule->getOrInsertFunction("CopyArgTensor", CopyArgTensorTy);

  
  //
  FunctionType *RemoveTensorScopeTy = FunctionType::get(
      Type::getFloatTy(*TheContext),
      {int8PtrTy,
       int8PtrTy,
       int8PtrTy,
       int8PtrTy}, 
      false 
  );
  TheModule->getOrInsertFunction("RemoveTensorScope", RemoveTensorScopeTy);


  //
  FunctionType *RemoveTensorScopeAttrOnIndexTy = FunctionType::get(
      Type::getFloatTy(*TheContext),
      {int8PtrTy,
       int8PtrTy,
       int8PtrTy,
       int8PtrTy,
       Type::getFloatTy(*TheContext)}, 
      false 
  );
  TheModule->getOrInsertFunction("RemoveTensorScopeAttrOnIndex", RemoveTensorScopeAttrOnIndexTy);


  //
  FunctionType *AttrTensorTy = FunctionType::get(
      Type::getFloatTy(*TheContext),
      {int8PtrTy,
       int8PtrTy}, 
      false 
  );
  TheModule->getOrInsertFunction("AttrTensor", AttrTensorTy);

  FunctionType *AttrTensorNoFreeTy = FunctionType::get(
      Type::getFloatTy(*TheContext),
      {int8PtrTy,
       floatPtrTy,
       int8PtrTy}, 
      false 
  );
  TheModule->getOrInsertFunction("AttrTensorNoFree", AttrTensorNoFreeTy);
  

  //
  FunctionType *AttrTensorOnIdxTy = FunctionType::get(
      Type::getFloatTy(*TheContext),
      {int8PtrTy,
       int8PtrTy,
       Type::getFloatTy(*TheContext)}, 
      false 
  );
  TheModule->getOrInsertFunction("AttrTensorOnIdx", AttrTensorOnIdxTy);
  

  //
  FunctionType *AttrTensorOnIdxTensorTy = FunctionType::get(
      Type::getFloatTy(*TheContext),
      {int8PtrTy, int8PtrTy, int8PtrTy},
      false 
  );
  TheModule->getOrInsertFunction("AttrTensorOnIdxTensor", AttrTensorOnIdxTensorTy);
  

  //
  FunctionType *cpuTy = FunctionType::get(
      Type::getFloatTy(*TheContext),
      {int8PtrTy},
      false 
  );
  TheModule->getOrInsertFunction("cpu", cpuTy);
  

  //
  FunctionType *cpu_idxTy = FunctionType::get(
      Type::getFloatTy(*TheContext),
      {int8PtrTy,
       Type::getFloatTy(*TheContext)}, 
      false 
  );
  TheModule->getOrInsertFunction("cpu_idx", cpu_idxTy);


  //
  FunctionType *printTTy = FunctionType::get(
      Type::getFloatTy(*TheContext),
      {int8PtrTy},
      false 
  );
  TheModule->getOrInsertFunction("PrintTensor", printTTy);
  
  
  //
  FunctionType *rand_likeTy = FunctionType::get(
      int8PtrTy,
      {int8PtrTy},
      false 
  );
  TheModule->getOrInsertFunction("rand_like", rand_likeTy);

  
  //
  FunctionType *printTy = FunctionType::get(
      Type::getFloatTy(*TheContext),
      {int8PtrTy, Type::getFloatTy(*TheContext)}, 
      false 
  );
  TheModule->getOrInsertFunction("print", printTy);
  

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




static void HandleClass() { ParseClass(); }

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
      fprintf(stderr, "Read extern: ");
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

    // Points __anon_expr
    auto Sym = ExitOnErr(TheJIT->lookup("__anon_expr"));
    //assert(Sym && "Function not found");
      
      
    // Get the symbol's address and cast it to the right type (takes no
    // arguments, returns a float) so we can call it as a native function.
    auto *FP = Sym.getAddress().toPtr<float (*)()>();
    auto fp = FP();
    
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
    //  std::cout << "MAIN LOOP, reading token: " << ReverseToken(CurTok) << "\n";
    

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


  if (pthread_mutex_init(&mutex, NULL) != 0) {
    printf("Mutex initialization failed\n");
    return 1;
  }


  lockVars["mutex"] = &mutex;
  

  
  cudnnCreate(&cudnn);



  InitializeNativeTarget();
  InitializeNativeTargetAsmPrinter(); // Prepare for target hardware
  InitializeNativeTargetAsmParser();

  leaf_ops = {leaf, tensor_leaf, weight_leaf, bias_leaf};
  loss_ops = {cross_entropy_op};
  gradless_ops = {onehot_op, max_op, argmax_op, equal_op};

  // Install standard binary operators.
  // 1 is lowest precedence.
  BinopPrecedence[tok_space] = 1;
  BinopPrecedence['='] = 4;
  BinopPrecedence[':'] = 9;
  BinopPrecedence['>'] = 10;
  BinopPrecedence['<'] = 10;
  BinopPrecedence[tok_equal] = 10;
  BinopPrecedence[tok_diff] = 10;
  BinopPrecedence[tok_minor_eq] = 10;
  BinopPrecedence[tok_higher_eq] = 10;
  BinopPrecedence['+'] = 20;
  BinopPrecedence['-'] = 20;
  BinopPrecedence['/'] = 39;
  BinopPrecedence['*'] = 40;  // highest.
  BinopPrecedence['^'] = 50;
  BinopPrecedence['@'] = 60;


  floatFunctions["log"] = "logE";
  floatFunctions["log2"] = "logE2";
  floatFunctions["log2f"] = "logE2f";
  floatFunctions["round"] = "roundE";
  floatFunctions["floor"] = "floorE";


  stringMethods["split"] = "SplitString";
  stringMethods["split_idx"] = "SplitStringIndexate";




  return_tensor_functions = {"gelu", "sigmoid", "_tanh", "relu", "softmax", "log", "rand_like", "print_tensor"};
  return_tensor_methods = {"view", "clip", "argmax", "tmax", "onehot", "shape", "permute", "cpu",
                            "sum", "prod", "mean", "tmin", "argmin", "topk", "repeat_interleave"};
  return_tensor_fn = concat_str_vec(return_tensor_functions, return_tensor_methods);

  return_pinned_methods = {"gpu", "gpuw"};


  // Universal
  vararg_methods = {"view", "sum", "mean", "prod", "tmax", "argmax"};
  string_methods = {"split", "split_idx"};


  // tensor + string + ...
  // e.g: x.view(), str.split()
  native_methods = {"split", "split_idx", "first_nonzero", "append"};
  native_methods = concat_str_vec(native_methods, return_tensor_methods);
  //native_methods = concat_str_vec(native_methods, return_pinned_methods);

  native_functions = {"ShuffleStrVec", "gload_img", "wload_img", "silent_sleep", "sleep",
                      "LenStrVec", "gpu", "gpuw", "zeros_vec", "ones_vec",
                      "_glob_b_", "print", "cross_entropy", "backprop", "AdamW",
                      "load_preprocess_img", "max", "min", "unbug", "is_null",
                      "cpu_idx", "eval", "train"};
  native_functions = concat_str_vec(native_functions, return_tensor_functions);
  native_fn = concat_str_vec(native_methods, native_functions);


  tensor_inits = {"binary", "int", "randu", "zeros", "ones", "xavu", "xavu_relu", "xavn"};


  // Prime the first token.
  //fprintf(stderr, "ready> ");
  getNextToken();

  TheJIT = ExitOnErr(KaleidoscopeJIT::Create());
  InitializeModule();

  // Run the main "interpreter loop" now.
  

  MainLoop();

  return 0;
}
