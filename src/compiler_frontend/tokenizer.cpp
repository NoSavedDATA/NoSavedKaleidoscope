
#include<map>
#include<vector>
#include<stdio.h>
#include<stdlib.h>
#include<ctype.h>
#include<string>
#include<iostream>

#include "tokenizer.h"



//===----------------------------------------------------------------------===//
// Lexer
//===----------------------------------------------------------------------===//






// The lexer returns tokens [0-255] if it is an unknown character, otherwise one
// of these for known words.


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

  { tok_post_class_attr_attr, ".attr."},
  { tok_post_class_attr_identifier, ".identifier"},
  
  // var definition
  { tok_var, "float"},
  { tok_tensor, "tensor"},
  { tok_var_str, "var str"},
  { tok_attr_var, "tok attr var"},
  { tok_attr_tensor, "tok attr tensor"},
  { tok_conv2d, "Conv2d"},
  { tok_lstm, "LSTM"},
  { tok_embedding, "Embedding"},
  { tok_maxpool2d, "MaxPool2d"},
  { tok_avgpool2d, "AvgPool2d"},
  { tok_batchnorm2d, "BatchNorm2d"},
  { tok_bn2drelu, "BN2dRelu"},
  { tok_relu, "Relu"},

  { tok_global, "global"},
  { tok_no_grad, "no_grad"},

  { tok_mhsa, "MHSA"},
  { tok_linear, "Linear"},
  

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


std::string IdentifierStr; // Filled in if tok_identifier
float NumVal;             // Filled in if tok_number

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
  // std::cout << "\nGet token. " << " last char: " << LastChar << ".\n";

  

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
      if (IdentifierStr == "LSTM")
      {
        LastChar = getchar();
        return tok_lstm;
      }
      if (IdentifierStr == "MHSA")
      {
        LastChar = getchar();
        return tok_mhsa;
      }
      if (IdentifierStr == "Linear")
      {
        LastChar = getchar();
        return tok_linear;
      }
      if (IdentifierStr == "Embedding")
      {
        LastChar = getchar();
        return tok_embedding;
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
      if (IdentifierStr == "BN2dRelu")
      {
        LastChar = getchar();
        return tok_bn2drelu;
      }
      if (IdentifierStr == "Relu")
      {
        LastChar = getchar();
        return tok_relu;
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
    if (IdentifierStr == "global")
      return tok_global;
    if (IdentifierStr == "no_grad")
      return tok_no_grad;
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
    if (IdentifierStr == "sleep")
      IdentifierStr = "__slee_p_";
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



/// CurTok/getNextToken - Provide a simple token buffer.  CurTok is the current
/// token the parser is looking at.  getNextToken reads another token from the
/// lexer and updates CurTok with its results.
int CurTok;
int getNextToken() { return CurTok = get_token(); }


/// BinopPrecedence - This holds the precedence for each binary operator that is
/// defined.
std::map<char, int> BinopPrecedence;
/// get_tokenPrecedence - Get the precedence of the pending binary operator token.
int get_tokenPrecedence() {
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