
#include <ctype.h>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <stack>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <vector>

#include "../common/extension_functions.h"
#include "logging_v.h"
#include "tokenizer.h"




namespace fs = std::filesystem;




//===----------------------------------------------------------------------===//
// Lexer
//===----------------------------------------------------------------------===//



Tokenizer::Tokenizer() : current(&std::cin) {
  files.push("main file");
  dirs.push(current_dir);
  line_counters.push(1);
}




// The lexer returns tokens [0-255] if it is an unknown character, otherwise one
// of these for known words.


std::map<int, std::string> token_to_string = {
  { tok_eof, "eof" },
  { tok_finish, "finish" },

  // functions/classes
  { tok_def, "def" },
  { tok_class, "class" },
  { tok_import, "import" },
  { tok_self, "self" },
  { tok_class_attr, "object attr" },
  { tok_extern, "extern" },

  // primary
  { tok_identifier, "tok identifier" },
  { tok_number, "tok number" },
  { tok_str, "tok str `` ''" },
  { tok_var, "var" },

  

  // control
  { tok_if, "if" },
  { tok_then, "then" },
  { tok_else, "else" },
  { tok_for, "for" },
  { tok_while, "while" },
  { tok_spawn, "spawn" },
  { tok_channel, "channel" },
  { tok_async, "async" },
  { tok_asyncs, "asyncs" },
  { tok_async_finish, "finish finish/async" },
  { tok_tab, "tok tab" },
  { tok_return, "tok return"},
  { tok_tuple, "tok tuple"},
  { tok_list, "tok list"},
  { tok_array, "tok array"},
  { tok_dict, "tok dict"},
  { tok_as, "tok as"},
  { tok_in, "tok in"},


  // operators
  { tok_binary, "tok binary" },
  { tok_unary, "tok unary" },

  { tok_main , "tok main" },


  { tok_space, "tok_space" },

  { tok_post_class_attr_attr, ".attr."},
  { tok_post_class_attr_identifier, ".identifier"},
  
  // var definition
  { tok_attr_var, "tok attr var"},
  { tok_attr_tensor, "tok attr tensor"},

  { tok_global, "global"},
  { tok_no_grad, "no_grad"},

  { tok_arrow, "<-"},

  

  { 10, "tok space"},
  { 32, "blank space"},
  { 13, "carriage return"},


  { tok_and, "tok and" },
  { tok_not, "tok not" },
  { tok_or, "tok or" },
  { tok_xor, "tok xor" },
  
  { '.', "dot<.>" },

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
  { tok_int_div, "//" },
  { tok_higher_eq, ">=" },
  { tok_minor_eq, "<=" },
  { tok_mod, "//" },


  { static_cast<int>(','), "," },
  { static_cast<int>(';'), ";" },
  { static_cast<int>(':'), ":" },

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
std::vector<char> terminal_tokens = {';', tok_def, tok_extern, tok_class, tok_eof};


extern std::vector<std::string> LLVM_IR_Functions = {"pow", "sqrt"};

std::vector<std::string> data_tokens = {"tensor", "pinned_tensor", "int", "bool", "str", "str_vec", "float_vec", "MHSA", "LSTM", "Linear", "tuple",
										"list", "dict", "array",
                                        "Embedding", "EmbeddingLn", "Conv2d", "Pool2d", "BatchNorm2d", "float", "int_vec"};
std::vector<std::string> compound_tokens = {"tuple", "list", "array", "dict"};
std::vector<std::string> primary_data_tokens = {"int", "float", "bool", "foreach_control_var"};



std::unordered_map<std::string, uint16_t> data_name_to_type = {{"int", 2}, {"float", 3}, {"bool", 4}, {"str", 5}, {"list", 6},
                                                               {"tuple", 7}, {"dict", 8}, {"channel", 9}, {"int_vec", 10},
                                                               {"float_vec", 11}, {"array", 12}};
std::unordered_map<uint16_t, std::string> data_type_to_name = {{2, "int"}, {3, "float"}, {4, "bool"}, {5, "str"}, {6, "list"},
                                                               {7, "tuple"}, {8, "dict"}, {9, "channel"}, {10, "int_vec"},
                                                               {11, "float_vec"}, {12, "array"}};

uint16_t data_type_count=12;


std::map<std::string, char> string_tokens = {{"var", tok_var}, {"self", tok_self}, {"def", tok_def}, {"class", tok_class}, {"extern", tok_extern},
                                             {"import", tok_import}, {"if", tok_if}, {"then", tok_then}, {"else", tok_else}, {"for", tok_for},
										     {"while", tok_while}, {"async", tok_async}, {"asyncs", tok_asyncs}, {"finish", tok_async_finish},
											 {"in", tok_in}, {"global", tok_global}, {"no_grad", tok_no_grad}, {"lock", tok_lock},
											 {"unlock", tok_unlock}, {"binary", tok_binary}, {"unary", tok_unary}, {"return", tok_ret},
											 {"as", tok_as}, {"spawn", tok_spawn}, {"channel", tok_channel}, {"main", tok_main}, {"and", tok_and},
										     {"not", tok_not}, {"or", tok_or}, {"xor", tok_xor}};

std::string IdentifierStr; // Filled in if tok_identifier
float NumVal;             // Filled in if tok_number
bool BoolVal;

std::string ReverseToken(int _char)
{
  /*
  if (_char>=48 && _char<=57) // Handle number
    return std::to_string(NumVal);
  */
  if (_char==tok_identifier||_char==tok_data||_char==tok_struct)
    return IdentifierStr;

  return token_to_string[_char];
}

int LineCounter = 1;

int SeenTabs = 0;
int LastSeenTabs = 0;



std::istream& Tokenizer::get_word() {
    while ((current == nullptr || current->eof()) && !inputStack.empty()) {
        inputStack.pop();
        current = inputStack.empty() ? nullptr : inputStack.top().get();
    }
    return *current;
}



std::string cur_line = "";

char Tokenizer::get() {
    while (true) {
        if (!current) return tok_eof;
 
        char c = current->get();
        if (c != EOF) {
          cur_line += c;
          cur_c = c;
          return c;
        }

        // Handle EOF
        if (!inputStack.empty()) {
            bool must_return = false;

            inputStack.pop();
            dirs.pop(); 
            files.pop();
            line_counters.pop();
            if (inputStack.empty()&&has_main)
              must_return=true;
            current = inputStack.empty() ? &std::cin : inputStack.top().get();
            current_dir = dirs.top();
            current_file = files.top();
            LineCounter = line_counters.top();

            if(must_return)
              return tok_eof;
            
            
            // Don't return EOF here - immediately try reading from the new source
        } else if (has_main && current!=&std::cin) {
            current = &std::cin;
            // Don't return EOF here - immediately try reading from std::cin
        } else {
            // We're already at std::cin and got EOF - this is a real EOF
            current = nullptr;
            return tok_eof;
        }
    }
}


bool Tokenizer::openFile(std::string filename) {
    has_main=true;
    auto file = std::make_unique<std::ifstream>(filename);
    if (!file->is_open())
    {
      LogErrorC(-1, "Failed to open file: " + filename);
      return false;
    }
    
    current_file = filename;
    std::string base = fs::path(filename).parent_path().string();
    if(filename[0]=='/')
      current_dir = base;
    else
      current_dir = current_dir + "/" + base;


    // Then push the new file
    inputStack.push(std::move(file));
    dirs.push(current_dir);
    files.push(current_file);
    current = inputStack.top().get(); // this get() turns std::unique_ptr<> as *

    line_counters.top() = LineCounter;
    line_counters.push(1);

    return true;
}

bool Tokenizer::importFile(std::string filename, int dots) {
    
    auto file = std::make_unique<std::ifstream>(filename);
    if (!file->is_open())
    {
      std::cout << "" << current_dir << ".\n";
      LogErrorC(-1, "Failed to open library: " + filename);
      return false;
    }
    
    current_file = filename;
    current_dir = fs::path(filename).parent_path().string();


    // Then push the new file
    inputStack.push(std::move(file));
    dirs.push(current_dir);
    files.push(current_file);
    current = inputStack.top().get(); // this get() turns std::unique_ptr<> as *

    line_counters.top() = LineCounter;
    line_counters.push(1);

    return true;
}







Tokenizer tokenizer = Tokenizer();




/// get_token - Return the next token from standard input.
static int get_token() {
  static int LastChar = ' ';





  // Skip any whitespace and backspace.  
  while (LastChar==32 || LastChar==tok_tab || LastChar==13)
    LastChar = tokenizer.get();
  
    

  if (LastChar=='[')
  {
    LastChar = tokenizer.get();
    return '[';
  }

  

  if (LastChar=='"')
  {

    LastChar = tokenizer.get();
    if (LastChar=='"') {
      IdentifierStr = "";
      LastChar = tokenizer.get();
      return tok_str;
    }
    IdentifierStr = LastChar;

    while (true)
    {
      LastChar = tokenizer.get();
      if(LastChar=='"')
        break;
      IdentifierStr += LastChar;
    }
    LastChar = tokenizer.get();
    
    return tok_str;
  }


  if(LastChar=='.') 
  {
    LastChar = tokenizer.get();
    return '.';
  }
  

  if (isalpha(LastChar) || LastChar=='_') { // identifier: [a-zA-Z][a-zA-Z0-9]*
    IdentifierStr = LastChar;
    bool name_ok=true;
    while(true)
    {
      LastChar = tokenizer.get();

      if (LastChar=='['||LastChar=='.')
        break;
      
      if(isalnum(LastChar) || LastChar=='_')
      {
        IdentifierStr += LastChar;
        continue;
      }
        
      break;
    }


 
    if (in_str(IdentifierStr, compound_tokens))
      return tok_struct;
    if (in_str(IdentifierStr, data_tokens))
      return tok_data;
    if(string_tokens.count(IdentifierStr)>0)
      return string_tokens[IdentifierStr];
    if (IdentifierStr=="true"||IdentifierStr=="false")
    {
      if(IdentifierStr=="true")
        BoolVal = true;
      if(IdentifierStr=="false")
        BoolVal = false;
      return tok_bool;
    }
    if (IdentifierStr == "glob")
      IdentifierStr = "_glob_b_";
    if (IdentifierStr == "sleep")
      IdentifierStr = "__slee_p_";
    if (IdentifierStr == "tanh")
      IdentifierStr = "_tanh";
    return tok_identifier;
  }


  // if (LastChar=='@') {
  //   LastChar = tokenizer.get();

  //   std::string NumStr;
  //   do {
  //     NumStr += LastChar;
  //     LastChar = tokenizer.get();
  //   } while(isdigit(LastChar));

  //   NumVal = strtod(NumStr.c_str(), nullptr);

  //   return tok_int;
  // }

  if (isdigit(LastChar)) { // Number: [-.]+[0-9.]+
    bool is_float=false;
    
    std::string NumStr;
    if (LastChar == '-') { // Check for optional minus sign
      NumStr += LastChar;
      LastChar = tokenizer.get();
    }
    do {
      if(LastChar=='.')
      {
        
        is_float=true;
      }
      NumStr += LastChar;
      LastChar = tokenizer.get();
    } while (isdigit(LastChar) || LastChar == '.');

    NumVal = strtod(NumStr.c_str(), nullptr);

    return (is_float) ? tok_number : tok_int;
  }

  if (LastChar == '#') {
    // Comment until end of line.
    do
    {
      LastChar = tokenizer.get();
      if (LastChar==10)
        LineCounter++;
    }
    while (LastChar != EOF && LastChar != '\n' && LastChar != 10 && LastChar != '\r');

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
      // std::cout << "Process Line Feed"  << ".\n";
      if(LastChar==10)
        LineCounter++;
      if(ThisChar==10)
      {
        cur_line = "";
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
      LastChar = tokenizer.get(); 
      // std::cout << "Line Feed post: " << LastChar  << ".\n";
    }
    //std::cout << "\nThisChar: " << ThisChar << " LastChar " << LastChar << "\n";

    return tok_space;
  }


  LastChar = tokenizer.get();
  int otherChar = LastChar;


  if(ThisChar=='<' && LastChar=='-')
  {
    LastChar = tokenizer.get();
    return tok_arrow;
  }

  if (ThisChar=='=' && otherChar=='=')
  {
    LastChar = tokenizer.get();
    return tok_equal;
  }
  if (ThisChar=='!' && otherChar=='=')
  {
    LastChar = tokenizer.get();
    return tok_diff;
  }
  if (ThisChar=='>' && otherChar=='=')
  {
    LastChar = tokenizer.get();
    return tok_higher_eq;
  }
  if (ThisChar=='<' && otherChar=='=')
  {
    LastChar = tokenizer.get();
    return tok_minor_eq;
  }

  if((ThisChar=='/')&&(otherChar == '/')){
    LastChar = tokenizer.get();
    return tok_int_div;
  }

  //std::cout << "Post char: " << ReverseToken(ThisChar) << "\n";

  // else: return ascii number of the character.
  return ThisChar;
}



/// CurTok/getNextToken - Provide a simple token buffer.  CurTok is the current
/// token the parser is looking at.  getNextToken reads another token from the
/// lexer and updates CurTok with its results.
int CurTok;
int getNextToken() {
  CurTok = get_token(); 
  // std::cout << "\nLine: " << cur_line << "\n";
  return CurTok;
}




void get_tok_util_space() {
  if(CurTok!=tok_space&&tokenizer.cur_c!=10) // gets until the \n before switching files
  {
    // std::cout << "CurTok: " << ReverseToken(CurTok) << " / " << CurTok  << " / " << std::to_string(int(tokenizer.cur_c)) << ".\n";
    char c=' ';
    while(c!=10)
    {
      int _c = c;
      c = tokenizer.get();

      // std::cout << "Get " << c << ".\n";
    }
  }
  CurTok = tok_space;
  // LogBlue("Line: " + cur_line);
  cur_line = "";
  LineCounter++;
}


// void get_tok_util_dot_or_space() {
//   char c=' ';
//   IdentifierStr="";
//   if(!(CurTok==tok_space||tokenizer.cur_c==10))
//   {
//     while(c!=10&&c!='.') {
//       c = tokenizer.get();
//       if(c!=32&&c!='.')
//         IdentifierStr += c;
//     }
//   } else
//     c = tokenizer.cur_c;
//   if(c==10) {
//     CurTok = tok_space;
//     cur_line = "";
//     LineCounter++;
//   }
//   if(c=='.')
//     CurTok='.';
// }


/// BinopPrecedence - This holds the precedence for each binary operator that is
/// defined.
std::map<char, int> BinopPrecedence;
/// get_tokenPrecedence - Get the precedence of the pending binary operator token.
int get_tokenPrecedence() {
  if (CurTok==tok_space)
  {
    if (CurTok==10)
      LineCounter++;
    return 1;
  }


  if (BinopPrecedence.find(CurTok) == BinopPrecedence.end()) // if not found
    return -1;

  // Make sure it's a declared binop.
  int TokPrec = BinopPrecedence[CurTok];
  if (TokPrec <= 0)
    return -1;
  return TokPrec;
}
