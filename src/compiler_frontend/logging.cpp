

#include "llvm/IR/Value.h"



#include <algorithm>
#include <cstdarg>
#include <cassert>
#include <cctype>
#include <cstring>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <map>
#include <memory>
#include <string>
#include <iostream>
#include <numeric>
#include <utility>
#include <iomanip>
#include <math.h>
#include <fenv.h>
#include <tuple>
#include <chrono>
#include <thread>
#include <random>
#include <float.h>
#include <fstream>
#include <sstream>
#include <filesystem>
#include <stdio.h>
#include <stdlib.h>
#include <vector>



namespace fs = std::filesystem;

#include "../common/include.h"
#include "include.h"



using namespace llvm;



/// LogError* - These are little helper functions for error handling.
std::unique_ptr<ExprAST> LogErrorS(int line, std::string Str) {
  Shall_Exit = true;

  
  if(line!=-1)
  {
    if (tokenizer.current_file!="main file") {
      std::cout << "\n\n" << tokenizer.current_file << "\n";

      
      
      std::ifstream file;
      std::string str_line;
      
      int l=0;
      file.open(tokenizer.current_file);
      while(l<line&&std::getline(file, str_line))
        l++;

      file.close();
      printf("%s", str_line.c_str());
    }

    std::cout << "\nLine: " << line << "\n   ";
  } else
    std::cout << "\n\n" << tokenizer.current_file << "\n";


  if (Str!=" ")
    std::cout << "\033[31m Error: \033[0m " << Str << "\n\n";
  
  
  return nullptr;
}


void LogBlue(std::string msg) {
  std::cout << "\n\t\033[34m Log: \033[0m " << msg << "\n\n";
}


std::unique_ptr<ExprAST> LogError(int line, std::string Str) {
  //fprintf(stderr, "\033[31m Error: \033[0m%s\n", Str);
  LogErrorS(line, Str);

  while(CurTok!=tok_space && CurTok!=',' && CurTok!=')' && !in_char(CurTok, terminal_tokens))
    getNextToken();
  
  return nullptr;
}


std::unique_ptr<ExprAST> LogError_toNextToken(int line, std::string Str) {
  //fprintf(stderr, "\033[31m Error: \033[0m%s\n", Str);
  LogErrorS(line, Str);

  getNextToken();
  
  return nullptr;
}


std::unique_ptr<ExprAST> LogErrorBreakLine(int line, std::string Str) {
  //fprintf(stderr, "\033[31m Error: \033[0m%s\n", Str);
  LogErrorS(line, Str);

  while(CurTok!=tok_space && !in_char(CurTok, terminal_tokens))
    getNextToken();

  if (CurTok==tok_space)
    getNextToken();
  
  return nullptr;
}

std::unique_ptr<ExprAST> LogErrorNextBlock(int line, std::string Str) {
  //fprintf(stderr, "\033[31m Error: \033[0m%s\n", Str);
  LogErrorS(line, Str);



  while(((CurTok!=tok_def || SeenTabs>0) && CurTok!=tok_class && CurTok!=tok_main && CurTok!=tok_eof && CurTok!=tok_finish))
    getNextToken();

  if (CurTok==tok_space)
    getNextToken();
  
  return nullptr;
}

std::unique_ptr<ExprAST> LogErrorNextFloatingBlock(int line, std::string Str) {
  //fprintf(stderr, "\033[31m Error: \033[0m%s\n", Str);
  LogErrorS(line, Str);



  while(((CurTok!=tok_def) && CurTok!=tok_class && CurTok!=tok_main && CurTok!=tok_eof && CurTok!=tok_finish))
    getNextToken();

  if (CurTok==tok_space)
    getNextToken();
  
  return nullptr;
}

void LogWarning(const char *Str) {
  std::cout << "\nLine: " << LineCounter << "\n   \033[33m Aviso: \033[0m " << Str << "\n\n";
}

// Modified LogError function with token parameter
std::unique_ptr<ExprAST> LogErrorT(int line, int CurTok) {
  Shall_Exit = true;
  //char buf[100];
  //snprintf(buf, sizeof(buf), "token %d inesperado.", CurTok);
  //fprintf(stderr, "\033[31mError: \033[0m%s\n", buf);
  std::cout << "\n\n" << tokenizer.current_file << "\n";
  std::cout << "\nLine: " << line << "\n   \033[31m Error: \033[0mUnexpected token " << ReverseToken(CurTok) << ". Expected an expression.\n\n";

  // while (true) {
  //   char c = tokenizer.get();
  //   std::cout << "c is " << c << ".\n";
  //   // tokenizer.get_word() >> tokenizer.token;
  //   // std::cout << "tokenized: " << tokenizer.token << ".\n";
  // }
  
  while(CurTok!=tok_space && !in_char(CurTok, terminal_tokens))
    getNextToken();

  return nullptr;
}


std::unique_ptr<PrototypeAST> LogErrorP(int line, const char *Str) {
  LogError(line, Str);
  while(CurTok!=tok_space && !in_char(CurTok, terminal_tokens))
    getNextToken();
  return nullptr;
}


std::unique_ptr<PrototypeAST> LogErrorP_to_comma(int line, std::string Str) {
  LogError(line, Str);
  while(CurTok!=tok_space && CurTok!=',' && CurTok!=')' && !in_char(CurTok, terminal_tokens))
  {
    std::cout << "LogErrorP: " << IdentifierStr << "\n";
    
    getNextToken();
  }
  return nullptr;
}

Value *LogErrorV(int line, std::string Str) {
  LogErrorS(line, Str);
  return nullptr;
}


extern "C" void LogErrorCall(int line, char *msg) {
  LogErrorS(line, msg);
}



extern "C" void print_codegen(char *msg)
{
  std::cout << "~~" << msg << ".\n";
}


extern "C" void print_codegen_silent(char *msg)
{
  std::cout << msg << "\n";
}

void p2n() {

  call("print_codegen_silent", {Builder->CreateGlobalString("...")});
}

void p2t(std::string msg)
{
  // return;
 

  bool shall_log = true;
  std::vector<std::string> ignore_expresions = {"FunctionAST", "CallExpr", "DataExpr", "p2t", "VariableExpr", "NumberExpr", "VecIdxExpr", "codegenAsyncFunction", "AsyncExpr"};

  for (int i=0; i<ignore_expresions.size(); ++i)
  {
    if ((msg.find(ignore_expresions[i]) != std::string::npos))
      shall_log = false;
  }

  if (shall_log)
  {
    std::cout << msg << ".\n";
    call("print_codegen", {Builder->CreateGlobalString(msg)});
  }
}
