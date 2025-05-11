

#include "llvm/IR/Value.h"



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
#include <fstream>
#include <sstream>
#include <filesystem>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>


#include "../common/include.h"
#include "include.h"



using namespace llvm;



/// LogError* - These are little helper functions for error handling.
std::unique_ptr<ExprAST> LogErrorS(std::string Str) {
  ShallCodegen = false;
  //fprintf(stderr, "\033[31m Error: \033[0m%s\n", Str);
  if (Str!=" ")
    std::cout << "\nLine: " << LineCounter << "\n   \033[31m Error: \033[0m " << Str << "\n\n";
  
  
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
  std::cout << "\nLine: " << LineCounter << "\n   \033[33m Aviso: \033[0m " << Str << "\n\n";
}

// Modified LogError function with token parameter
std::unique_ptr<ExprAST> LogErrorT(int CurTok) {
  ShallCodegen = false;
  //char buf[100];
  //snprintf(buf, sizeof(buf), "token %d inesperado.", CurTok);
  //fprintf(stderr, "\033[31mError: \033[0m%s\n", buf);
  std::cout << "\nLine: " << LineCounter << "\n   \033[31m Error: \033[0mUnexpected token " << ReverseToken(CurTok) << ". Expected an expression.\n\n";
  
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


std::unique_ptr<PrototypeAST> LogErrorP_to_comma(std::string Str) {
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



extern "C" void print_codegen(char *msg)
{
  std::cout << "-- print_codegen: " << msg << ".\n";
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
