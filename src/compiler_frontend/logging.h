#pragma once
#include "expressions.h"


/// LogError* - These are little helper functions for error handling.
std::unique_ptr<ExprAST> LogErrorS(std::string Str);

std::unique_ptr<ExprAST> LogError(std::string Str); 

void LogBlue(std::string);

std::unique_ptr<ExprAST> LogError_toNextToken(std::string Str); 


std::unique_ptr<ExprAST> LogErrorBreakLine(std::string Str); 

void LogWarning(const char *Str); 

// Modified LogError function with token parameter
std::unique_ptr<ExprAST> LogErrorT(int CurTok); 


std::unique_ptr<PrototypeAST> LogErrorP(const char *Str); 



std::unique_ptr<PrototypeAST> LogErrorP_to_comma(std::string Str);

Value *LogErrorV(std::string Str); 


extern "C" void print_codegen(char *msg);

void p2t(std::string msg);
void p2n();
