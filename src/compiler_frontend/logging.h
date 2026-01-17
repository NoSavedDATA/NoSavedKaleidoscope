#pragma once
#include "expressions.h"
#include "logging_execution.h"



/// LogError* - These are little helper functions for error handling.
std::unique_ptr<ExprAST> LogErrorS(int, std::string Str);

std::unique_ptr<ExprAST> LogError(int, std::string Str); 

void LogBlue(std::string);

std::unique_ptr<ExprAST> LogError_toNextToken(int, std::string Str); 

std::unique_ptr<ExprAST> LogErrorNextBlock(int line, std::string Str);
std::unique_ptr<ExprAST> LogErrorNextFloatingBlock(int line, std::string Str);


std::unique_ptr<ExprAST> LogErrorBreakLine(int, std::string Str); 

void LogWarning(const char *Str); 

// Modified LogError function with token parameter
std::unique_ptr<ExprAST> LogErrorT(int line, int CurTok); 


std::unique_ptr<PrototypeAST> LogErrorP(int ,const char *Str); 



std::unique_ptr<PrototypeAST> LogErrorP_to_comma(int, std::string Str);

Value *LogErrorV(int, std::string Str); 


extern "C" void print_codegen(char *msg);

void p2t(std::string msg);
void p2n();
