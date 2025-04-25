#pragma once

#include "llvm/IR/Value.h"


#include <map>
#include <string>
#include <vector>

#include "../data_types/include.h"
#include "../notators/include.h"
#include "../tensor/include.h"
#include "../KaleidoscopeJIT.h"
#include "tokenizer.h"
#include "include.h"



using namespace llvm;



extern std::map<std::string, std::string> ops_type_return;
extern std::map<int, std::string> op_map;


std::unique_ptr<ExprAST> ParseExpression(std::string class_name="", bool can_be_list=true);
std::unique_ptr<ExprAST> ParsePrimary(std::string class_name, bool can_be_list=true);

/// numberexpr ::= number
std::unique_ptr<ExprAST> ParseNumberExpr();

std::unique_ptr<ExprAST> ParseStringExpr(); 



/// parenexpr ::= '(' expression ')'
std::unique_ptr<ExprAST> ParseParenExpr(std::string class_name=""); 



std::unique_ptr<ExprAST> ParseObjectInstantiationExpr(std::string _class, std::string class_name);


std::vector<std::unique_ptr<ExprAST>> ParseIdx(std::string class_name=""); 



/// identifierexpr
///   ::= identifier
///   ::= identifier '(' expression* ')'
std::unique_ptr<ExprAST> ParseIdentifierExpr(std::string class_name="", bool can_be_string=false, bool can_be_list=true); 





/// ifexpr ::= 'if' expression 'then' expression 'else' expression
std::unique_ptr<ExprAST> ParseIfExpr(std::string class_name=""); 




std::vector<std::unique_ptr<ExprAST>> ParseIdentedBodies(int cur_level_tabs, std::string class_name="");



/// forexpr ::= 'for' identifier '=' expr ',' expr (',' expr)? 'in' expression
std::unique_ptr<ExprAST> ParseForExpr(std::string class_name=""); 



/// whileexpr ::= 'while' identifier '=' expr ',' expr (',' expr)? 'in' expression
std::unique_ptr<ExprAST> ParseWhileExpr(std::string class_name=""); 



std::unique_ptr<ExprAST> ParseAsyncExpr(std::string class_name=""); 



std::unique_ptr<ExprAST> ParseFinishExpr(std::string class_name=""); 





std::unique_ptr<ExprAST> ParseNewVector(std::string class_name=""); 


std::unique_ptr<ExprAST> ParseStrVecExpr(); 


std::unique_ptr<ExprAST> ParseSelfExpr(std::string class_name=""); 


std::unique_ptr<ExprAST> ParseChainCallExpr(std::unique_ptr<ExprAST>, std::string class_name); 
  
  


std::unique_ptr<ExprAST> ParsePinnedTensorExpr(); 
   

std::unique_ptr<ExprAST> ParseDataExpr(std::string class_name=""); 


    
  
std::unique_ptr<ExprAST> ParseLSTMExpr(); 
  
  
std::unique_ptr<ExprAST> ParseEmbeddingExpr(); 
  
  
std::unique_ptr<ExprAST> ParseLinearExpr(); 
  
  
  
  
  
  
std::unique_ptr<ExprAST> ParseMHSAExpr(); 
  
  
  

std::unique_ptr<ExprAST> ParseMaxPool2dExpr(); 
  
  
  
  
  
  //
std::unique_ptr<ExprAST> ParseBatchNorm2dExpr(); 
  
  
  
std::unique_ptr<ExprAST> ParseBN2dReluExpr(); 
  
  

std::unique_ptr<ExprAST> ParseReluExpr(); 
  
    
std::unique_ptr<ExprAST> ParseLockExpr(std::string class_name=""); 
  
  
  
std::unique_ptr<ExprAST> ParseNoGradExpr(std::string class_name=""); 
  
  
  
std::unique_ptr<ExprAST> ParseMustBeVar(std::string class_name="", std::string expr_name=""); 
  
  
  
std::unique_ptr<ExprAST> ParseGlobalExpr(std::string class_name=""); 
    
std::unique_ptr<ExprAST> ParseReturnExpr(std::string class_name=""); 
  
  
  
  
  
  /// unary
  ///   ::= primary
  ///   ::= '!' unary
std::unique_ptr<ExprAST> ParseUnary(std::string class_name="", bool can_be_list=true); 
  
  
  /// binoprhs
  ///   ::= ('+' unary)*
std::tuple<std::unique_ptr<ExprAST>, int, std::string> ParseBinOpRHS(int ExprPrec,
                                                std::unique_ptr<ExprAST> LHS,
                                                std::string class_name=""); 
  
  
  /// prototype
  ///   ::= id '(' id* ')'
  ///   ::= binary LETTER number? (id, id)
  ///   ::= unary LETTER (id)
std::unique_ptr<PrototypeAST> ParsePrototype(std::string class_name=""); 
  
  
  
  /// definition ::= 'def' prototype expression
std::unique_ptr<FunctionAST> ParseDefinition(std::string class_name=""); 
  
  
  /// toplevelexpr ::= expression
std::unique_ptr<FunctionAST> ParseTopLevelExpr();
  
  
  
  /// external ::= 'extern' prototype
std::unique_ptr<PrototypeAST> ParseExtern(); 
  
  
  
  
std::unique_ptr<ExprAST> ParseClass(); 
 


