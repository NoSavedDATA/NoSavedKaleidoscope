#pragma once

#include "llvm/IR/Value.h"


#include <map>
#include <string>
#include <vector>

#include "../data_types/include.h"
#include "../notators/include.h"
#include "../tensor/include.h"
#include "../KaleidoscopeJIT.h"
#include "parser_struct.h"
#include "tokenizer.h"
#include "include.h"



using namespace llvm;



extern std::map<std::string, std::string> ops_type_return;
extern std::map<int, std::string> op_map;









std::unique_ptr<ExprAST> ParseExpression(Parser_Struct parser_struct, std::string class_name="", bool can_be_list=true);
std::unique_ptr<ExprAST> ParsePrimary(Parser_Struct parser_struct, std::string class_name, bool can_be_list=true);

/// numberexpr ::= number
std::unique_ptr<ExprAST> ParseNumberExpr(Parser_Struct parser_struct);

std::unique_ptr<ExprAST> ParseStringExpr(Parser_Struct parser_struct); 



/// parenexpr ::= '(' expression ')'
std::unique_ptr<ExprAST> ParseParenExpr(Parser_Struct parser_struct, std::string class_name=""); 



std::unique_ptr<ExprAST> ParseObjectInstantiationExpr(Parser_Struct parser_struct, std::string _class, std::string class_name);


std::vector<std::unique_ptr<ExprAST>> ParseIdx(Parser_Struct parser_struct, std::string class_name=""); 



/// identifierexpr
///   ::= identifier
///   ::= identifier '(' expression* ')'
std::unique_ptr<ExprAST> ParseIdentifierExpr(Parser_Struct parser_struct, std::string class_name="", bool can_be_string=false, bool can_be_list=true); 





/// ifexpr ::= 'if' expression 'then' expression 'else' expression
std::unique_ptr<ExprAST> ParseIfExpr(Parser_Struct parser_struct, std::string class_name=""); 




std::vector<std::unique_ptr<ExprAST>> ParseIdentedBodies(Parser_Struct parser_struct, int cur_level_tabs, std::string class_name="");



/// forexpr ::= 'for' identifier '=' expr ',' expr (',' expr)? 'in' expression
std::unique_ptr<ExprAST> ParseForExpr(Parser_Struct parser_struct, std::string class_name=""); 



/// whileexpr ::= 'while' identifier '=' expr ',' expr (',' expr)? 'in' expression
std::unique_ptr<ExprAST> ParseWhileExpr(Parser_Struct parser_struct, std::string class_name=""); 



std::unique_ptr<ExprAST> ParseAsyncExpr(Parser_Struct parser_struct, std::string class_name=""); 



std::unique_ptr<ExprAST> ParseFinishExpr(Parser_Struct parser_struct, std::string class_name=""); 





std::unique_ptr<ExprAST> ParseNewVector(Parser_Struct parser_struct, std::string class_name=""); 


std::unique_ptr<ExprAST> ParseStrVecExpr(Parser_Struct parser_struct); 


std::unique_ptr<ExprAST> ParseSelfExpr(Parser_Struct parser_struct, std::string class_name=""); 


std::unique_ptr<ExprAST> ParseChainCallExpr(Parser_Struct parser_struct, std::unique_ptr<ExprAST>, std::string class_name); 
  
  


std::unique_ptr<ExprAST> ParsePinnedTensorExpr(Parser_Struct parser_struct); 
   

std::unique_ptr<ExprAST> ParseDataExpr(Parser_Struct parser_struct, std::string class_name=""); 


    
  
std::unique_ptr<ExprAST> ParseLSTMExpr(Parser_Struct parser_struct); 
  
  
std::unique_ptr<ExprAST> ParseEmbeddingExpr(Parser_Struct parser_struct); 
  
  
std::unique_ptr<ExprAST> ParseLinearExpr(Parser_Struct parser_struct); 
  
  
  
  
  
  
std::unique_ptr<ExprAST> ParseMHSAExpr(Parser_Struct parser_struct); 
  
  
  

std::unique_ptr<ExprAST> ParseMaxPool2dExpr(Parser_Struct parser_struct); 
  
  
  
  
  
  //
std::unique_ptr<ExprAST> ParseBatchNorm2dExpr(Parser_Struct parser_struct); 
  
  
  
std::unique_ptr<ExprAST> ParseBN2dReluExpr(Parser_Struct parser_struct); 
  
  

std::unique_ptr<ExprAST> ParseReluExpr(Parser_Struct parser_struct); 
  
    
std::unique_ptr<ExprAST> ParseLockExpr(Parser_Struct parser_struct, std::string class_name=""); 
  
  
  
std::unique_ptr<ExprAST> ParseNoGradExpr(Parser_Struct parser_struct, std::string class_name=""); 
  
  
  
std::unique_ptr<ExprAST> ParseMustBeVar(std::string class_name="", std::string expr_name=""); 
  
  
  
std::unique_ptr<ExprAST> ParseGlobalExpr(Parser_Struct parser_struct, std::string class_name=""); 
    
  
  
  
  
  
  /// unary
  ///   ::= primary
  ///   ::= '!' unary
std::unique_ptr<ExprAST> ParseUnary(Parser_Struct parser_struct, std::string class_name="", bool can_be_list=true); 
  
  
  /// binoprhs
  ///   ::= ('+' unary)*
std::tuple<std::unique_ptr<ExprAST>, int, std::string> ParseBinOpRHS(Parser_Struct parser_struct, int ExprPrec,
                                                std::unique_ptr<ExprAST> LHS,
                                                std::string class_name=""); 
  
  
  /// prototype
  ///   ::= id '(' id* ')'
  ///   ::= binary LETTER number? (id, id)
  ///   ::= unary LETTER (id)
std::unique_ptr<PrototypeAST> ParsePrototype(Parser_Struct parser_struct); 
  
  
  
  /// definition ::= 'def' prototype expression
std::unique_ptr<FunctionAST> ParseDefinition(Parser_Struct parser_struct, std::string class_name=""); 
  
  
  /// toplevelexpr ::= expression
std::unique_ptr<FunctionAST> ParseTopLevelExpr(Parser_Struct parser_struct);
  
  
  
  /// external ::= 'extern' prototype
std::unique_ptr<PrototypeAST> ParseExtern(Parser_Struct parser_struct); 
  
  
  
  
std::unique_ptr<ExprAST> ParseClass(); 
 


