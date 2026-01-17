#pragma once

#include "llvm/IR/Value.h"


#include <map>
#include <string>
#include <unordered_map>
#include <vector>

#include "../data_types/include.h"
#include "../KaleidoscopeJIT.h"
#include "../notators/include.h"
#include "parser_struct.h"
#include "tokenizer.h"

#include "include.h"



using namespace llvm;


enum ChannelDirection {
    ch_sender = 0,
    ch_receiver = 1,
    ch_both = 2
};

extern std::map<std::string, std::string> elements_type_return, ops_type_return;
extern std::map<int, std::string> op_map;
extern std::vector<std::string> op_map_names;

extern std::map<std::string, std::map<std::string, Data_Tree>> Object_toClass;


extern std::map<std::string, std::map<std::string, Data_Tree>> data_typeVars;
extern std::map<std::string, std::map<std::string, std::string>> typeVars;



extern std::map<std::string, std::map<std::string, int>> ClassVariables;
extern std::map<std::string, llvm::Type *> ClassStructs;

extern std::map<std::string, std::map<std::string, int>> ChannelDirections;
// extern std::map<std::string, std::vector<std::unique_ptr<ExprAST>>> asyncs_body;




std::string Extract_List_Suffix(const std::string&);
std::string Extract_List_Prefix(const std::string& input);


std::unique_ptr<ExprAST> ParseExpression(Parser_Struct parser_struct, std::string class_name="", bool can_be_list=true);
std::unique_ptr<ExprAST> ParsePrimary(Parser_Struct parser_struct, std::string class_name, bool can_be_list=true);

/// numberexpr ::= number
std::unique_ptr<ExprAST> ParseNumberExpr(Parser_Struct parser_struct);

std::unique_ptr<ExprAST> ParseStringExpr(Parser_Struct parser_struct); 



/// parenexpr ::= '(' expression ')'
std::unique_ptr<ExprAST> ParseParenExpr(Parser_Struct parser_struct, std::string class_name=""); 



std::unique_ptr<ExprAST> ParseObjectInstantiationExpr(Parser_Struct parser_struct, std::string _class, std::string class_name);



std::unique_ptr<IndexExprAST> ParseIdx(Parser_Struct parser_struct, std::string class_name="");




std::optional<std::vector<std::unique_ptr<ExprAST>>> Parse_Arguments(Parser_Struct parser_struct, std::string class_name);



/// ifexpr ::= 'if' expression 'then' expression 'else' expression
std::unique_ptr<ExprAST> ParseIfExpr(Parser_Struct parser_struct, std::string class_name=""); 




std::vector<std::unique_ptr<ExprAST>> ParseIdentedBodies(Parser_Struct parser_struct, int cur_level_tabs, std::string class_name="");



/// forexpr ::= 'for' identifier '=' expr ',' expr (',' expr)? 'in' expression
std::unique_ptr<ExprAST> ParseForExpr(Parser_Struct parser_struct, std::string class_name=""); 



/// whileexpr ::= 'while' identifier '=' expr ',' expr (',' expr)? 'in' expression
std::unique_ptr<ExprAST> ParseWhileExpr(Parser_Struct parser_struct, std::string class_name=""); 



std::unique_ptr<ExprAST> ParseAsyncExpr(Parser_Struct parser_struct, std::string class_name=""); 



std::unique_ptr<ExprAST> ParseFinishExpr(Parser_Struct parser_struct, std::string class_name=""); 





std::unique_ptr<ExprAST> ParseNewList(Parser_Struct parser_struct, std::string class_name=""); 


std::unique_ptr<ExprAST> ParseStrVecExpr(Parser_Struct parser_struct); 




  
  


   
std::unique_ptr<ExprAST> ParseTupleExpr(Parser_Struct parser_struct, std::string class_name="");

std::unique_ptr<ExprAST> ParseDataExpr(Parser_Struct parser_struct, std::string class_name=""); 


    
   
  

  
  
  
  
  
  //
  
    
std::unique_ptr<ExprAST> ParseLockExpr(Parser_Struct parser_struct, std::string class_name=""); 
  
  
  
std::unique_ptr<ExprAST> ParseNoGradExpr(Parser_Struct parser_struct, std::string class_name=""); 
  
  
  
std::unique_ptr<ExprAST> ParseMustBeVar(std::string class_name="", std::string expr_name=""); 
  
  
  
    
  
  
  
  
  
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
std::unique_ptr<PrototypeAST> ParsePrototype(Parser_Struct parser_struct, bool from_ctor=false); 
  
  


std::unique_ptr<ExprAST> ParseImport(Parser_Struct);
  
  /// definition ::= 'def' prototype expression
std::unique_ptr<FunctionAST> ParseDefinition(Parser_Struct parser_struct, std::string class_name=""); 
  
  
  /// toplevelexpr ::= expression
std::unique_ptr<FunctionAST> ParseTopLevelExpr(Parser_Struct parser_struct);
  
  
  
  /// external ::= 'extern' prototype
std::unique_ptr<PrototypeAST> ParseExtern(Parser_Struct parser_struct); 
  
  
  
  
std::unique_ptr<ExprAST> ParseClass(Parser_Struct parser_struct); 
 

std::unique_ptr<ExprAST> ParseLLVM_IR_CallExpr(Parser_Struct parser_struct, std::unique_ptr<Nameable> inner, std::string class_name="");

std::unique_ptr<ExprAST> ParseIdxExpr(Parser_Struct parser_struct, std::unique_ptr<Nameable> inner, std::string class_name, int depth=0);
std::unique_ptr<ExprAST> ParseCallExpr(Parser_Struct parser_struct, std::unique_ptr<Nameable> inner, std::string class_name="", int depth=0);
std::unique_ptr<ExprAST> ParseNameableExpr(Parser_Struct parser_struct, std::unique_ptr<Nameable> inner, std::string class_name="", bool can_be_list=true, int depth=0);


