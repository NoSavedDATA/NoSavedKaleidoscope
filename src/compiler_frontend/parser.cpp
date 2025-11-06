#include "llvm/IR/Value.h"


#include <cstdio>
#include <execinfo.h>
#include <iostream>
#include <map>
#include <random>
#include <stdio.h>
#include <thread>
#include <unordered_map>
#include <vector>



#include "../codegen/string.h"
#include "../common/include.h"
#include "../data_types/data_tree.h"
#include "include.h"



void print_caller() {
    void* buffer[10];
    int nptrs = backtrace(buffer, 10);
    char** symbols = backtrace_symbols(buffer, nptrs);

    if (nptrs >= 3)
        std::cout << "Previous function: " << symbols[2] << '\n';
    else
        std::cout << "No previous function found.\n";

    free(symbols);
}



std::map<std::string, std::map<std::string, Data_Tree>> Object_toClass;
std::map<std::string, std::vector<std::string>> Equivalent_Types = {{"float", {"int"}}, {"int", {"float"}}};

std::map<std::string, std::map<std::string, Data_Tree>> Function_Arg_DataTypes;
std::map<std::string, std::map<std::string, std::string>> Function_Arg_Types;
std::map<std::string, std::vector<std::string>> Function_Arg_Names;

std::map<std::string, std::map<std::string, int>> ChannelDirections;

int has_previous_async=0;

std::vector<std::unique_ptr<ExprAST>> asyncs_body;

using namespace llvm;



std::map<std::string, std::string> elements_type_return, ops_type_return;
std::map<int, std::string> op_map;
std::vector<std::string> op_map_names;

std::map<std::string, std::map<std::string, Data_Tree>> data_typeVars;
std::map<std::string, std::map<std::string, std::string>> typeVars;

std::map<std::string, std::map<std::string, int>> ClassVariables;
std::map<std::string, int> ClassSize;
std::unordered_map<std::string, std::vector<int>> ClassPointers;
std::unordered_map<std::string, std::vector<std::string>> ClassPointersType;
std::map<std::string, llvm::Type *> ClassStructs;





std::string Extract_List_Suffix(const std::string& input) {
    static std::mutex mtx;
    std::lock_guard<std::mutex> lock(mtx);

    std::string target = "_list";
    size_t pos = input.find(target);
    if (pos != std::string::npos) {
        // Find the beginning of "list"
        size_t list_start = input.find("list", pos);
        if (list_start != std::string::npos) {
            return input.substr(list_start);
        }
    }


    target = "_dict";
    pos = input.find(target);
    if (pos != std::string::npos) {
        // Find the beginning of "list"
        size_t list_start = input.find("dict", pos);
        if (list_start != std::string::npos) {
            return input.substr(list_start);
        }
    }

    return input; // Return original string if "_list" not found
}


std::string Extract_List_Prefix(const std::string& input) {
    static std::mutex mtx;
    std::lock_guard<std::mutex> lock(mtx);

    std::string target = "_list";
    size_t pos = input.find(target);
    if (pos != std::string::npos)
        return input.substr(0, pos);

    target = "_dict";
    pos = input.find(target);
    if (pos != std::string::npos)
        return input.substr(0, pos);

    target = "_vec";
    pos = input.find(target);
    if (pos != std::string::npos)
        return input.substr(0, pos);


    return input; // Return original string if "_list" not found
}


Data_Tree Parse_Data_Type(std::string root_type, Parser_Struct parser_struct) {

  Data_Tree data_type = Data_Tree(root_type);



  getNextToken(); // eat <  
  while(CurTok!='>')
  {
    std::string dt = IdentifierStr;

    if(CurTok!=tok_data&&CurTok!=tok_struct&&!in_str(dt, Classes)) {
      LogErrorBreakLine(parser_struct.line, root_type + " requires a data type.");
      return data_type;
    }

    getNextToken();

    
    if(CurTok=='<') 
      data_type.Nested_Data.push_back(Parse_Data_Type(dt, parser_struct));
    else {

      if(ends_with(dt,"_vec")) {
        Data_Tree vec_tree = Data_Tree("vec");
        vec_tree.Nested_Data.push_back(remove_suffix(dt,"_vec"));
        data_type.Nested_Data.push_back(vec_tree);
      } else    
        data_type.Nested_Data.push_back(Data_Tree(dt));
    }

    if(CurTok==',')
      getNextToken();
  }
  getNextToken(); // eat >
  
  


  return data_type;
}

Data_Tree ParseDataTree(std::string data_type, bool is_struct, Parser_Struct parser_struct) {
  
  getNextToken(); // eat data token. 
  Data_Tree data_tree;

  if(!is_struct&&CurTok=='<')
    LogError(parser_struct.line, "Found \"<\", but expected a compound data type, got data: " + data_type);
  
  if (is_struct&&CurTok=='<')
    data_tree = Parse_Data_Type(data_type, parser_struct);
  else
  {
    if(ends_with(data_type,"_vec")) {
      data_tree = Data_Tree("vec");
      data_tree.Nested_Data.push_back(remove_suffix(data_type,"_vec"));
    } else    
      data_tree = Data_Tree(data_type);
  }
  return data_tree;
}


/// numberexpr ::= number
std::unique_ptr<ExprAST> ParseNumberExpr(Parser_Struct parser_struct) {
  auto Result = std::make_unique<NumberExprAST>(NumVal);
  getNextToken(); // consume the number
  return std::move(Result);
}

std::unique_ptr<ExprAST> ParseIntExpr(Parser_Struct parser_struct) {
  auto Result = std::make_unique<IntExprAST>((int)NumVal);
  getNextToken(); // consume the number
  return std::move(Result);
}


std::unique_ptr<ExprAST> ParseBoolExpr(Parser_Struct parser_struct) {
  auto Result = std::make_unique<BoolExprAST>(BoolVal);
  getNextToken(); // consume the bool
  return std::move(Result);
}


std::unique_ptr<ExprAST> ParseStringExpr(Parser_Struct parser_struct) {
  auto Result = std::make_unique<StringExprAST>(IdentifierStr);
  getNextToken(); // consume the "
  return std::move(Result);
}


inline void handle_tok_space() {
  while(CurTok==tok_space)
    getNextToken();
}


/// parenexpr ::= '(' expression ')'
std::unique_ptr<ExprAST> ParseParenExpr(Parser_Struct parser_struct, std::string class_name) {

  
  getNextToken(); // eat (.

  auto V = ParseExpression(parser_struct, class_name);
  if (!V)
    return nullptr;

  if (CurTok != ')')
    return LogError(parser_struct.line, "Expected ')' on parenthesis expression.");
  

  getNextToken(); // eat ).
  return V;
}



std::unique_ptr<ExprAST> ParseObjectInstantiationExpr(Parser_Struct parser_struct, std::string class_name) {


  std::string _class = IdentifierStr;

  getNextToken(); // eat class name

  
  //std::cout << "Object name: " << IdentifierStr << " and Class: " << Classes[i]<< "\n";
  bool is_self=false;
  bool is_attr=false;
  std::string pre_dot;
  std::unique_ptr<ExprAST> VecInitSize = nullptr;
      


  if (CurTok==tok_self)
  {
    getNextToken();
    is_self=true;
  }


  std::string Name;
  std::vector<std::pair<std::string, std::unique_ptr<ExprAST>>> VarNames;
  std::vector<std::vector<std::unique_ptr<ExprAST>>> Args;
  std::vector<bool> has_init;
  while (true) {

    Name = IdentifierStr;
    typeVars[parser_struct.function_name][Name] = _class;
    data_typeVars[parser_struct.function_name][Name] = Data_Tree(_class);



    if (!is_self)
    {
      Object_toClass[parser_struct.function_name][IdentifierStr] = Data_Tree(_class);
    } else {

      // std::cout << "++++++++Object instatiation of " << IdentifierStr << "/" << _class << ".\n";
    }

    getNextToken(); // eat identifier.

        
    std::unique_ptr<ExprAST> Init = nullptr;  
    


    
    if(CurTok=='=')
    {
      getNextToken(); // eat =
      Init = ParseExpression(parser_struct, class_name, false);
    }
    
    if(CurTok=='(') {
      has_init.push_back(true);
      Args.push_back(Parse_Arguments(parser_struct, class_name));
    }
    else {
      has_init.push_back(false);
      Args.push_back({});
    }

    VarNames.push_back(std::make_pair(Name, std::move(Init)));

    // End of var list, exit loop.
    if (CurTok != ',')
      break;
    getNextToken(); // eat the ','.

    if (CurTok != tok_identifier)
      return LogError(parser_struct.line, "Expected object identifier names.");
  }



  auto aux = std::make_unique<ObjectExprAST>(parser_struct, std::move(VarNames), std::move(has_init), std::move(Args), "object", std::move(VecInitSize), ClassSize[_class], _class);
  aux->SetSelf(is_self);
  aux->SetIsAttribute(is_attr);


  return aux;
}


std::unique_ptr<IndexExprAST> ParseIdx(Parser_Struct parser_struct, std::string class_name) {
  
  std::vector<std::unique_ptr<ExprAST>> idx, second_idx;

  bool has_sliced_vec=false;

  if (CurTok==':') // [:-1]
  {
    getNextToken();
    has_sliced_vec=true;

    idx.push_back(std::make_unique<IntExprAST>(0));
    if (CurTok==','||CurTok==']')
      second_idx.push_back(std::make_unique<IntExprAST>(COPY_TO_END_INST));
    else
      second_idx.push_back(ParseExpression(parser_struct, class_name, false));
  } else { 

    idx.push_back(ParseExpression(parser_struct, class_name, false));

    if (CurTok==':')
    {
      getNextToken();
      has_sliced_vec=true;

      if (CurTok==','||CurTok==']')
        second_idx.push_back(std::make_unique<IntExprAST>(COPY_TO_END_INST));
      else
        second_idx.push_back(ParseExpression(parser_struct, class_name, false));
    }
  }



  while(CurTok==',')
  {
    getNextToken(); // eat ,
    idx.push_back(ParseExpression(parser_struct, class_name, false));

    if (CurTok==':')
    {
      getNextToken();
      second_idx.push_back(ParseExpression(parser_struct, class_name, false));
      has_sliced_vec=true;
    }
  }
  idx.push_back(std::make_unique<IntExprAST>(TERMINATE_VARARG));
  second_idx.push_back(std::make_unique<IntExprAST>(TERMINATE_VARARG));


  


  return std::make_unique<IndexExprAST>(std::move(idx), std::move(second_idx), has_sliced_vec);
}



std::unique_ptr<ExprAST> ParseIdentifierListExpr(Parser_Struct parser_struct, std::string class_name, std::vector<std::tuple<std::string, int, std::vector<std::unique_ptr<ExprAST>>>> Names) {

  std::string type;
  std::vector<std::string> types;
  std::string IdName = IdentifierStr;

  std::vector<std::unique_ptr<ExprAST>> name_solvers;

  

  
  if (typeVars[parser_struct.function_name].find(IdName) != typeVars[parser_struct.function_name].end())
    type = typeVars[parser_struct.function_name][IdName];

  types.push_back(type);

  auto name_solver_expr = std::make_unique<NameSolverAST>(std::move(Names));
  name_solvers.push_back(std::move(name_solver_expr));
  Names.clear();

  

  while(CurTok==',')
  {
    getNextToken();


    

    IdName = IdentifierStr;
    std::cout << "NEW IDNAME " << IdName << ".\n";

    getNextToken();
  }

}




void PrintNameable(const std::unique_ptr<Nameable> &name) {

  std::vector<Nameable*> names;


  const Nameable* curr = name.get();
  while(name->Inner!=nullptr)
  {
    names.push_back(const_cast<Nameable*>(curr));
    curr = curr->Inner.get();
  }
  for(auto &n : names)
    std::cout << n->Name << ".";
  std::cout << "\n";
}



std::unique_ptr<ExprAST> ParseLLVM_IR_CallExpr(Parser_Struct parser_struct, std::unique_ptr<Nameable> inner, std::string class_name) {
  std::vector<std::unique_ptr<ExprAST>> Args = Parse_Arguments(parser_struct, class_name);
  std::unique_ptr<NameableLLVMIRCall> call_expr = std::make_unique<NameableLLVMIRCall>(parser_struct, std::move(inner), std::move(Args));

  // if (CurTok=='.')
  // {
  //   std::cout << "call expr parsing to nameable" << ".\n";
  //   getNextToken();
  //   return ParseNameableExpr(parser_struct, std::move(call_expr), class_name, false, depth);
  // }

  std::unique_ptr<ExprAST> expr_ptr(static_cast<ExprAST*>(call_expr.release()));
  return std::move(expr_ptr);
}


std::unique_ptr<ExprAST> ParseIdxExpr(Parser_Struct parser_struct, std::unique_ptr<Nameable> inner, std::string class_name, int depth) {
  
  getNextToken(); // eat [
  std::unique_ptr<IndexExprAST> index_expr = ParseIdx(parser_struct, class_name);
  if(CurTok!=']')
    LogError(parser_struct.line, "Expected \"]\"");
  getNextToken(); // eat ]
  

  std::unique_ptr<NameableIdx> vec_expr = std::make_unique<NameableIdx>(parser_struct, std::move(inner), std::move(index_expr));



  if (CurTok=='(')
  {
    LogBlue("Nested vector found call");
    return ParseCallExpr(parser_struct, std::move(vec_expr), class_name, depth);
  }
  if (CurTok=='[')
    return ParseIdxExpr(parser_struct, std::move(vec_expr), class_name, depth);
  if (CurTok=='.')
  {
    getNextToken();
    return ParseNameableExpr(parser_struct, std::move(vec_expr), class_name, false, depth);
  }
  

  std::unique_ptr<ExprAST> expr_ptr(static_cast<ExprAST*>(vec_expr.release()));
  return std::move(expr_ptr);
}



std::unique_ptr<ExprAST> ParseCallExpr(Parser_Struct parser_struct, std::unique_ptr<Nameable> inner, std::string class_name, int depth) {

  std::vector<std::unique_ptr<ExprAST>> Args = Parse_Arguments(parser_struct, class_name);

  std::unique_ptr<NameableCall> call_expr = std::make_unique<NameableCall>(parser_struct, std::move(inner), std::move(Args));

  if (CurTok=='.')
  {
    getNextToken();
    return ParseNameableExpr(parser_struct, std::move(call_expr), class_name, false, depth);
  }

  std::unique_ptr<ExprAST> expr_ptr(static_cast<ExprAST*>(call_expr.release()));
  return std::move(expr_ptr);
}




std::unique_ptr<ExprAST> ParseNameableExpr(Parser_Struct parser_struct, std::unique_ptr<Nameable> inner, std::string class_name, bool can_be_list, int depth)
{
  depth++;

  std::string IdName = IdentifierStr;
  getNextToken(); // eat identifier.

  std::unique_ptr<Nameable> nameable = std::make_unique<Nameable>(parser_struct, IdName, depth);
  nameable->AddNested(std::move(inner));
  



  if (CurTok==',' && can_be_list && depth==1)
  {
    std::vector<std::unique_ptr<Nameable>> IdentifierList;

    IdentifierList.push_back(std::move(nameable));
    while(CurTok==',')
    {
      getNextToken(); // get comma
      getNextToken(); // get identifier

      nameable = std::make_unique<Nameable>(parser_struct, IdentifierStr, depth);
      IdentifierList.push_back(std::move(nameable));
    } 
    
    return std::move(std::make_unique<VariableListExprAST>(std::move(IdentifierList)));
  } 


  
  if(in_str(IdName,LLVM_IR_Functions) && CurTok=='(' && depth==1)
    return ParseLLVM_IR_CallExpr(parser_struct, std::move(nameable), class_name);

  if (CurTok=='.')
  {
    getNextToken();
    return ParseNameableExpr(parser_struct, std::move(nameable), class_name, can_be_list, depth);
  }
  if (CurTok=='(')
    return ParseCallExpr(parser_struct, std::move(nameable), class_name, depth);

  if (CurTok=='[')
    return ParseIdxExpr(parser_struct, std::move(nameable), class_name, depth);
  
  
  std::unique_ptr<ExprAST> expr_ptr(static_cast<ExprAST*>(nameable.release()));
  return std::move(expr_ptr);
}









/// ifexpr ::= 'if' expression 'then' expression 'else' expression
std::unique_ptr<ExprAST> ParseIfExpr(Parser_Struct parser_struct, std::string class_name)
{
  
  int cur_level_tabs = SeenTabs;

  getNextToken(); // eat the if.

  
  //std::cout << "If tabs level: " << cur_level_tabs <<  "\n";
  

  // condition.
  auto Cond = ParseExpression(parser_struct, class_name);

  if (!Cond) {
    std::cout << "cond is null" << ".\n";
    return nullptr;
  }


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
    
    auto body = ParseExpression(parser_struct, class_name);
    if (!body)
      return nullptr;
    Then.push_back(std::move(body));
    
  }
  
  if (Then.size()==0)
  {
    LogError(parser_struct.line, "Then is null");
    return nullptr;
  }
  
  
  if(CurTok == tok_space)
    getNextToken();

  //std::cout << "\n\nIf else token: " << ReverseToken(CurTok) <<  "\n\n\n";

  if (CurTok != tok_else) {
    Else.push_back(std::make_unique<IntExprAST>(0));

    return std::make_unique<IfExprAST>(std::move(Cond), std::move(Then),
                                      std::move(Else));
  }
  else {
    getNextToken(); //eat else
    if(CurTok != tok_space)
      LogError(parser_struct.line, "else requer barra de espaço.");
    getNextToken();

    while(true)
    {

      if (SeenTabs <= cur_level_tabs && CurTok != tok_space)
        break;

      while (CurTok == tok_space)
        getNextToken();

      if (SeenTabs <= cur_level_tabs)
        break;
      
      auto body = ParseExpression(parser_struct, class_name);
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




std::vector<std::unique_ptr<ExprAST>> ParseIndentedBodies(Parser_Struct parser_struct, int cur_level_tabs, std::string class_name)
{
  std::vector<std::unique_ptr<ExprAST>> Body, NullBody;
  //std::cout << "\nSeen tabs on for body POST: " << SeenTabs << "\n\n";

  handle_tok_space();
  while(!in_char(CurTok, terminal_tokens))
  {
    //std::cout << "\n\nParsing new expression with tabs: " << SeenTabs << " tok: " << ReverseToken(CurTok) << "\n";
    if (SeenTabs <= cur_level_tabs && CurTok != tok_space)
    {
      //std::cout << "Breaking for with cur tok: " << ReverseToken(CurTok) << " Seen Tabs:" << SeenTabs <<  "\n";
      break;
    } 
    //std::cout << "\nSeen tabs on for body: " << SeenTabs << "\nCur tok: " << ReverseToken(CurTok) << "\n\n";


    //std::cout << "Post space has " << SeenTabs << " tabs.\n";
    if (SeenTabs <= cur_level_tabs)
      break;


    auto body = ParseExpression(parser_struct, class_name);
    if (!body)
      return std::move(NullBody);
    Body.push_back(std::move(body));
    
    handle_tok_space();
  }

  handle_tok_space();
    
  return std::move(Body);
}





std::unique_ptr<ExprAST> ParseStandardForExpr(Parser_Struct parser_struct, std::string class_name, int cur_level_tabs, std::string IdName) {
  getNextToken(); // eat '='.

  auto Start = ParseExpression(parser_struct, class_name, false);
  if (!Start)
    return nullptr;
  if (CurTok != ',')
    return LogError(parser_struct.line, "Expected ',' after for's control variable initial value.");
  getNextToken();

  
  
  typeVars[parser_struct.function_name][IdName] = Start->GetType();
  data_typeVars[parser_struct.function_name][IdName] = Start->GetDataTree();
  


  auto End = ParseExpression(parser_struct, class_name);
  if (!End)
    return nullptr;



  std::unique_ptr<ExprAST> Step = std::make_unique<IntExprAST>(1.0);
  if (CurTok == ',') { // The step value is optional.
    getNextToken();
    auto aux = ParseExpression(parser_struct, class_name);
    if (aux)
      Step = std::move(aux);
  }
  
  std::vector<std::unique_ptr<ExprAST>> Body;

  Body = ParseIndentedBodies(parser_struct, cur_level_tabs, class_name);

  return std::make_unique<ForExprAST>(IdName, std::move(Start), std::move(End),
                                       std::move(Step), std::move(Body), parser_struct);
}


std::unique_ptr<ExprAST> ParseForEachExpr(Parser_Struct parser_struct, std::string class_name, int cur_level_tabs, std::string IdName) {

  
  getNextToken(); // eat "in".


  auto Vec = ParseExpression(parser_struct, class_name);
  if (!Vec)
    return nullptr;
  

  Data_Tree data_type = Vec->GetDataTree();
  if(data_type.Nested_Data.size()==0)
  {
    LogError(parser_struct.line, "Using a non-compound data type at a \"for in\" expression.");
    data_type.Print();
  }

  data_typeVars[parser_struct.function_name][IdName] = data_type.Nested_Data[0];

  std::vector<std::unique_ptr<ExprAST>> Body;
  Body = ParseIndentedBodies(parser_struct, cur_level_tabs, class_name);


  return std::make_unique<ForEachExprAST>(IdName, std::move(Vec), std::move(Body), parser_struct, data_type);
}


/// forexpr ::= 'for' identifier '=' expr ',' expr (',' expr)? 'in' expression
std::unique_ptr<ExprAST> ParseForExpr(Parser_Struct parser_struct, std::string class_name) {

  int cur_level_tabs = SeenTabs;

  //std::cout << "\nSeen tabs on for: " << SeenTabs << "\n\n";

  getNextToken(); // eat the for.

  if(CurTok!=tok_identifier)
    LogError(parser_struct.line, "Unexpected token \"" + ReverseToken(CurTok) + "\" on \"for in\" expression. Expected control variable name.");
 

  std::string IdName = IdentifierStr;
  getNextToken(); // eat identifier.  



  if (CurTok==tok_in)
    return ParseForEachExpr(parser_struct, class_name, cur_level_tabs, IdName);
  

  if (CurTok=='=')
  {
    typeVars[parser_struct.function_name][IdName] = "int";
    data_typeVars[parser_struct.function_name][IdName] = Data_Tree("int");
    return ParseStandardForExpr(parser_struct, class_name, cur_level_tabs, IdName);
  }
  else
    return LogError(parser_struct.line, "Expected for's control variable initial value.");
}



/// whileexpr ::= 'while' identifier '=' expr ',' expr (',' expr)? 'in' expression
std::unique_ptr<ExprAST> ParseWhileExpr(Parser_Struct parser_struct, std::string class_name) {

  int cur_level_tabs = SeenTabs;

  getNextToken(); // eat the while.


  //if (CurTok != tok_identifier)
  //  return LogError(parser_struct.line, "Identificador da variável de controle esperado depois do while.");


  auto Cond = ParseExpression(parser_struct, class_name);
  if (!Cond)
    return nullptr;
  
  std::vector<std::unique_ptr<ExprAST>> Body = ParseIndentedBodies(parser_struct, cur_level_tabs, class_name);

  return std::make_unique<WhileExprAST>(std::move(Cond), std::move(Body), parser_struct);
}



std::unique_ptr<ExprAST> ParseAsyncExpr(Parser_Struct parser_struct, std::string class_name) {

  int cur_level_tabs = SeenTabs;

  getNextToken(); // eat the async.

  
  std::vector<std::unique_ptr<ExprAST>> Bodies;
  
  std::string async_scope = parser_struct.function_name + "_async";
  for (auto pair : typeVars[parser_struct.function_name])
    typeVars[async_scope][pair.first] = pair.second;
  for (auto pair : data_typeVars[parser_struct.function_name])
    data_typeVars[async_scope][pair.first] = pair.second;



  Parser_Struct body_parser_struct = parser_struct;
  body_parser_struct.function_name = async_scope;

  Bodies.push_back(std::make_unique<IncThreadIdExprAST>());
  if (CurTok != tok_space)
    Bodies.push_back(std::move(ParseExpression(body_parser_struct, class_name)));
  else
    Bodies = ParseIndentedBodies(body_parser_struct, cur_level_tabs, class_name);
  
  
  
  if(parser_struct.function_name!="__anon_expr")
    has_previous_async++;

  return std::make_unique<AsyncExprAST>(std::move(Bodies), parser_struct);
}


std::unique_ptr<ExprAST> ParseSpawnExpr(Parser_Struct parser_struct, std::string class_name) {
  int cur_level_tabs = SeenTabs;

  getNextToken(); // eat spawn
  

  std::string async_scope = parser_struct.function_name + "_spawn";
  for (auto pair : typeVars[parser_struct.function_name])
    typeVars[async_scope][pair.first] = pair.second;
  for (auto pair : data_typeVars[parser_struct.function_name])
    data_typeVars[async_scope][pair.first] = pair.second;


  Parser_Struct body_parser_struct = parser_struct;
  body_parser_struct.function_name = async_scope;

  std::vector<std::unique_ptr<ExprAST>> Bodies;
  Bodies.push_back(std::make_unique<IncThreadIdExprAST>());
  if (CurTok != tok_space)
    Bodies.push_back(std::move(ParseExpression(body_parser_struct, class_name)));
  else
    Bodies = ParseIndentedBodies(body_parser_struct, cur_level_tabs, class_name);
  
  
  if(parser_struct.function_name!="__anon_expr")
    has_previous_async++;
   
  return std::make_unique<SpawnExprAST>(std::move(Bodies), parser_struct);
}

std::unique_ptr<ExprAST> ParseAsyncsExpr(Parser_Struct parser_struct, std::string class_name) {
  int cur_level_tabs = SeenTabs;

  getNextToken(); // eat the async.


  if (CurTok!=tok_int)
    LogError(parser_struct.line, "asyncs expression expects the number of asynchrnonous functions.");

  int async_count = NumVal;
  getNextToken();
  
  std::vector<std::unique_ptr<ExprAST>> Bodies;
  
  std::string async_scope = parser_struct.function_name + "_asyncs";
  for (auto pair : typeVars[parser_struct.function_name])
    typeVars[async_scope][pair.first] = pair.second;
  for (auto pair : data_typeVars[parser_struct.function_name])
    data_typeVars[async_scope][pair.first] = pair.second;


  Parser_Struct body_parser_struct = parser_struct;
  body_parser_struct.function_name = async_scope;
  Bodies.push_back(std::make_unique<IncThreadIdExprAST>());
  if (CurTok != tok_space)
    Bodies.push_back(std::move(ParseExpression(body_parser_struct, class_name)));
  else
    Bodies = ParseIndentedBodies(body_parser_struct, cur_level_tabs, class_name);
  
  
  
  if(parser_struct.function_name!="__anon_expr")
    has_previous_async++;

  return std::make_unique<AsyncsExprAST>(std::move(Bodies), async_count, parser_struct);
}



std::unique_ptr<ExprAST> ParseFinishExpr(Parser_Struct parser_struct, std::string class_name) {

  int cur_level_tabs = SeenTabs;
  
  getNextToken(); // eat finish.

  std::vector<std::unique_ptr<ExprAST>> Bodies;
  std::vector<bool> IsAsync;
  

  if (CurTok!=tok_space)
    LogError(parser_struct.line, "Finish requires a line break.");

  getNextToken(); 

  handle_tok_space();

  while(!in_char(CurTok, terminal_tokens))
  {
    if (SeenTabs <= cur_level_tabs && CurTok != tok_space)
      break;
    Bodies.push_back(std::move(ParseExpression(parser_struct, class_name)));
    IsAsync.push_back(false);
    handle_tok_space();
  }


  return std::make_unique<FinishExprAST>(std::move(Bodies),
                                         std::move(IsAsync));
}





std::unique_ptr<ExprAST> ParseMainExpr(Parser_Struct parser_struct, std::string class_name) {



  

  
  int cur_level_tabs = SeenTabs;
  
  getNextToken(); // eat main

  std::vector<std::unique_ptr<ExprAST>> Bodies;
  std::vector<bool> IsAsync;

  for (int i=0; i<has_previous_async; ++i)
    Bodies.push_back(std::make_unique<AsyncFnPriorExprAST>());
  
  

  if (CurTok!=tok_space)
    LogError(parser_struct.line, "\"main\" requires a line break.");

  getNextToken(); 

  handle_tok_space();

  while(!in_char(CurTok, terminal_tokens))
  {
    if (SeenTabs <= cur_level_tabs && CurTok != tok_space)
      break;
    Bodies.push_back(std::move(ParseExpression(parser_struct, class_name)));
    handle_tok_space();
  }


  return std::make_unique<MainExprAST>(std::move(Bodies));
}





std::unique_ptr<ExprAST> ParseNewList(Parser_Struct parser_struct, std::string class_name) {

  
  
  getNextToken(); // [
  std::vector<std::unique_ptr<ExprAST>> Elements;
  if (CurTok != ']') {
    while (true) {
      std::string element_type;


      if (auto element = ParseExpression(parser_struct, class_name, false))
      {
        element_type = element->GetDataTree().Type;
 
        Elements.push_back(std::make_unique<StringExprAST>(element_type));
        Elements.push_back(std::move(element));
      } 
      else
        return nullptr;

      if (CurTok == ']')
        break;
      if (CurTok != ',')
      {
        LogErrorBreakLine(parser_struct.line, "Expected \",\" or \"]\" at the new tuple expression.");
        return nullptr;
      }
      getNextToken();
    }
  }   
  getNextToken(); // ]

    
  Elements.push_back(std::make_unique<StringExprAST>("TERMINATE_VARARG"));

  
  return std::make_unique<NewVecExprAST>(std::move(Elements), "list");
}



std::unique_ptr<ExprAST> ParseNewDict(Parser_Struct parser_struct, std::string class_name) {




  
  getNextToken(); // {
  std::vector<std::unique_ptr<ExprAST>> Keys, Elements;
  if (CurTok != '}') {
    while (true) {
      // std::cout << "CURRENT TOKEN: " << ReverseToken(CurTok) << ".\n";
      std::string element_type;

      if (auto key = ParseExpression(parser_struct, class_name, false))
        Keys.push_back(std::move(key));
      else {
        LogError(parser_struct.line, "Expected an expression at the dict key.");
        return nullptr;
      }


      if (CurTok!=':')
      {
        LogError(parser_struct.line, "Expected \":\" at the dict expression.");
        return nullptr;
      }
      getNextToken(); //:


      if (auto element = ParseExpression(parser_struct, class_name, false))
      {
        element_type = element->GetType();

        
        Elements.push_back(std::move(element));
      } 
      else
        return nullptr;

      if (CurTok == '}')
        break;
      if (CurTok != ',')
      {
        LogError(parser_struct.line, "Expected \",\" or \"}\" at the dict expression.");
      }
      getNextToken();
    }
  }   
  getNextToken(); // }

  
  Elements.push_back(std::make_unique<StringExprAST>("TERMINATE_VARARG"));

  return std::make_unique<NewDictExprAST>(std::move(Keys), std::move(Elements), "dict", parser_struct);
}


std::string CheckListAppend(std::string callee, std::vector<std::unique_ptr<ExprAST>> &arguments, std::string fn_name, std::string Prev_IdName, Parser_Struct parser_struct)
{
  if(callee!="list_append")
    return callee;

  Data_Tree list_data_type = data_typeVars[fn_name][Prev_IdName];

  std::string list_type = list_data_type.Nested_Data[0].Type;

  std::string argument_type = arguments[0]->GetDataTree().Type;

  if (list_type!=argument_type)
    LogError(parser_struct.line, "Tried to append " + argument_type + " to " + list_type + " list.");


  if(in_str(list_type,{"int","float","bool"}))
    return callee+"_"+list_type;
  
  arguments.push_back(std::make_unique<StringExprAST>(list_type));
  return callee;
}


inline std::vector<std::unique_ptr<ExprAST>> Parse_Argument_List(Parser_Struct parser_struct, std::string class_name, std::string expression_name)
{
  std::vector<std::unique_ptr<ExprAST>> Args;
  if(CurTok!='(')
  {
    LogError(parser_struct.line, "Expected ( afther the method name of the " + expression_name + " Expression.");
    return std::move(Args);
  }

  
  getNextToken(); // eat (
  if (CurTok != ')') {
    while (true) {
      if (auto Arg = ParseExpression(parser_struct, class_name))
      {
        //std::cout << "Parsed arg " << Arg->GetName() << "\n";
        Args.push_back(std::move(Arg));
      } 
      else
        return std::move(Args);

      if (CurTok == ')')
        break;
      if (CurTok != ',')
      {
        LogError(parser_struct.line, "Expected ')' or ',' on the Function Call arguments list.");
        return std::move(Args);
      }
      getNextToken();
    }
  } 
  
  // Eat the ')'.
  getNextToken();

  return std::move(Args);
}



std::vector<std::unique_ptr<ExprAST>> Parse_Arguments(Parser_Struct parser_struct, std::string class_name)
{
  getNextToken(); // eat (
  std::vector<std::unique_ptr<ExprAST>> Args;
  if (CurTok != ')') {
    while (true) {
      if (CurTok=='>')
      {
        getNextToken(); // eat >
        auto inner_vec = ParseExpression(parser_struct, class_name, false);

        auto Arg = std::make_unique<SplitParallelExprAST>(std::move(inner_vec));
        Args.push_back(std::move(Arg));
      }
      else if (CurTok==':')
      {
        getNextToken(); // eat :
        auto inner_vec = ParseExpression(parser_struct, class_name, false);
        
        auto Arg = std::make_unique<SplitStridedParallelExprAST>(std::move(inner_vec));
        Args.push_back(std::move(Arg));
      }
      else if (auto Arg = ParseExpression(parser_struct, class_name, false))
      {
        Args.push_back(std::move(Arg));
      }        
      else
        return {};

      if (CurTok == ')')
        break;

      if (CurTok != ',')
      {

        LogError(parser_struct.line, "Expected ')' or ',' on the call arguments list.");
        return {};
      }
      getNextToken();
    }
  }
  // Eat the ')'.
  getNextToken();
  return std::move(Args);
}









std::unique_ptr<ExprAST> ParseVarExpr(Parser_Struct parser_struct, std::string suffix, std::string class_name) {

  // std::cout << "Parsing data with data type: " << IdentifierStr << ".\n";

  std::string data_type = suffix + "var";

  getNextToken(); // eat var token.



    

  
  std::vector<std::pair<std::string, std::unique_ptr<ExprAST>>> VarNames;
  std::vector<std::unique_ptr<ExprAST>> notes;



  // Get the Notes vector
  if (CurTok == '[')
  {
    return LogErrorBreakLine(parser_struct.line, "Cannot use notes inside a var data type.");
  }




  std::string pre_dot="";
  bool is_self = false;
  if (CurTok == tok_self)
  {
    is_self=true; //TODO: set self per VarName instead.
    getNextToken(); // get self
    getNextToken(); // get dot
  }

  if (CurTok != tok_identifier)
    return LogError(parser_struct.line, "Expected " + data_type + " identifier name.");



  while (true) {
    std::string Name = IdentifierStr;


    getNextToken(); // eat identifier.

    


    std::unique_ptr<ExprAST> Init;
    if (CurTok == '=' || CurTok == tok_arrow)
    {
      bool is_message = CurTok==tok_arrow;
      getNextToken(); // eat the '='.
      Init = ParseExpression(parser_struct, class_name);
      Init->SetIsMsg(is_message);
      if (!Init)
        return nullptr;
    } else
      Init = std::make_unique<NullPtrExprAST>();

    VarNames.push_back(std::make_pair(Name, std::move(Init)));

    // End of var list, exit loop.
    if (CurTok != ',')
      break;
    getNextToken(); // eat the ','.

    if (CurTok != tok_identifier)
      return LogError(parser_struct.line, "Expected " + data_type + " identifier name(s).");
  }


  auto aux = std::make_unique<UnkVarExprAST>(parser_struct, std::move(VarNames), data_type,
                                             std::move(notes));
  aux->SetSelf(is_self);
  aux->SetPreDot(pre_dot);
  
  

  return aux;
}





std::unique_ptr<ExprAST> ParseTupleExpr(Parser_Struct parser_struct, std::string class_name) {
  

  // std::string data_type = IdentifierStr;
  // getNextToken(); // eat tuple token.


   
  // std::vector<std::pair<std::string, std::unique_ptr<ExprAST>>> VarNames;
  // std::vector<std::string> types;



  // if(CurTok!='<'&&data_type=="tuple")
  //   return LogErrorBreakLine(parser_struct.line, "Tuple definition expects its data types.");

  // Data_Tree data_tree;
  // if (CurTok=='<')
  //  data_tree = Parse_Data_Type(data_type, parser_struct);
  // else
  //   data_tree = Data_Tree(data_type);


  // std::string pre_dot="";
  // bool is_self = false;
  // bool is_attr = false;
  // if (CurTok == tok_self)
  // {
  //   is_self=true; //TODO: set self per VarName instead.
  //   getNextToken();
  // }
  // if (CurTok == tok_class_attr)
  // {
  //   is_attr=true;
  //   pre_dot = IdentifierStr;
  //   getNextToken();
  // }

  // if (CurTok != tok_identifier)
  //   return LogError(parser_struct.line, "Expected " + data_type + " identifier name. Got token: " + ReverseToken(CurTok));



  // while (true) {
  //   std::string Name = IdentifierStr;


  //   std::string prefix_datatype = Extract_List_Prefix(data_type);
  //   if ((ends_with(data_type, "_list")||ends_with(data_type,"_dict")) && !in_str(prefix_datatype, data_tokens) )
  //     Object_toClass[parser_struct.function_name][IdentifierStr] = prefix_datatype;
  //   typeVars[parser_struct.function_name][IdentifierStr] = data_type;
  //   getNextToken(); // eat identifier.

  

  //   std::unique_ptr<ExprAST> Init;
  //   if (CurTok == '=')
  //   {
  //     getNextToken(); // eat the '='.
  //     Init = ParseExpression(parser_struct, class_name);
  //     if (!Init)
  //       return nullptr;
  //   } else
  //     Init = std::make_unique<NullPtrExprAST>();

  //   VarNames.push_back(std::make_pair(Name, std::move(Init)));

  //   // End of var list, exit loop.
  //   if (CurTok != ',')
  //     break;
  //   getNextToken(); // eat the ','.

  //   if (CurTok != tok_identifier)
  //     return LogError(parser_struct.line, "Expected " + data_type + " identifier name(s). Got token: " + ReverseToken(CurTok)); 
  // }

  // // if(data_type=="tuple")
  //   auto aux = std::make_unique<TupleExprAST>(parser_struct, std::move(VarNames), data_type, data_tree);
  // // if(data_type=="list")
  // //   auto aux = std::make_unique<ListExprAST>(parser_struct, std::move(VarNames), data_type, data_tree);
  // // if(data_type=="dict")
  // //   auto aux = std::make_unique<TupleExprAST>(parser_struct, std::move(VarNames), data_type, data_tree);
  // aux->SetSelf(is_self);
  // aux->SetIsAttribute(is_attr);
  // aux->SetPreDot(pre_dot);
 
  // return aux;
  return nullptr;
}




std::unique_ptr<ExprAST> ParseChannelExpr(Parser_Struct parser_struct, std::string class_name, Data_Tree inner_data_tree) {

  LogBlue("Parse channel");

  Data_Tree data_tree = Data_Tree("channel");
  data_tree.Nested_Data.push_back(inner_data_tree);

  getNextToken(); // eat channel

  int buffer_size = 1;
  if (CurTok==tok_int) {
    buffer_size = NumVal;
    getNextToken();
  }
  

  if(CurTok!=tok_identifier)
    LogError(parser_struct.line, "Unexpected token " + ReverseToken(CurTok) + ". Expected channel name.");

  std::string IdName = IdentifierStr;

  getNextToken(); // eat identifier



  data_typeVars[parser_struct.function_name][IdName] = data_tree;
  ChannelDirections[parser_struct.function_name][IdName] = ch_both;

  
  return std::make_unique<ChannelExprAST>(parser_struct, data_tree, IdName, buffer_size);
}


std::unique_ptr<ExprAST> ParseDataExpr(Parser_Struct parser_struct, std::string class_name) {

  bool is_struct=(CurTok==tok_struct);

  std::string data_type = IdentifierStr; 
  Data_Tree data_tree = ParseDataTree(data_type, is_struct, parser_struct);
  

  if (CurTok==tok_channel)
    return ParseChannelExpr(parser_struct, class_name, data_tree);


  std::vector<std::pair<std::string, std::unique_ptr<ExprAST>>> VarNames;
  std::vector<std::unique_ptr<ExprAST>> notes;
  bool has_notes=false;
    
  




  // Get the Notes vector
  if (CurTok == '[')
  {
    has_notes=true;
    getNextToken();
     
    while (true) {
      if (CurTok!=tok_number && CurTok!=tok_int && CurTok!=tok_identifier && CurTok!=tok_self && CurTok!=tok_str)
        return LogError(parser_struct.line, "Expected a number, string or var on the notes of " + data_type + ".");
      
      
      if (CurTok==tok_number)
      { 
        notes.push_back(std::make_unique<NumberExprAST>(NumVal));
        getNextToken();
      } else if (CurTok==tok_str) {
        notes.push_back(std::make_unique<StringExprAST>(IdentifierStr));
        getNextToken();

      } else if (CurTok==tok_number) {
        notes.push_back(std::make_unique<IntExprAST>(NumVal));
        getNextToken();
        
      } else if (CurTok==tok_identifier||CurTok==tok_self)
        notes.push_back(ParseNameableExpr(parser_struct, std::make_unique<NameableRoot>(parser_struct), class_name, false));
      else {
        notes.push_back(std::move(ParsePrimary(parser_struct, class_name, false)));
      }

      
      if (CurTok != ',')
        break;
      getNextToken(); // eat the ','.
    }

    
    if (CurTok != ']')
      return LogError(parser_struct.line, "] not found at " + data_type + " variable declaration.");
    getNextToken();
  }




  std::string pre_dot="";
  bool is_self = false;
  

  if (CurTok==tok_self)
  {
    is_self=true; //TODO: set self per VarName instead.
    getNextToken(); // eat self
    getNextToken(); // eat dot
  }

  if (CurTok != tok_identifier)
    return LogError(parser_struct.line, "Expected " + data_type + " identifier name.");



  while (true) {
    std::string Name = IdentifierStr;


    std::string prefix_datatype = Extract_List_Prefix(data_type);
    if ((ends_with(data_type, "_list")||ends_with(data_type,"_dict")) && !in_str(prefix_datatype, data_tokens) )
      Object_toClass[parser_struct.function_name][IdentifierStr] = Data_Tree(prefix_datatype);
    typeVars[parser_struct.function_name][IdentifierStr] = data_type;
    data_typeVars[parser_struct.function_name][IdentifierStr] = data_tree;
    getNextToken(); // eat identifier.

    


    std::unique_ptr<ExprAST> Init;
    if (CurTok=='='||CurTok==tok_arrow)
    {
      bool is_message = CurTok==tok_arrow;
      getNextToken(); // eat the '='.
      Init = ParseExpression(parser_struct, class_name);
      Init->SetIsMsg(is_message);
      if (!Init)
        return nullptr;
    } else
    {
      if (data_type=="float")
        Init = std::make_unique<NumberExprAST>(0.0f);
      else if (data_type=="int")
        Init = std::make_unique<IntExprAST>(0);
      else if (data_type=="bool")
        Init = std::make_unique<BoolExprAST>(false);
      else if (data_type=="str")
        Init = std::make_unique<StringExprAST>("");
      else
        Init = std::make_unique<NullPtrExprAST>();
    }
    VarNames.push_back(std::make_pair(Name, std::move(Init)));

    // End of var list, exit loop.
    if (CurTok != ',')
      break;
    getNextToken(); // eat the ','.

    if (CurTok != tok_identifier)
      return LogError(parser_struct.line, "Expected " + data_type + " identifier name(s).");
  }



  auto aux = std::make_unique<DataExprAST>(parser_struct, std::move(VarNames), data_type, data_tree, has_notes, is_struct,
                                             std::move(notes));
  aux->SetSelf(is_self);
  
  

  return aux;
}










































std::unique_ptr<ExprAST> ParseLockExpr(Parser_Struct parser_struct, std::string class_name) {
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
    
    lockVars[IdentifierStr] = new SpinLock();
  }
  
  
  std::vector<std::unique_ptr<ExprAST>> Body = ParseIndentedBodies(parser_struct, cur_level_tabs, class_name);


  return std::make_unique<LockExprAST>(std::move(Body), Name);
}



std::unique_ptr<ExprAST> ParseNoGradExpr(Parser_Struct parser_struct, std::string class_name) {
  int cur_level_tabs = SeenTabs;
  getNextToken(); // eat no_grad
  
  std::vector<std::unique_ptr<ExprAST>> Body = ParseIndentedBodies(parser_struct, cur_level_tabs, class_name);

  return std::make_unique<NoGradExprAST>(std::move(Body));
}








std::unique_ptr<ExprAST> ParseRetExpr(Parser_Struct parser_struct, std::string class_name) {
  getNextToken(); //eat return

  std::vector<std::unique_ptr<ExprAST>> Vars;

  std::unique_ptr<ExprAST> expr;
  


  
  while(true) {
    
    if (CurTok==tok_number)
    {
      expr = std::make_unique<NumberExprAST>(NumVal);
      getNextToken();
    } else if (CurTok==tok_int) {
      expr = std::make_unique<IntExprAST>(NumVal);
      getNextToken();
    }
    else
      expr = ParseExpression(parser_struct, class_name, false);
    
    Vars.push_back(std::move(expr));
    if(CurTok!=',')
      break;
    getNextToken(); // eat ,
  }

  return make_unique<RetExprAST>(std::move(Vars), parser_struct);
}






/// primary
///   ::= identifierexpr
///   ::= numberexpr
///   ::= parenexpr
///   ::= ifexpr
///   ::= forexpr
std::unique_ptr<ExprAST> ParsePrimary(Parser_Struct parser_struct, std::string class_name, bool can_be_list) {
  switch (CurTok) {
  default:
    //return std::move(std::make_unique<NumberExprAST>(0.0f));
    return LogErrorT(parser_struct.line, CurTok);
  case tok_eof:
    return std::make_unique<EmptyStrExprAST>();
  case tok_identifier:
  {
    if(in_str(IdentifierStr, Classes))
      return ParseObjectInstantiationExpr(parser_struct, class_name);
    return ParseNameableExpr(parser_struct, std::make_unique<NameableRoot>(parser_struct), class_name, can_be_list);
  }
  case tok_self:
    return ParseNameableExpr(parser_struct, std::make_unique<NameableRoot>(parser_struct), class_name, can_be_list);
  case tok_number:
    return ParseNumberExpr(parser_struct);
  case tok_int:
    return ParseIntExpr(parser_struct);
  case tok_bool:
    return ParseBoolExpr(parser_struct);
  case tok_str:
    return ParseStringExpr(parser_struct);
  case '(':
    return ParseParenExpr(parser_struct);
  case tok_if:
    return ParseIfExpr(parser_struct, class_name);
  case tok_for:
    return ParseForExpr(parser_struct, class_name);
  case tok_while:
    return ParseWhileExpr(parser_struct, class_name);
  case tok_async_finish:
    return ParseFinishExpr(parser_struct, class_name);
  case tok_async:
    return ParseAsyncExpr(parser_struct, class_name);
  case tok_asyncs:
    return ParseAsyncsExpr(parser_struct, class_name);
  case tok_spawn:
    return ParseSpawnExpr(parser_struct, class_name);
  case tok_lock:
    return ParseLockExpr(parser_struct, class_name);
  case tok_main:
    return ParseMainExpr(parser_struct, class_name);
  case tok_no_grad:
    return ParseNoGradExpr(parser_struct, class_name);
  case tok_ret:
    return ParseRetExpr(parser_struct, class_name);
  case tok_data:
    return ParseDataExpr(parser_struct, class_name);
  case tok_struct:
    return ParseDataExpr(parser_struct, class_name);
  case tok_var:
    return ParseVarExpr(parser_struct, "", class_name);
  case '[':
    return ParseNewList(parser_struct, class_name);
  case '{':
    return ParseNewDict(parser_struct, class_name);
  case tok_space:
    getNextToken();
    return ParsePrimary(parser_struct ,class_name, can_be_list);
  }
}

/// unary
///   ::= primary
///   ::= '!' unary
std::unique_ptr<ExprAST> ParseUnary(Parser_Struct parser_struct, std::string class_name, bool can_be_list) {
  // std::cout <<"Parse unary got can_be_list: " << can_be_list <<  "\n";
  // If the current token is not an operator, it must be a primary expr.
  
  
  if ((!isascii(CurTok) || CurTok == '(' || CurTok == ',' || CurTok == '[' || CurTok == '{' || CurTok=='<' || CurTok=='>' || CurTok==':')&&CurTok!=tok_not)
  {
    //std::cout << "Returning, non-ascii found.\n";
    // if(CurTok=='>'||CurTok=='^')
    // {
    //   std::cout << "PARALELIZE VECTOR AT UNARY" << ".\n";
    // }
    return ParsePrimary(parser_struct, class_name, can_be_list);
  }

  
  // If this is a unary operator, read it.
  int Opc = CurTok;
  
  
  getNextToken();
  if (auto Operand = ParseUnary(parser_struct, class_name, can_be_list))
  {    
    std::string operand_type = Operand->GetDataTree().Type;
    auto expr = std::make_unique<UnaryExprAST>(Opc, std::move(Operand), parser_struct);
    expr->SetType(operand_type);

    return expr;
  }
  return nullptr;
}






/// binoprhs
///   ::= ('+' unary)*
std::tuple<std::unique_ptr<ExprAST>, int, std::string> ParseBinOpRHS(Parser_Struct parser_struct, int ExprPrec,
                                              std::unique_ptr<ExprAST> LHS,
                                              std::string class_name) {
  
  
  int LhsTok = 0;
  int RhsTok = 0;

  int L_cuda = type_float;
  int R_cuda = type_float;

  std::string LName, RName;
  

  std::string L_type="";
  std::string R_type="";

  while (true)
  {
    

    // If this is a binop, find its precedence.
    int TokPrec = get_tokenPrecedence();
    // If this is a binop that binds at least as tightly as the current binop,
    // consume it, otherwise we are done.
    

    if (TokPrec < ExprPrec)
      return std::make_tuple(std::move(LHS), L_cuda, L_type);
    
      


    if (CurTok == tok_space)
    {
      //std::cout << "Returning tok space with " << SeenTabs << " tabs. \n\n\n";
      getNextToken();
      return std::make_tuple(std::move(LHS), L_cuda, L_type);
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




    if (CurTok==')')
      return std::make_tuple(std::move(LHS), L_cuda, L_type);

    
    getNextToken(); // eat binop
    if (CurTok==tok_number)
      RName = std::to_string(NumVal);
    else
      RName = IdentifierStr;


    // Get the Right Hand Side token for debugging only
    RhsTok = CurTok;

    
    auto RHS = ParseUnary(parser_struct, class_name, false); // Returns an identifier, number or expression result
    if (!RHS)
      return std::make_tuple(nullptr,0,"None");


    
    R_type = RHS->GetType();
    // std::cout << "--Rtype is: " << R_type << ".\n";
    
    
    

    // If BinOp binds less tightly with RHS than the operator after RHS, let
    // the pending operator take RHS as its LHS.
    int NextPrec = get_tokenPrecedence();
    

    if (TokPrec < NextPrec)
    {
      //std::cout << NextPrec << " Next Prec\n";
        
      auto tuple = ParseBinOpRHS(parser_struct, TokPrec + 1, std::move(RHS));
      RHS = std::move(std::get<0>(tuple));
      R_cuda = std::get<1>(tuple);
      R_type = std::get<2>(tuple);

      // std::cout << "--Updated type is: " << R_type << ".\n";


      if (!RHS)
        return std::make_tuple(nullptr,0,"None");
      
    }


    
    
    
    LHS = std::make_unique<BinaryExprAST>(BinOp, std::move(LHS), std::move(RHS), parser_struct);
    

    LhsTok = RhsTok;
  }
}


/// expression
///   ::= unary binoprhs
///
std::unique_ptr<ExprAST> ParseExpression(Parser_Struct parser_struct, std::string class_name, bool can_be_list) {
  parser_struct.line = LineCounter;
  

  int pre_tabs = SeenTabs;
  
  auto LHS = ParseUnary(parser_struct, class_name, can_be_list);
  if (!LHS)
    return nullptr;

  if (CurTok!=tok_space && pre_tabs!=SeenTabs)
    return std::move(LHS);

  return std::get<0>(ParseBinOpRHS(parser_struct, 0, std::move(LHS), class_name));
}

/// prototype
///   ::= id '(' id* ')'
///   ::= binary LETTER number? (id, id)
///   ::= unary LETTER (id)
std::unique_ptr<PrototypeAST> ParsePrototype(Parser_Struct parser_struct) {
  std::string FnName="";
  if (parser_struct.class_name!="")
    FnName = parser_struct.class_name+"_";
  std::string _class, method;
  method = "";
  _class = parser_struct.class_name;



  std::string return_type;

  if (!in_str(IdentifierStr, data_tokens)) {
    LogErrorBreakLine(parser_struct.line, "Expected prototype function return type.");
    return nullptr;
  }

  return_type = IdentifierStr;
  Data_Tree return_data_type = ParseDataTree(return_type, in_str(IdentifierStr,compound_tokens), parser_struct);
  // std::cout << "return is"  << ".\n";
  // return_data_type.Print();

  


  unsigned Kind = 0; // 0 = identifier, 1 = unary, 2 = binary.
  unsigned BinaryPrecedence = 30;

  switch (CurTok) {
  default:
    return LogErrorP(parser_struct.line, "Expected prototype function name.");
  case tok_identifier:
    FnName += IdentifierStr;
    method = IdentifierStr;
    Kind = 0;
    getNextToken();
    break;
  case tok_unary:
    getNextToken();
    if (!isascii(CurTok))
      return LogErrorP(parser_struct.line, "Esperado operador unário");
    FnName += "unary";
    FnName += (char)CurTok;
    Kind = 1;
    getNextToken();
    break;
  case tok_binary:
    getNextToken();
    if (!isascii(CurTok))
      return LogErrorP(parser_struct.line, "Esperado operador binário");
    FnName += "binary";
    FnName += (char)CurTok;
    Kind = 2;
    getNextToken();

    // Read the precedence if present.
    if (CurTok == tok_number) {
      if (NumVal < 1 || NumVal > 100)
        return LogErrorP(parser_struct.line, "Precedência inválida: deve ser entre 1 e 100");
      BinaryPrecedence = (unsigned)NumVal;
      getNextToken();
    }
    break;
  }

  if (CurTok != '(')
    return LogErrorP(parser_struct.line, "Expected \"(\" at function prototype.");
  getNextToken(); // eat (



  std::string type;
  std::vector<std::string> ArgNames, Types;

  


  Types.push_back("s");
  ArgNames.push_back("scope_struct");

  Function_Arg_Names[FnName].push_back("0");
  Function_Arg_Types[FnName]["0"] = "Scope_Struct";
  Function_Arg_DataTypes[FnName]["0"] = Data_Tree("Scope_Struct");


  while (CurTok != ')')
  {
    if (IdentifierStr=="s"||IdentifierStr=="str")
      type="str";
    else if (IdentifierStr=="t"||IdentifierStr=="tensor")
      type="tensor";
    else if (IdentifierStr=="b"||IdentifierStr=="bool")
      type="bool";
    else if (IdentifierStr=="i"||IdentifierStr=="int")
      type="int";
    else if (IdentifierStr=="f"||IdentifierStr=="float")
      type="float";
    else
      type=IdentifierStr;

    if (IdentifierStr!="s" && IdentifierStr!="t" && IdentifierStr!="f" && IdentifierStr!="i" && (!in_str(IdentifierStr, data_tokens)) && !in_str(type, Classes))
      LogErrorP_to_comma(parser_struct.line, "Prototype var type must be s, t, i, f or a data type. Got " + IdentifierStr);
    else {
      std::string data_type = type;

      Data_Tree data_tree = ParseDataTree(type, in_str(type, compound_tokens), parser_struct);

      std::string IdName = IdentifierStr;

      if(CurTok==tok_identifier)
        getNextToken(); // get arg name;
      else if(CurTok==tok_channel)
      {
        Data_Tree channel_data_tree = Data_Tree("channel");
        channel_data_tree.Nested_Data.push_back(data_tree);
        data_tree = channel_data_tree;
        data_type = data_type + "_channel";

        int channel_direction = ch_both;

        getNextToken(); // get channel

        if (CurTok==tok_arrow)
        {
          channel_direction = ch_sender;
          getNextToken(); // <- ch
        }

        if(CurTok!=tok_identifier)
          LogError(parser_struct.line, "Unexpected token " + ReverseToken(CurTok) + ". Expected argument name.");
        
        IdName = IdentifierStr;
        getNextToken(); // get arg name;
        
        if (CurTok==tok_arrow)
        {
          channel_direction = ch_receiver;
          getNextToken(); // ch <-
        }
        ChannelDirections[FnName][IdName] = channel_direction;
      } else 
        LogError(parser_struct.line, "Unexpected token " + ReverseToken(CurTok) + ". Expected argument name.");
       

      Types.push_back(data_type);
      ArgNames.push_back(IdName);

      Function_Arg_Names[FnName].push_back(IdName);
      Function_Arg_Types[FnName][IdName] = data_type;
      Function_Arg_DataTypes[FnName][IdName] = data_tree;

      typeVars[FnName][IdName] = data_type;
      data_typeVars[FnName][IdName] = data_tree;



      if(in_str(data_type, Classes))
        Object_toClass[FnName][IdName] = Data_Tree(data_type);
      
    }
    


    if (CurTok == ')')
        break;
      
    if (CurTok != ',')
    {
      return LogErrorP(parser_struct.line, "Expected ')' or ',' at prototype arguments list.");
    }
    getNextToken();
  }

  // std::cout << "CREATING PROTO WITH " << ArgNames.size() << " PARAMETERS.\n";

  // success.
  getNextToken(); // eat ')'.

  // Verify right number of names for operator.
  if (Kind && ArgNames.size() != Kind)
    return LogErrorP(parser_struct.line, "Prototype got an invalid amount of operators.");

  if (CurTok!=tok_space)
    LogError(parser_struct.line, "Post prototype parsing requires a line break.");
  getNextToken();

  functions_return_type[FnName] = return_type;
  functions_return_data_type[FnName] = return_data_type;


  return std::make_unique<PrototypeAST>(FnName, return_type, _class, method, ArgNames, Types, Kind != 0,
                                         BinaryPrecedence);
}


std::unique_ptr<ExprAST> ParseImport(Parser_Struct parser_struct) {

  getNextToken(); // eat import

  if(CurTok!=tok_identifier)
    return LogError(parser_struct.line, "Expected library name after \"import\". Got token: " + ReverseToken(CurTok) + ".");

  bool is_default = false;
  if(IdentifierStr=="default")
  {
    // std::cout << "Got a default import" << ".\n";
    is_default=true;
    getNextToken();
  }




  // Get lib name
  std::string lib_name = IdentifierStr;
  getNextToken();
  // get_tok_util_dot_or_space();

  int dots=0; 

  while(CurTok=='.')
  {
    dots++;
    getNextToken(); // get dot
    lib_name += "/" + IdentifierStr;
    getNextToken();
  }
  

  std::string full_path_lib = tokenizer.current_dir+"/"+lib_name+".ai";

  // Import logic
  if(fs::exists(full_path_lib))
  {

    // std::cout << "Reverse: " << ReverseToken(CurTok) << ".\n";
    // std::cout << "cur_line: " << cur_line << ".\n";

    get_tok_util_space();
    tokenizer.importFile(full_path_lib, dots);

    return nullptr;
  } else
    return std::make_unique<LibImportExprAST>(lib_name, is_default, parser_struct);
  

}



/// definition ::= 'def' prototype expression
std::unique_ptr<FunctionAST> ParseDefinition(Parser_Struct parser_struct, std::string class_name) {
  

  int cur_level_tabs = SeenTabs;

  getNextToken(); // eat def.


  auto Proto = ParsePrototype(parser_struct);
  if (!Proto)
  {
    std::string _error = "Error defining " + parser_struct.class_name + " prototype.";  
    LogError(parser_struct.line, _error);
    return nullptr;
  } 

  // std::cout << "PROTO NAME IS " << Proto->getName() << ".\n";
  parser_struct.function_name = Proto->getName();
  // parser_struct.function_name = class_name+"_"+Proto->getName();
  
  
  std::vector<std::unique_ptr<ExprAST>> Body;



  while(!in_char(CurTok, terminal_tokens))
  {
    if (SeenTabs <= cur_level_tabs && CurTok != tok_space)
      break;
       
    
    handle_tok_space();

    if (SeenTabs <= cur_level_tabs)
      break;

    Body.push_back(std::move(ParseExpression(parser_struct, class_name)));
  }

  //std::cout << "function number of expressions: " << Body.size() << "\n";

  if (Body.size()==0)
  {
    std::string _error = "Function " + parser_struct.function_name + "'s body was not declared.";  
    LogError(parser_struct.line, _error);
    return nullptr;
  } 

  return std::make_unique<FunctionAST>(std::move(Proto), std::move(Body));
  
}


/// toplevelexpr ::= expression
std::unique_ptr<FunctionAST> ParseTopLevelExpr(Parser_Struct parser_struct) {
  
  
  std::vector<std::unique_ptr<ExprAST>> Body;
  while(!in_char(CurTok, terminal_tokens))
  {
    Body.push_back(std::move(ParseExpression(parser_struct)));
    // std::cout << "curtok post " << CurTok << ".\n";

    while(CurTok==tok_space)
      getNextToken();
  }
  

  // Make an anonymous proto.
  auto Proto = std::make_unique<PrototypeAST>("__anon_expr", "float", "", "",
                                                std::vector<std::string>(),
                                                std::vector<std::string>());
    
  return std::make_unique<FunctionAST>(std::move(Proto), std::move(Body));
  
  return nullptr;
}



/// external ::= 'extern' prototype
std::unique_ptr<PrototypeAST> ParseExtern(Parser_Struct parser_struct) {
  getNextToken(); // eat extern.
  return ParsePrototype(parser_struct);
}



std::unique_ptr<ExprAST> ParseClass(Parser_Struct parser_struct) {
  getNextToken(); // eat class.

  if (CurTok != tok_identifier)
    return LogError(parser_struct.line, "Expected class name");
  std::string Name = IdentifierStr;

  Classes.push_back(Name);

  getNextToken(); // eat name

  if(CurTok==tok_space)
    getNextToken();


  

  int last_offset=0;
  std::vector<llvm::Type *> llvm_types;
  while(CurTok==tok_data||CurTok==tok_identifier||CurTok==tok_struct)
  { 
    std::string data_type = IdentifierStr;
    bool is_object = in_str(data_type, Classes);
    bool is_channel=false;
    
    Data_Tree data_tree = ParseDataTree(data_type, in_str(data_type, compound_tokens), parser_struct);
    
    // LogBlue("data type is: " + data_type);
    
    if (CurTok==tok_channel) 
    {
      is_channel=true;

      data_type = "channel";
      Data_Tree channel_data_tree = Data_Tree("channel");
      channel_data_tree.Nested_Data.push_back(data_tree); 
      data_tree = channel_data_tree;

      getNextToken();
    }
    
    
    while(true)
    {
      if (CurTok!=tok_identifier)
        LogError(parser_struct.line, "Class " + Name + " variables definition requires simple non-attribute names.");

      if (is_object) {
        Object_toClass[Name][IdentifierStr] = Data_Tree(data_type);
      }
      typeVars[Name][IdentifierStr] = data_type; 
      data_typeVars[Name][IdentifierStr] = data_tree; 
      ClassVariables[Name][IdentifierStr] = last_offset;
      
      if (data_type=="float"||data_type=="int") {
        llvm_types.push_back(Type::getInt32Ty(*TheContext));
        last_offset+=4;
      } else if(data_type=="bool") {
        llvm_types.push_back(Type::getInt1Ty(*TheContext));
        last_offset+=1;
      } else {
        ClassPointers[Name].push_back(last_offset);
        ClassPointersType[Name].push_back(data_type);
        // LogBlue("push class pointer type " + data_type + " for class " + Name + " on offset " + std::to_string(last_offset) + " for " + IdentifierStr);
        llvm_types.push_back(int8PtrTy);
        last_offset+=8;
      }

      if(is_channel)
        ChannelDirections[Name][IdentifierStr] = ch_both;
      

      getNextToken(); // eat name

      if (CurTok!=',')
        break;
      getNextToken(); // eat ','
    }
    while (CurTok==tok_space)
      getNextToken();
  }
  ClassSize[Name] = last_offset;


  

  // for (auto &pair : ClassVariables[Name])
  // {
  //   std::cout << Name << ": " << pair.first << " - " << pair.second << ".\n";
  // }
  
  // llvm::Type *class_struct = StructType::create(*TheContext);
  // class_struct->setBody(llvm_types);
  // ClassStructs[Name] = class_struct; // I fear this approach may lead to stack overflow, like what happend to the previous string allocas.






  if (CurTok!=tok_def)
    return LogError(parser_struct.line, "A class definition requires it's functions. Got token: " + std::to_string(CurTok) + "/" + ReverseToken(CurTok));
    // return LogError(parser_struct.line, "A class definition requires it's functions. Got token: " + ReverseToken(CurTok));

  int i=0;

  parser_struct.class_name = Name;

  while(CurTok==tok_def)
  {
    if(SeenTabs==0)
      break;
    
    auto Func = ParseDefinition(parser_struct, Name);
    if (!Func)
      return nullptr;
      //return LogError(parser_struct.line, "Falha no parsing da função da Classe.");
    if (!ends_with(Func->getProto().getName(),"__init__") && i==0)
      return LogError(parser_struct.line, "Class requires __init__ method");
    
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