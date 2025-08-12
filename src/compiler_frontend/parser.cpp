#include "llvm/IR/Value.h"


#include <cstdio>
#include <stdio.h>
#include <iostream>
#include <map>
#include <random>
#include <thread>
#include <vector>



#include "../codegen/string.h"
#include "../common/include.h"
#include "include.h"




std::map<std::string, std::map<std::string, std::string>> Object_toClass;
std::map<std::string, std::vector<std::string>> Equivalent_Types;

std::map<std::string, std::map<std::string, std::string>> Function_Arg_Types;
std::map<std::string, std::vector<std::string>> Function_Arg_Names;

using namespace llvm;



std::map<std::string, std::string> ops_type_return;
std::map<int, std::string> op_map;
std::vector<std::string> op_map_names;

std::map<std::string, std::vector<std::string>> data_typeVars;
std::map<std::string, std::map<std::string, std::string>> typeVars;

std::map<std::string, std::map<std::string, int>> ClassVariables;
std::map<std::string, int> ClassSize;
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
    

    return input; // Return original string if "_list" not found
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

std::unique_ptr<ExprAST> ParseStringExpr(Parser_Struct parser_struct) {
  auto Result = std::make_unique<StringExprAST>(IdentifierStr);
  getNextToken(); // consume the "
  return std::move(Result);
}


inline void handle_tok_space() {
  if(CurTok==tok_space)
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



std::unique_ptr<ExprAST> ParseObjectInstantiationExpr(Parser_Struct parser_struct, std::string _class, std::string class_name) {
  getNextToken();
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
  while (true) {

    Name = IdentifierStr;
    typeVars[parser_struct.function_name][Name] = _class;



    if (!is_self)
    {
      // std::cout << "========Object instatiation of " << IdentifierStr << "/" << _class << ".\n";
      Object_toClass[parser_struct.function_name][IdentifierStr] = _class;
    } else {

      std::cout << "++++++++Object instatiation of " << IdentifierStr << "/" << _class << ".\n";
    }

    getNextToken(); // eat identifier.

        
    std::unique_ptr<ExprAST> Init = nullptr;  
    VarNames.push_back(std::make_pair(Name, std::move(Init)));

    // End of var list, exit loop.
    if (CurTok != ',')
      break;
    getNextToken(); // eat the ','.

    if (CurTok != tok_identifier)
      return LogError(parser_struct.line, "Expected object identifier names.");
  }


  auto aux = std::make_unique<ObjectExprAST>(parser_struct, std::move(VarNames), "object", std::move(VecInitSize), ClassSize[_class]);
  aux->SetSelf(is_self);
  aux->SetIsAttribute(is_attr);
  aux->SetPreDot(pre_dot);


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



std::unique_ptr<ExprAST> ParseIdentifierListExpr(Parser_Struct parser_struct, std::string class_name, bool can_be_string, std::vector<std::tuple<std::string, int, std::vector<std::unique_ptr<ExprAST>>>> Names) {

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



/// identifierexpr
///   ::= identifier
///   ::= identifier '(' expression* ')'
std::unique_ptr<ExprAST> ParseIdentifierExpr(Parser_Struct parser_struct, std::string class_name, bool can_be_string, bool can_be_list) {
  
  for(int i=0; i<Classes.size(); i++)
    if(IdentifierStr==Classes[i])  // Object object
      return ParseObjectInstantiationExpr(parser_struct, Classes[i], class_name);

  std::vector<std::tuple<std::string, int, std::vector<std::unique_ptr<ExprAST>>>> Names;
  std::string IdName, type;
  IdName = IdentifierStr;
  //std::cout << "Identifier " << IdName <<  "\n";
  getNextToken(); // eat identifier.
  
  Names.push_back(std::make_tuple(IdName, type_var, std::vector<std::unique_ptr<ExprAST>>{}));



  std::unique_ptr<ExprAST> aux;

  if (CurTok==',' && can_be_list)
  {

    std::vector<std::unique_ptr<ExprAST>> IdentifierList;
    while(true)
    {
      IdName = IdentifierStr;

      if (typeVars[parser_struct.function_name].find(IdName) != typeVars[parser_struct.function_name].end())
        type = typeVars[parser_struct.function_name][IdName];
      else
        type = "none";

      
      if (type!="none")
      {
        auto name_solver_expr = std::make_unique<NameSolverAST>(std::move(Names));
        aux = std::make_unique<VariableExprAST>(std::move(name_solver_expr), type, IdName, parser_struct);
      } else {
        if(!can_be_string)
        {
          std::string _error = "Variable " + IdName + " was not found on scope "+parser_struct.function_name+".";
          return LogError(parser_struct.line, _error);
        }  

        aux = std::make_unique<StringExprAST>(IdName);
      }

      IdentifierList.push_back(std::move(aux));


      if (CurTok!=',')
        break;
      getNextToken(); // get comma
      getNextToken(); // get identifier
      Names.push_back(std::make_tuple(IdentifierStr, type_var, std::vector<std::unique_ptr<ExprAST>>{}));
    }


    aux = std::make_unique<VariableListExprAST>(std::move(IdentifierList));
    return std::move(aux);
  } 



  if (CurTok != '(' && CurTok != '[') // Simple variable ref.
  {
    if (typeVars[parser_struct.function_name].find(IdName) != typeVars[parser_struct.function_name].end())
      type = typeVars[parser_struct.function_name][IdName];
    else
    {
      type = "none";
      // std::string _error = "Variable " + IdName + " not found.";
      // return LogError(parser_struct.line, _error);
    } 

    // std::cout << "Var type is: " << type << ".\n";

    if (type!="none")
    {
      auto name_solver_expr = std::make_unique<NameSolverAST>(std::move(Names));
      aux = std::make_unique<VariableExprAST>(std::move(name_solver_expr), type, IdName, parser_struct);
    } else {
      if(!can_be_string)
      {
        std::string _error = "Variable " + IdName + " was not found on scope " + parser_struct.function_name + ".";
        return LogError(parser_struct.line, _error);
      }  

      aux = std::make_unique<StringExprAST>(IdName);
    }
    
    

    return std::move(aux);
  }


  

  if (CurTok=='[')
  {
    getNextToken(); // eat [
    
    std::unique_ptr<IndexExprAST> Idx = ParseIdx(parser_struct, class_name);
    
    if (typeVars[parser_struct.function_name].find(IdName) != typeVars[parser_struct.function_name].end())
      type = typeVars[parser_struct.function_name][IdName];
    

    auto name_solver_expr = std::make_unique<NameSolverAST>(std::move(Names));

    aux = std::make_unique<VariableExprAST>(std::move(name_solver_expr), type, IdName, parser_struct);
    aux = std::make_unique<VecIdxExprAST>(std::move(aux), std::move(Idx), type);
    aux->SetIsVec(true);
    
    getNextToken(); // eat ]
    
    return std::move(aux);
  }
  

  if (CurTok=='(')
  {
    // Call.
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
          Args.push_back(std::move(Arg));
        else
          return nullptr;
        

        if (CurTok == ')')
          break;

        if (CurTok != ',')
          return LogError(parser_struct.line, "Expected ')' or ',' on argument list");
        getNextToken();
      }
    }

    // varargs
    if (in_str(IdName, vararg_methods))
      Args.push_back(std::make_unique<IntExprAST>(TERMINATE_VARARG));
    
    

    // Eat the ')'.
    getNextToken();

    
    std::string callee_override = "none";
    bool name_solve_to_last = false;
    if(typeVars[parser_struct.function_name].count(IdName)>0)
    {
      name_solve_to_last = true;
      callee_override = typeVars[parser_struct.function_name][IdName];
    }
    

    bool is_var_forward = false;
    bool return_tensor = false;
    if (floatFunctions.find(IdName) != floatFunctions.end()) // if found
    {
      is_var_forward = true;
      callee_override = floatFunctions[IdName];
    }
    if (IdName=="to_float")
    {
      callee_override = "StrToFloat";
      is_var_forward = true;
    }
    
    auto name_solver_expr = std::make_unique<NameSolverAST>(std::move(Names));
    name_solver_expr->SetNameSolveToLast(name_solve_to_last);
    // if ()
    // name_solver_expr->SetType;
    std::string scope_random_string = RandomString(14);

    

    aux = std::make_unique<CallExprAST>(std::move(name_solver_expr), IdName, IdName, std::move(Args),
                                                  "None", "None", "none", is_var_forward, callee_override, scope_random_string, "none", parser_struct);

  

    std::string fname = (callee_override!="none") ? callee_override : IdName;


    if (functions_return_type.count(fname)>0)
    {
      // std::cout << "----RETURN OF " << fname << " IS: " << functions_return_type[fname] << ".\n";
      aux->SetType(functions_return_type[fname]);  
    }
    if (return_tensor)
      aux->SetType("tensor");

    
    if(aux->GetType()=="void")
      LogBlue(IdName + ", " + fname + ": ");

    
    if (CurTok == tok_post_class_attr_identifier)
      return ParseChainCallExpr(parser_struct, std::move(aux), class_name);
    
    return aux;
  }
  
  // if (CurTok==',')
  // {
  //   auto aux = ParseIdentifierListExpr(Parser_Struct parser_struct, class_name, can_be_string, std::move(Names));
  //   return aux;
  // }
}





/// ifexpr ::= 'if' expression 'then' expression 'else' expression
std::unique_ptr<ExprAST> ParseIfExpr(Parser_Struct parser_struct, std::string class_name) {
  
  int cur_level_tabs = SeenTabs;

  getNextToken(); // eat the if.

  
  //std::cout << "If tabs level: " << cur_level_tabs <<  "\n";
  

  // condition.
  auto Cond = ParseExpression(parser_struct, class_name);
  if (!Cond)
    return nullptr;

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




std::vector<std::unique_ptr<ExprAST>> ParseIdentedBodies(Parser_Struct parser_struct, int cur_level_tabs, std::string class_name)
{
  std::vector<std::unique_ptr<ExprAST>> Body, NullBody;
  //std::cout << "\nSeen tabs on for body POST: " << SeenTabs << "\n\n";
  if (CurTok==tok_space)
    getNextToken();

  while(true)
  {
    //std::cout << "\n\nParsing new expression with tabs: " << SeenTabs << " tok: " << ReverseToken(CurTok) << "\n";
    if (SeenTabs <= cur_level_tabs && CurTok != tok_space)
    {
      //std::cout << "Breaking for with cur tok: " << ReverseToken(CurTok) << " Seen Tabs:" << SeenTabs <<  "\n";
      break;
    } 
    //std::cout << "\nSeen tabs on for body: " << SeenTabs << "\nCur tok: " << ReverseToken(CurTok) << "\n\n";

    while (CurTok == tok_space)
    {
      //std::cout << "\nJumping tok space\n\n";
      getNextToken();
    }

    //std::cout << "Post space has " << SeenTabs << " tabs.\n";
    if (SeenTabs <= cur_level_tabs)
      break;


    auto body = ParseExpression(parser_struct, class_name);
    if (!body)
      return std::move(NullBody);
    Body.push_back(std::move(body));
    //getNextToken();
  }

  if (CurTok==tok_space)
    getNextToken();

  return std::move(Body);
}





std::unique_ptr<ExprAST> ParseStandardForExpr(Parser_Struct parser_struct, std::string class_name, int cur_level_tabs, std::string IdName) {
  getNextToken(); // eat '='.

  auto Start = ParseExpression(parser_struct, class_name);
  if (!Start)
    return nullptr;
  if (CurTok != ',')
    return LogError(parser_struct.line, "Expected ',' after for's control variable initial value.");
  getNextToken();

  if(Start->GetType()=="float")
    typeVars[parser_struct.function_name][IdName] = "float";


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

  Body = ParseIdentedBodies(parser_struct, cur_level_tabs, class_name);

  return std::make_unique<ForExprAST>(IdName, std::move(Start), std::move(End),
                                       std::move(Step), std::move(Body), parser_struct);
}


std::unique_ptr<ExprAST> ParseForEachExpr(Parser_Struct parser_struct, std::string class_name, int cur_level_tabs, std::string IdName) {
  std::string data_type = IdName;

  getNextToken(); // eat data type
  IdName = IdentifierStr;
  getNextToken(); // eat identifier name

  if (CurTok!=tok_in)
    LogError(parser_struct.line, "Expected in at for each expression");




  if (in_str(data_type, Classes))
    Object_toClass[parser_struct.function_name][IdName] = data_type;

  typeVars[parser_struct.function_name][IdName] = data_type;

  getNextToken(); // eat "in".


  auto Vec = ParseExpression(parser_struct, class_name);
  if (!Vec)
    return nullptr;

  std::string vec_type = Vec->GetType();

  
  std::vector<std::unique_ptr<ExprAST>> Body;

  Body = ParseIdentedBodies(parser_struct, cur_level_tabs, class_name);

  return std::make_unique<ForEachExprAST>(IdName, std::move(Vec), std::move(Body), parser_struct, data_type, vec_type);
}


/// forexpr ::= 'for' identifier '=' expr ',' expr (',' expr)? 'in' expression
std::unique_ptr<ExprAST> ParseForExpr(Parser_Struct parser_struct, std::string class_name) {

  int cur_level_tabs = SeenTabs;

  //std::cout << "\nSeen tabs on for: " << SeenTabs << "\n\n";

  getNextToken(); // eat the for.

  if (CurTok==tok_data||in_str(IdentifierStr, Classes))
    return ParseForEachExpr(parser_struct, class_name, cur_level_tabs, IdentifierStr);

  if (CurTok != tok_identifier)
    return LogError(parser_struct.line, "Expected for's control variable identifier.");

  std::string IdName = IdentifierStr;
  getNextToken(); // eat identifier.

  typeVars[parser_struct.function_name][IdName] = "int";

  if (CurTok=='=')
    return ParseStandardForExpr(parser_struct, class_name, cur_level_tabs, IdName);
  else if(CurTok==tok_in)
    return ParseForEachExpr(parser_struct, class_name, cur_level_tabs, "int");
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
  
  std::vector<std::unique_ptr<ExprAST>> Body = ParseIdentedBodies(parser_struct, cur_level_tabs, class_name);

  return std::make_unique<WhileExprAST>(std::move(Cond), std::move(Body));
}



std::unique_ptr<ExprAST> ParseAsyncExpr(Parser_Struct parser_struct, std::string class_name) {

  int cur_level_tabs = SeenTabs;

  getNextToken(); // eat the async.

  
  std::vector<std::unique_ptr<ExprAST>> Bodies;
  
  //std::cout << "Pre expression token: " << ReverseToken(CurTok) << "\n";


  Bodies.push_back(std::make_unique<IncThreadIdExprAST>());
  if (CurTok != tok_space)
    Bodies.push_back(std::move(ParseExpression(parser_struct, class_name)));
  else
    Bodies = ParseIdentedBodies(parser_struct, cur_level_tabs, class_name);
  
  
  
  //std::cout << "Post async: " << ReverseToken(CurTok) << "\n";

  return std::make_unique<AsyncExprAST>(std::move(Bodies), parser_struct);
}

std::unique_ptr<ExprAST> ParseAsyncsExpr(Parser_Struct parser_struct, std::string class_name) {

  int cur_level_tabs = SeenTabs;

  getNextToken(); // eat the async.


  if (CurTok!=tok_int)
    LogError(parser_struct.line, "asyncs expression expect the number of asynchrnonous functions.");

  int async_count = NumVal;
  getNextToken();
  
  std::vector<std::unique_ptr<ExprAST>> Bodies;
  
  //std::cout << "Pre expression token: " << ReverseToken(CurTok) << "\n";

  for (auto pair : typeVars[parser_struct.function_name])
    typeVars["asyncs"][pair.first] = pair.second;

  parser_struct.function_name = "asyncs";
  Bodies.push_back(std::make_unique<IncThreadIdExprAST>());
  if (CurTok != tok_space)
    Bodies.push_back(std::move(ParseExpression(parser_struct, class_name)));
  else
    Bodies = ParseIdentedBodies(parser_struct, cur_level_tabs, class_name);
  
  
  
  //std::cout << "Post async: " << ReverseToken(CurTok) << "\n";

  return std::make_unique<AsyncsExprAST>(std::move(Bodies), async_count, parser_struct);
}



std::unique_ptr<ExprAST> ParseFinishExpr(Parser_Struct parser_struct, std::string class_name) {

  int cur_level_tabs = SeenTabs;
  //std::cout << "Finish tabs level: " << cur_level_tabs <<  "\n";

  getNextToken(); // eat the finish.


  std::vector<std::unique_ptr<ExprAST>> Bodies;
  std::vector<bool> IsAsync;
  

  if (CurTok!=tok_space)
    LogError(parser_struct.line, "Finish requires line break.");
  getNextToken(); 


  while(!in_char(CurTok, terminal_tokens))
  {

    if (SeenTabs <= cur_level_tabs && CurTok != tok_space)
    {
      //std::cout << "Breaking finish with cur tok: " << ReverseToken(CurTok) << "\n";
      //std::cout << "Current tabs: " << SeenTabs << "\n";
      break;
    }
    //std::cout << "\nSeen tabs on finish body: " << SeenTabs << "\nCur tok: " << CurTok << "\n\n";




    if (CurTok==tok_space)
      getNextToken();
    

    /*
    if (CurTok == tok_async)
    {
      Bodies.push_back(std::move(ParseAsyncExpr(Parser_Struct parser_struct, class_name)));
      IsAsync.push_back(true);
    }
    else
    {
      Bodies.push_back(std::move(ParseExpression(parser_struct, class_name)));
      IsAsync.push_back(false);
    }
    */
    Bodies.push_back(std::move(ParseExpression(parser_struct, class_name)));
    IsAsync.push_back(false);
  }


  return std::make_unique<FinishExprAST>(std::move(Bodies),
                                         std::move(IsAsync));
}




std::unique_ptr<ExprAST> ParseNewVector(Parser_Struct parser_struct, std::string class_name) {
  std::cout << "Parsing new vector" << ReverseToken(CurTok)  << "\n";


  bool is_unknown_type=false;
  std::string first_element_type="";
  int i=0;


  
  getNextToken(); // [
  std::vector<std::unique_ptr<ExprAST>> Elements;
  if (CurTok != ']') {
    while (true) {
      // std::cout << "CURRENT TOKEN: " << ReverseToken(CurTok) << ".\n";
      std::string element_type;


      if (auto element = ParseExpression(parser_struct, class_name, false))
      {
        element_type = element->GetType();
        if (i==0)
          first_element_type = element_type;
        else {
          if (element_type!=first_element_type)
            is_unknown_type=true;
        }

        
 
        Elements.push_back(std::make_unique<StringExprAST>(element_type));
        Elements.push_back(std::move(element));
        i+=1;
      } 
      else
        return nullptr;

      if (CurTok == ']')
        break;
      if (CurTok != ',')
      {
        LogError(parser_struct.line, "Expected ']' or ',' on the List elements list.");
      }
      getNextToken();
    }
  }   
  getNextToken(); // ]

  std::string type = ((is_unknown_type) ? "unknown" : first_element_type) + "_list";
  
  Elements.push_back(std::make_unique<StringExprAST>("TERMINATE_VARARG"));

  //TODO: vector for other types
  return std::make_unique<NewVecExprAST>(std::move(Elements), type);
}



std::unique_ptr<ExprAST> ParseNewDict(Parser_Struct parser_struct, std::string class_name) {


  bool is_unknown_type=false;
  std::string first_element_type="";
  int i=0;


  
  getNextToken(); // {
  std::vector<std::unique_ptr<ExprAST>> Keys, Elements;
  if (CurTok != '}') {
    while (true) {
      // std::cout << "CURRENT TOKEN: " << ReverseToken(CurTok) << ".\n";
      std::string element_type;

      if (auto key = ParseExpression(parser_struct, class_name, false))
        Keys.push_back(std::move(key));
      else {
        LogError(parser_struct.line, "Expected an expression at dict key.");
        return nullptr;
      }


      if (CurTok!=':')
      {
        LogError(parser_struct.line, "Expected \":\" at dictionary.");
        return nullptr;
      }
      getNextToken(); //:


      if (auto element = ParseExpression(parser_struct, class_name, false))
      {
        element_type = element->GetType();
        if (i==0)
          first_element_type = element_type;
        else {
          if (element_type!=first_element_type)
            is_unknown_type=true;
        }

        
        Elements.push_back(std::move(element));
        i+=1;
      } 
      else
        return nullptr;

      if (CurTok == '}')
        break;
      if (CurTok != ',')
      {
        LogError(parser_struct.line, "Expected '}' or ',' at the new dict expression.");
      }
      getNextToken();
    }
  }   
  getNextToken(); // }

  std::string type = ((is_unknown_type) ? "unknown" : first_element_type) + "_dict";
  
  Elements.push_back(std::make_unique<StringExprAST>("TERMINATE_VARARG"));

  return std::make_unique<NewDictExprAST>(std::move(Keys), std::move(Elements), type, parser_struct);
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

        LogError(parser_struct.line, "Expected ')' or ',' on the Function Call arguments list.");
        return {};
      }
      getNextToken();
    }
  }
  // Eat the ')'.
  getNextToken();
  return std::move(Args);
}


std::unique_ptr<ExprAST> ParseSelfExpr(std::unique_ptr<NameableExprAST> inner_expr, Parser_Struct parser_struct, std::string class_name) {

  std::string pre_dot = "";
  std::string type = "None";
  std::string object_class;
  bool is_class_attr=false;
  bool is_self=false;
  bool is_vec=false;
  std::vector<std::tuple<std::string, int, std::vector<std::unique_ptr<ExprAST>>>> Names;
  std::string Prev_IdName = "";
  std::string IdName = "";


  is_class_attr = true;
  pre_dot="";


  // tok_identifier: abc
  // tok_self: self.
  // tok_class_attr: abc.
  // tok_post_class_attr_attr: .abc.
  // tok_post_class_attr_identifier: .abc

  if(CurTok==tok_self)
  { 
    std::unique_ptr<SelfExprAST> self_expr = std::make_unique<SelfExprAST>();
    getNextToken();
    return std::move(ParseSelfExpr(std::move(self_expr), parser_struct, class_name));
  }

  if (CurTok==tok_identifier||CurTok==tok_class_attr||CurTok==tok_post_class_attr_attr||CurTok==tok_post_class_attr_identifier)
  {
    IdName = IdentifierStr;
    getNextToken();
 
    std::unique_ptr<NestedStrExprAST> nested_expr = std::make_unique<NestedStrExprAST>(std::move(inner_expr), IdName, parser_struct);

    return std::move(ParseSelfExpr(std::move(nested_expr), parser_struct, class_name));
  }




  // Parse call expression
  if (CurTok=='(')
  {

    

    std::string Prev_IdName;

    IdName = inner_expr->Name;
    Prev_IdName = inner_expr->Inner_Expr->Name;
    

    std::string callee = IdName;

    inner_expr->skip=true; // skips the pointer object logic for the last name




    std::string prev_nested_name = Get_Nested_Name(inner_expr->Expr_String, parser_struct, false);
    std::string nested_name = Get_Nested_Name(inner_expr->Expr_String, parser_struct, true);
    std::string prev_fn_name = (prev_nested_name=="") ? parser_struct.function_name : prev_nested_name;
    std::string fn_name = (nested_name=="") ? parser_struct.function_name : nested_name;





    bool is_from_nsk=true;
    // model.linear_1(x)
    if(typeVars[prev_fn_name].count(IdName)>0)
    {
      callee = typeVars[prev_fn_name][IdName];
      inner_expr->skip=false;
    }
    // x.view()
    else if(typeVars[fn_name].count(Prev_IdName)>0)
      callee = typeVars[fn_name][Prev_IdName] + "_" + IdName;
    else {
      is_from_nsk=false; 
    }


    
    callee = Extract_List_Suffix(callee); // deals with nsk list methods



    


    if(inner_expr->skip)
    {
      inner_expr = std::move(inner_expr->Inner_Expr);
      inner_expr->IsLeaf=true;
    }

    if (prev_nested_name!=""&&!is_from_nsk)
      callee = prev_nested_name+"_"+callee; // Add variable high-level class to the name


    std::vector<std::unique_ptr<ExprAST>> arguments = Parse_Arguments(parser_struct, class_name);
    if (in_str(callee, vararg_methods))
      arguments.push_back(std::make_unique<IntExprAST>(TERMINATE_VARARG));


    // std::cout << "CALLING: " << callee << ".\n";
    std::unique_ptr<NestedCallExprAST> call_expr = std::make_unique<NestedCallExprAST>(std::move(inner_expr), callee, parser_struct, std::move(arguments));


    if (functions_return_type.count(callee)>0)
    {
      call_expr->SetType(functions_return_type[callee]);
    }


    if (CurTok == tok_post_class_attr_identifier)
      return ParseChainCallExpr(parser_struct, std::move(call_expr), class_name);

    


    return std::move(call_expr);
  }



  if (CurTok=='[') {


    IdName = inner_expr->Name;

    std::string fn_name = Get_Nested_Name(inner_expr->Expr_String, parser_struct, false);
    // std::string fn_name = (inner_expr->From_Self) ? parser_struct.class_name : parser_struct.function_name;


    if (typeVars[fn_name].find(IdName) != typeVars[fn_name].end())
      type = typeVars[fn_name][IdName];
    else {
      std::string _error = "Idexing self/attribute variable " + IdName + " was not found on scope " + fn_name + ".";
      return LogError(parser_struct.line, _error);
      // type = "none";
    }


    getNextToken(); // eat [
    std::unique_ptr<IndexExprAST> Idx = ParseIdx(parser_struct, class_name);
    getNextToken(); // eat ]

    
    std::unique_ptr<NestedVectorIdxExprAST> vec_expr = std::make_unique<NestedVectorIdxExprAST>(std::move(inner_expr), IdName, parser_struct, std::move(Idx), type);
    vec_expr->SetIsAttribute(true);

    // if(type=="list")
    // {

    //   std::cout << "\n\n\n\n------------\n\nOH YEAH I AM A LIST -----------------------------" << ".\n\n\n\n\n";
    //   std::exit(0);
    // }

    if(CurTok==tok_class_attr||CurTok==tok_post_class_attr_attr||CurTok==tok_post_class_attr_identifier)
    {

      return std::move(ParseSelfExpr(std::move(vec_expr), parser_struct, class_name));
    }


    return std::move(vec_expr);
  }



  // Parse variable expression

  
  std::string fn_name = Get_Nested_Name(inner_expr->Expr_String, parser_struct, false);
  // fn_name = (fn_name=="") ? parser_struct.function_name : fn_name;

  IdName = inner_expr->Name;

  if (typeVars[fn_name].find(IdName) != typeVars[fn_name].end())
    type = typeVars[fn_name][IdName];
  else {
    LogError(parser_struct.line, "here");
    std::string _error = "Self/attribute variable " + IdName + " was not found on scope " + fn_name + ".";
    return LogError(parser_struct.line, _error);
    // type = "none";
  }



  std::unique_ptr<NestedVariableExprAST> var_expr = std::make_unique<NestedVariableExprAST>(std::move(inner_expr), parser_struct, type);
  var_expr->SetIsAttribute(true);
  
  return std::move(var_expr);
}
















std::unique_ptr<ExprAST> ParseChainCallExpr(Parser_Struct parser_struct, std::unique_ptr<ExprAST> previous_call_expr, std::string class_name) {

  std::string IdName = IdentifierStr;
  getNextToken();



  std::vector<std::unique_ptr<ExprAST>> Args = Parse_Argument_List(parser_struct, class_name, "Chain Function Call");
    
  
  std::string type = previous_call_expr->Type;
  std::string call_of = type + "_" + IdName;
  
  // varargs
  if (in_str(call_of, vararg_methods))
    Args.push_back(std::make_unique<IntExprAST>(TERMINATE_VARARG));
  
  
  

  auto aux = std::make_unique<ChainCallExprAST>(call_of, std::move(Args), std::move(previous_call_expr), parser_struct);
  if (functions_return_type.count(call_of)>0)
    aux->SetType(functions_return_type[call_of]);
  

  if (CurTok == tok_post_class_attr_identifier)  
    return ParseChainCallExpr(parser_struct, std::move(aux), class_name);


  return std::move(aux);
}







std::unique_ptr<ExprAST> ParseDataExpr(Parser_Struct parser_struct, std::string suffix, std::string class_name) {

  // std::cout << "Parsing data with data type: " << IdentifierStr << ".\n";

  std::string data_type = suffix + IdentifierStr;

  getNextToken(); // eat data token.

  if (CurTok==tok_data && (IdentifierStr=="list"||IdentifierStr=="dict"))
    return ParseDataExpr(parser_struct, data_type+"_", class_name);


  if(data_type=="list")
    data_type="unknown_list";
  if(data_type=="dict")
    data_type="unknown_dict";
    

  
  std::vector<std::pair<std::string, std::unique_ptr<ExprAST>>> VarNames;
  std::vector<std::unique_ptr<ExprAST>> notes;

  if (CurTok == '[')
  {
    getNextToken();
     
    while (true) {
      if (CurTok != tok_number && CurTok != tok_int && CurTok != tok_identifier && CurTok != tok_self)
        return LogError(parser_struct.line, "Expected a number or var on the tensor dimension.");
      
      if (CurTok==tok_number)
      { 
        notes.push_back(std::make_unique<NumberExprAST>(NumVal));
        getNextToken();
      } else if (CurTok==tok_number) {
        notes.push_back(std::make_unique<IntExprAST>(NumVal));
        getNextToken();
        
      } else if (CurTok==tok_identifier)
        notes.push_back(std::move(ParseIdentifierExpr(parser_struct, class_name, true, false)));
      else {
        notes.push_back(std::move(ParsePrimary(parser_struct, class_name, false)));
      }

      
      if (CurTok != ',')
        break;
      getNextToken(); // eat the ','.
    }

    
    if (CurTok != ']')
      return LogError(parser_struct.line, "] not found at tensor declaration.");
    getNextToken();
  }




  std::string pre_dot="";
  bool is_self = false;
  bool is_attr = false;
  if (CurTok == tok_self)
  {
    is_self=true; //TODO: set self per VarName instead.
    getNextToken();
  }
  if (CurTok == tok_class_attr)
  {
    is_attr=true;
    pre_dot = IdentifierStr;
    getNextToken();
  }

  if (CurTok != tok_identifier)
    return LogError(parser_struct.line, "Expected " + data_type + " identifier name.");



  while (true) {
    std::string Name = IdentifierStr;

    typeVars[parser_struct.function_name][IdentifierStr] = data_type;
    getNextToken(); // eat identifier.

    


    std::unique_ptr<ExprAST> Init;
    if (CurTok == '=')
    {
      getNextToken(); // eat the '='.
      Init = ParseExpression(parser_struct, class_name);
      if (!Init)
        return nullptr;
    } else
    {
      if (data_type=="float")
        Init = std::make_unique<NumberExprAST>(0.0f);
      else if (data_type=="int")
        Init = std::make_unique<IntExprAST>(0);
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
      return LogError(parser_struct.line, "Expected " + data_type + " identifier names.");
  }



  auto aux = std::make_unique<DataExprAST>(parser_struct, std::move(VarNames), data_type,
                                             std::move(notes));
  aux->SetSelf(is_self);
  aux->SetIsAttribute(is_attr);
  aux->SetPreDot(pre_dot);
  
  

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
  
  
  std::vector<std::unique_ptr<ExprAST>> Body = ParseIdentedBodies(parser_struct, cur_level_tabs, class_name);


  return std::make_unique<LockExprAST>(std::move(Body), Name);
}



std::unique_ptr<ExprAST> ParseNoGradExpr(Parser_Struct parser_struct, std::string class_name) {
  int cur_level_tabs = SeenTabs;
  getNextToken(); // eat no_grad
  
  std::vector<std::unique_ptr<ExprAST>> Body = ParseIdentedBodies(parser_struct, cur_level_tabs, class_name);

  return std::make_unique<NoGradExprAST>(std::move(Body));
}



std::unique_ptr<ExprAST> ParseMustBeVar(Parser_Struct parser_struct, std::string class_name, std::string expr_name) {

  std::unique_ptr<ExprAST> expr;

  if (CurTok==tok_class_attr||CurTok==tok_self)
    expr = ParseSelfExpr(std::make_unique<EmptyStrExprAST>(), parser_struct, class_name);
  else if (CurTok==tok_identifier)
    expr = ParseIdentifierExpr(parser_struct, class_name, false, false);
  else
  {
    std::string _error = expr_name + " expression expected a simple identifier, not another expression.";
    LogError(parser_struct.line, _error);
  }

  return std::move(expr);
}



std::unique_ptr<ExprAST> ParseGlobalExpr(Parser_Struct parser_struct, std::string class_name) {
  getNextToken(); // eat global
  std::cout << "Parsing global expr\n";


  // At least one variable name is required.
  if (CurTok != tok_identifier)
    return LogError(parser_struct.line, "Expected identifier after global.");

  while (true) {


    if (CurTok!=tok_identifier)
      return LogError(parser_struct.line, "Global expression must contain identifiers only.");

    ParseIdentifierExpr(parser_struct, class_name);

    // End of var list, exit loop.
    if (CurTok != ',')
      break;
    getNextToken(); // eat the ','.

    
  }

  if (CurTok==tok_space)
    getNextToken();


  return std::make_unique<NumberExprAST>(0.0f);
}


std::unique_ptr<ExprAST> ParseRetExpr(Parser_Struct parser_struct, std::string class_name) {
  getNextToken(); //eat return

  std::vector<std::unique_ptr<ExprAST>> Vars;

  std::unique_ptr<ExprAST> expr;
  

  if (CurTok != tok_identifier && CurTok != tok_class_attr && CurTok != tok_self && CurTok != tok_number && CurTok != tok_int)
    return LogError(parser_struct.line, "Expected identifier after return.");

  
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
      expr = ParseMustBeVar(parser_struct, class_name, "return");
    
    Vars.push_back(std::move(expr));
    if(CurTok!=',')
      break;
    getNextToken(); // eat ,
  }

  return make_unique<RetExprAST>(std::move(Vars));
}






/// primary
///   ::= identifierexpr
///   ::= numberexpr
///   ::= parenexpr
///   ::= ifexpr
///   ::= forexpr
std::unique_ptr<ExprAST> ParsePrimary(Parser_Struct parser_struct, std::string class_name, bool can_be_list) {
  // std::cout << "ParsePrimary: " << ReverseToken(CurTok) << "can be list: " << can_be_list << ".\n";
  switch (CurTok) {
  default:
    //return std::move(std::make_unique<NumberExprAST>(0.0f));
    return LogErrorT(parser_struct.line, CurTok);
  case tok_identifier:
    return ParseIdentifierExpr(parser_struct, class_name, false, can_be_list);
  case tok_class_attr:
    return ParseSelfExpr(std::make_unique<EmptyStrExprAST>(), parser_struct, class_name);
  case tok_self:
    return ParseSelfExpr(std::make_unique<EmptyStrExprAST>(), parser_struct, class_name);
  case tok_number:
    return ParseNumberExpr(parser_struct);
  case tok_int:
    return ParseIntExpr(parser_struct);
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
  case tok_lock:
    return ParseLockExpr(parser_struct, class_name);
  case tok_no_grad:
    return ParseNoGradExpr(parser_struct, class_name);
  case tok_ret:
    return ParseRetExpr(parser_struct, class_name);
  case tok_data:
    return ParseDataExpr(parser_struct, "", class_name);
  case tok_global:
    return ParseGlobalExpr(parser_struct);
  case '[':
    return ParseNewVector(parser_struct, class_name);
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
  
  //std::cout << "Unary current token " << ReverseToken(CurTok) << "\n";
  if (!isascii(CurTok) || CurTok == '(' || CurTok == ',' || CurTok == '[' || CurTok == '{' || CurTok=='>' || CurTok==':')
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
  
  //std::cout << "Unary expr\n";
  
  getNextToken();
  if (auto Operand = ParseUnary(parser_struct, class_name, can_be_list))
  {    
    std::string operand_type = Operand->GetType();
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
  if (LHS->GetType()=="tensor")
    L_cuda = type_tensor;
  if (LHS->GetType()=="pinned_tensor")
    L_cuda = type_pinned_tensor;
  if (LHS->GetType()=="str")
    L_cuda = type_string;

  std::string L_type;
  std::string R_type;

  while (true)
  {
    L_type = LHS->GetType();

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


    if (RHS->GetType()=="tensor")
      R_cuda=type_tensor;
    if (RHS->GetType()=="pinned_tensor")
      R_cuda=type_pinned_tensor;
    if (RHS->GetType()=="object"||RHS->GetType()=="object_vec")
      R_cuda=type_object;
    if (RHS->GetType()=="str")
      R_cuda = type_string;
    
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


    //std::cout << "\nBinary expression of BinOp and Rhs:" << "\n";
    //std::cout << ReverseToken(BinOp) << " " << ReverseToken(RhsTok) << "\n";
    
    std::string op_elements = L_type + "_";
    op_elements = op_elements + R_type;

    // std::cout << "\n\n===============" << ".\n";
    // std::cout << "L type: " << L_type << " R type: " << R_type << "\n\n";
    // std::cout << "op type: " << op_elements << ".\n";

    if ((L_type=="list"||R_type=="list") && BinOp!='=')
    {
      LogError(parser_struct.line, "Tuple elements type are unknown during parsing type. Please load the element into a static type variable first.");
      return std::make_tuple(nullptr,0,"None");
    }

    bool shall_reverse_operands = false;
    if (reverse_ops.count(op_elements)>0)
    {
      op_elements = reverse_ops[op_elements];
      shall_reverse_operands = true;
    }

    std::string return_type = ops_type_return[op_elements];
    // std::cout << "return type: " << return_type << "...\n";


    // if (RHS->GetType()=="None")
    // {
    //   std::string pre = std::string("Binary Expr type is: ") + typeid(*RHS).name();
    //   std::cout << pre << ".\n";
    //   return std::make_tuple(nullptr,0,"float");
    // }






    if(shall_reverse_operands)
    {
      if (BinOp=='-') // inversion of 1 - tensor
      {
        // std::cout << "---REVERSING" << ".\n";

        std::string op_type = op_elements + "_mult";

        RHS = std::make_unique<BinaryExprAST>('*', op_elements, op_type,
                                std::move(RHS), std::move(std::make_unique<NumberExprAST>(-1.0f)), parser_struct);


        op_type = op_elements + "_add";
        LHS = std::make_unique<BinaryExprAST>('+', op_elements, op_type, std::move(RHS), std::move(LHS), parser_struct);

        // std::cout << "---Setting type as " << return_type << ".\n";
        LHS->SetType(return_type);

      } else {

        // std::cout << "Reverse 2" << ".\n";
                                                  
        std::string operation = op_map[BinOp];
        std::string op_type = op_elements + "_" + operation;

        LHS = std::make_unique<BinaryExprAST>(BinOp, op_elements, op_type, std::move(RHS), std::move(LHS), parser_struct);
        LHS->SetType(return_type);
        

      }
    } else {
      // std::cout << "No reverse" << ".\n";


      if (R_cuda==type_object)
      {
        LHS = std::make_unique<BinaryObjExprAST>(BinOp, std::move(LHS), std::move(RHS));
        LHS->SetType("object");
      }
      else
      {

        // std::cout << "Elements type: " << op_elements << ".\n";
        std::string operation = op_map[BinOp];
        std::string op_type = op_elements + "_" + operation;
        // std::cout << "op: " << BinOp << ".\n"; 
        
        
        LHS = std::make_unique<BinaryExprAST>(BinOp, op_elements, op_type, std::move(LHS), std::move(RHS), parser_struct);
        if (op_type=="int_int_div")
          return_type = "float";
        LHS->SetType(return_type);
      }
    }

    // std::string msg = "LHS type: " + LHS->GetType();
    // std::cout << msg << "\n";
    // std::cout << "====================================================================="  << ".\n";

    LhsTok = RhsTok;
  }
}


/// expression
///   ::= unary binoprhs
///
std::unique_ptr<ExprAST> ParseExpression(Parser_Struct parser_struct, std::string class_name, bool can_be_list) {
  parser_struct.line = LineCounter;
  
  //std::cout << "Parse Expression\n";
  
  auto LHS = ParseUnary(parser_struct, class_name, can_be_list);
  if (!LHS)
    return nullptr;

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

  if (CurTok!=tok_data) {
    LogErrorBreakLine(parser_struct.line, "Expected prototype function return type.");
    return nullptr;
  }

  return_type = IdentifierStr;
  

  
  // std::cout << "Token " << ReverseToken(CurTok) << ".\n";
  getNextToken(); // eat return_type
  // std::cout << "Token " << ReverseToken(CurTok) << ".\n"; 

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
    return LogErrorP(parser_struct.line, "Esperado '(' no protótipo");


  getNextToken();


  std::string type;
  std::vector<std::string> ArgNames, Types;

  


  Types.push_back("s");
  ArgNames.push_back("scope_struct");

  Function_Arg_Names[FnName].push_back("0");
  Function_Arg_Types[FnName]["0"] = "Scope_Struct";


  while (CurTok != ')')
  {
    if (IdentifierStr=="s"||IdentifierStr=="str")
      type="str";
    else if (IdentifierStr=="t"||IdentifierStr=="tensor")
      type="tensor";
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
      if (data_type=="list")
        data_type="unknown_list";

      getNextToken(); // eat arg type
      if(CurTok==tok_data&&IdentifierStr=="list")
      {
        data_type = data_type + "_list";
        getNextToken(); // eat list
      }

      Types.push_back(data_type);
      ArgNames.push_back(IdentifierStr);

      Function_Arg_Names[FnName].push_back(IdentifierStr);
      Function_Arg_Types[FnName][IdentifierStr] = data_type;


      typeVars[FnName][IdentifierStr] = data_type;

      if(in_str(data_type, Classes))
        Object_toClass[FnName][IdentifierStr] = data_type;

      
      getNextToken(); // eat arg name
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


  return std::make_unique<PrototypeAST>(FnName, return_type, _class, method, ArgNames, Types, Kind != 0,
                                         BinaryPrecedence);
}


std::unique_ptr<ExprAST> ParseImport(Parser_Struct parser_struct) {

  getNextToken(); // eat import

  if(CurTok!=tok_identifier&&CurTok!=tok_class_attr&&CurTok!=tok_post_class_attr_identifier)
    return LogError(parser_struct.line, "Expected library name after \"import\".");

  bool is_default = false;
  if(IdentifierStr=="default")
  {
    // std::cout << "Got a default import" << ".\n";
    is_default=true;
    getNextToken();
  }



  // Get lib name
  std::string lib_name = "";

  int dots=0; // dots before beggining the lib name.
  if(CurTok==tok_post_class_attr_identifier)
  {
    lib_name += IdentifierStr;
    getNextToken();
    dots++;
    IdentifierStr="";
    // std::cout << "get .lib" << ".\n";
    // LogBlue("Token is " + ReverseToken(CurTok));
  }

  while(CurTok==tok_class_attr)
  {
    lib_name += IdentifierStr + "/";
    getNextToken();
    // std::cout << "get tok_class_attr" << ".\n";
  }

  lib_name += IdentifierStr;

  std::string full_path_lib = tokenizer.current_dir+"/"+lib_name+".ai";



  // Import logic
  if(fs::exists(full_path_lib)||dots>0)
  {
    // std::cout << "READING AI LIB " << full_path_lib << ".\n";

    // std::cout << "Reverse: " << ReverseToken(CurTok) << ".\n";
    // std::cout << "cur_line: " << cur_line << ".\n";

    get_tok_util_space();
    tokenizer.importFile(full_path_lib, dots);

    return nullptr;
  }
  else
  {
    
    // getNextToken(); // moved to the LibImportExprAST constructor
    return std::make_unique<LibImportExprAST>(lib_name, is_default, parser_struct);
  }
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
    std::string _error = "Function " + class_name + "'s body was not declared.";  
    LogError(parser_struct.line, _error);
    return nullptr;
  } 

  return std::make_unique<FunctionAST>(std::move(Proto), std::move(Body));
  
}


/// toplevelexpr ::= expression
std::unique_ptr<FunctionAST> ParseTopLevelExpr(Parser_Struct parser_struct) {
  //std::cout << "Top Level Expression\n";

  
  std::vector<std::unique_ptr<ExprAST>> Body;
  while(!in_char(CurTok, terminal_tokens))
  {
    Body.push_back(std::move(ParseExpression(parser_struct)));
    //std::cout << "\n\nTop level expr cur tok: " << ReverseToken(CurTok) <<  ".\n";
    //std::cout << "Top level expr number of expressions: " << Body.size() <<  ".\n\n\n";
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

  getNextToken();

  if(CurTok==tok_space)
    getNextToken();
  

  int last_offset=0;
  std::vector<llvm::Type *> llvm_types;
  while(CurTok==tok_data||CurTok==tok_identifier)
  { 
    std::string data_type = IdentifierStr;

    bool is_object = in_str(data_type, Classes);

    getNextToken();
    while(true)
    {
      if (CurTok!=tok_identifier)
        LogError(parser_struct.line, "Class " + Name + " variables definition requires simple non-attribute names.");



      if (is_object) {
        typeVars[Name][IdentifierStr] = data_type;
        Object_toClass[Name][IdentifierStr] = data_type; 
      }
      else
       typeVars[Name][IdentifierStr] = data_type;
      
      ClassVariables[Name][IdentifierStr] = last_offset;
      
      if (data_type=="float")
      {
        llvm_types.push_back(Type::getFloatTy(*TheContext));
        last_offset+=4;
      }
      else if (data_type=="int")
      {
        llvm_types.push_back(Type::getInt32Ty(*TheContext));
        last_offset+=4;
      }
      else
      {
        llvm_types.push_back(int8PtrTy);
        last_offset+=8;
      }

      getNextToken();

      if (CurTok!=',')
        break;
      getNextToken();
    }
    if (CurTok==tok_space)
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
    return LogError(parser_struct.line, "A class definition requires it's functions. Got token: " + ReverseToken(CurTok));

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