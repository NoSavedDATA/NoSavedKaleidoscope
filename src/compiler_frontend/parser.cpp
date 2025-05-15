#include "llvm/IR/Value.h"



#include <iostream>
#include <map>
#include <random>
#include <thread>
#include <vector>



#include "../codegen/string.h"
#include "../common/include.h"
#include "include.h"




using namespace llvm;



std::map<std::string, std::string> ops_type_return;
std::map<int, std::string> op_map;

std::map<std::string, std::vector<std::string>> data_typeVars;
std::map<std::string, std::string> typeVars;


/// numberexpr ::= number
std::unique_ptr<ExprAST> ParseNumberExpr(Parser_Struct parser_struct) {
  auto Result = std::make_unique<NumberExprAST>(NumVal);
  getNextToken(); // consume the number
  return std::move(Result);
}

std::unique_ptr<ExprAST> ParseStringExpr(Parser_Struct parser_struct) {
  auto Result = std::make_unique<StringExprAST>(IdentifierStr);
  getNextToken(); // consume the "
  return std::move(Result);
}



/// parenexpr ::= '(' expression ')'
std::unique_ptr<ExprAST> ParseParenExpr(Parser_Struct parser_struct, std::string class_name) {

  
  getNextToken(); // eat (.

  auto V = ParseExpression(parser_struct, class_name);
  if (!V)
    return nullptr;

  if (CurTok != ')')
    return LogError("Expected ')' on parenthesis expression.");
  

  getNextToken(); // eat ).
  return V;
}



std::unique_ptr<ExprAST> ParseObjectInstantiationExpr(Parser_Struct parser_struct, std::string _class, std::string class_name) {
  getNextToken();
  //std::cout << "Object name: " << IdentifierStr << " and Class: " << Classes[i]<< "\n";
  bool is_vec=false;
  bool is_self=false;
  bool is_attr=false;
  std::string pre_dot;
  std::unique_ptr<ExprAST> VecInitSize = nullptr;
      
  //std::cout << "\n\n\n\nCUR TOK IS: " << ReverseToken(CurTok) << "\n\n\n\n\n\n";
  if (CurTok==tok_vec)
  {
    getNextToken();
    is_vec=true;
        
    if(CurTok=='[')
    {
      getNextToken();
      VecInitSize = ParsePrimary(parser_struct, class_name);
      if (CurTok!=']')
        LogError("Expected ] at object vec");
      getNextToken();
    }
    Object_toClassVec[IdentifierStr] = _class; //todo: this doesn't deal with self. expr
  }


  if (CurTok==tok_self)
  {
    getNextToken();
    is_self=true;
  }


  std::vector<std::pair<std::string, std::unique_ptr<ExprAST>>> VarNames;
  while (true) {
    std::string Name = IdentifierStr;
    objectVars.push_back(Name);
    if (!is_vec)
      Object_toClass[IdentifierStr] = _class;
    getNextToken(); // eat identifier.

        
    std::unique_ptr<ExprAST> Init = nullptr;  
    VarNames.push_back(std::make_pair(Name, std::move(Init)));

    // End of var list, exit loop.
    if (CurTok != ',')
      break;
    getNextToken(); // eat the ','.

    if (CurTok != tok_identifier)
      return LogError("Expected object identifier names.");
  }

  auto aux = std::make_unique<ObjectExprAST>(std::move(VarNames), "object", std::move(VecInitSize));
  aux->SetSelf(is_self);
  aux->SetIsAttribute(is_attr);
  aux->SetPreDot(pre_dot);
  aux->SetIsVec(is_vec);


  return aux;
}


std::vector<std::unique_ptr<ExprAST>> ParseIdx(Parser_Struct parser_struct, std::string class_name) {

  std::vector<std::unique_ptr<ExprAST>> Idx;
    
  Idx.push_back(ParseExpression(parser_struct, class_name, false));
  while(CurTok==',')
  {
    getNextToken(); // eat ,
    Idx.push_back(ParseExpression(parser_struct, class_name, false));
  }
  Idx.push_back(std::make_unique<NumberExprAST>(TERMINATE_VARARG));

  return Idx;
}



std::unique_ptr<ExprAST> ParseIdentifierListExpr(Parser_Struct parser_struct, std::string class_name, bool can_be_string, std::vector<std::tuple<std::string, int, std::vector<std::unique_ptr<ExprAST>>>> Names) {

  std::string type;
  std::vector<std::string> types;
  std::string IdName = IdentifierStr;

  std::vector<std::unique_ptr<ExprAST>> name_solvers;

  

  
  if (typeVars.find(IdName) != typeVars.end())
    type = typeVars[IdName];
  else if (in_str(IdName, objectVars))
    type = "object";

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

      if (typeVars.find(IdName) != typeVars.end())
        type = typeVars[IdName];
      else if (in_str(IdName, objectVars))
        type = "object";
      else
        type = "none";

      
      if (type!="none")
      {
        std::cout << "add variable of type " << type << ".\n";
        auto name_solver_expr = std::make_unique<NameSolverAST>(std::move(Names));
        aux = std::make_unique<VariableExprAST>(std::move(name_solver_expr), type, IdName, parser_struct);
      } else {
        if(!can_be_string)
        {
          std::string _error = "Variable " + IdName + " was not found in scope.";
          return LogError(_error);
        }  

        std::cout << "Returning ParseIdentifierExpr as a String Expression: " << IdName << "\n";
        aux = std::make_unique<StringExprAST>(IdName);
      }

      IdentifierList.push_back(std::move(aux));

      std::cout << "ADD " << IdName << " OF TYPE " << type << " TO LIST.\n";

      if (CurTok!=',')
        break;
      getNextToken(); // get comma
      getNextToken(); // get identifier
      Names.push_back(std::make_tuple(IdentifierStr, type_var, std::vector<std::unique_ptr<ExprAST>>{}));
    }


    std::cout << "RETURNTING VARIABLE LIST" << ".\n";
    aux = std::make_unique<VariableListExprAST>(std::move(IdentifierList));
    return std::move(aux);
  } 



  if (CurTok != '(' && CurTok != '[') // Simple variable ref.
  {
    if (typeVars.find(IdName) != typeVars.end())
      type = typeVars[IdName];
    else if (in_str(IdName, objectVars))
      type = "object";
    else
    {
      type = "none";
      // std::string _error = "Variable " + IdName + " not found.";
      // return LogError(_error);
    } 

    // std::cout << "Var type is: " << type << ".\n";

    if (type!="none")
    {
      auto name_solver_expr = std::make_unique<NameSolverAST>(std::move(Names));
      aux = std::make_unique<VariableExprAST>(std::move(name_solver_expr), type, IdName, parser_struct);
    } else {
      if(!can_be_string)
      {
        std::string _error = "Variable " + IdName + " was not found in scope.";
        return LogError(_error);
      }  

      std::cout << "Returning ParseIdentifierExpr as a String Expression: " << IdName << "\n";
      aux = std::make_unique<StringExprAST>(IdName);
    }
    
    

    return std::move(aux);
  }


  

  if (CurTok=='[')
  {
    getNextToken(); // eat [
    
    std::vector<std::unique_ptr<ExprAST>> Idx;
    Idx = ParseIdx(parser_struct, class_name);
    
    if (typeVars.find(IdName) != typeVars.end())
      type = typeVars[IdName];
    

    auto name_solver_expr = std::make_unique<NameSolverAST>(std::move(Names));
    aux = std::make_unique<VecIdxExprAST>(std::move(name_solver_expr), std::move(Idx), type);
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
          getNextToken(); // eat >
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
          return LogError("Expected ')' or ',' on argument list");
        getNextToken();
      }
    }

    // varargs
    if (in_str(IdName, vararg_methods))
      Args.push_back(std::make_unique<NumberExprAST>(TERMINATE_VARARG));
    
    

    // Eat the ')'.
    getNextToken();

    
    std::string callee_override = "none";
    bool name_solve_to_last = false;
    if(typeVars.count(IdName)>0)
    {
      name_solve_to_last = true;
      callee_override = typeVars[IdName];
    }
    

    bool is_var_forward = false;
    bool return_tensor = false;
    if (functionVars.find(IdName) != functionVars.end()) // if found
    {
      is_var_forward = true;
      return_tensor = true;
      name_solve_to_last = true;
      callee_override = functionVars[IdName];
    }
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
    LogError("Then is null");
    return nullptr;
  }
  
  
  if(CurTok == tok_space)
    getNextToken();

  //std::cout << "\n\nIf else token: " << ReverseToken(CurTok) <<  "\n\n\n";

  if (CurTok != tok_else) {
    Else.push_back(std::make_unique<NumberExprAST>(0));

    return std::make_unique<IfExprAST>(std::move(Cond), std::move(Then),
                                      std::move(Else));
  }
  else {
    getNextToken(); //eat else
    if(CurTok != tok_space)
      LogError("else requer barra de espaço.");
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
    return LogError("Expected ',' after for's control variable initial value.");
  getNextToken();



  auto End = ParseExpression(parser_struct, class_name);
  if (!End)
    return nullptr;

  


  std::unique_ptr<ExprAST> Step = std::make_unique<NumberExprAST>(1.0);
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
  getNextToken(); // eat "in".


  auto Vec = ParseExpression(parser_struct, class_name);
  if (!Vec)
    return nullptr;

  
  std::vector<std::unique_ptr<ExprAST>> Body;

  Body = ParseIdentedBodies(parser_struct, cur_level_tabs, class_name);

  return std::make_unique<ForEachExprAST>(IdName, std::move(Vec), std::move(Body), parser_struct);
}


/// forexpr ::= 'for' identifier '=' expr ',' expr (',' expr)? 'in' expression
std::unique_ptr<ExprAST> ParseForExpr(Parser_Struct parser_struct, std::string class_name) {

  int cur_level_tabs = SeenTabs;

  //std::cout << "\nSeen tabs on for: " << SeenTabs << "\n\n";

  getNextToken(); // eat the for.


  if (CurTok != tok_identifier)
    return LogError("Expected for's control variable identifier.");

  std::string IdName = IdentifierStr;
  getNextToken(); // eat identifier.

  typeVars[IdName] = "float";

  if (CurTok=='=')
    return ParseStandardForExpr(parser_struct, class_name, cur_level_tabs, IdName);
  else if(CurTok==tok_in)
    return ParseForEachExpr(parser_struct, class_name, cur_level_tabs, IdName);
  else
    return LogError("Expected for's control variable initial value.");
}



/// whileexpr ::= 'while' identifier '=' expr ',' expr (',' expr)? 'in' expression
std::unique_ptr<ExprAST> ParseWhileExpr(Parser_Struct parser_struct, std::string class_name) {

  int cur_level_tabs = SeenTabs;

  getNextToken(); // eat the while.


  //if (CurTok != tok_identifier)
  //  return LogError("Identificador da variável de controle esperado depois do while.");


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

  return std::make_unique<AsyncExprAST>(std::move(Bodies));
}

std::unique_ptr<ExprAST> ParseAsyncsExpr(Parser_Struct parser_struct, std::string class_name) {

  int cur_level_tabs = SeenTabs;

  getNextToken(); // eat the async.


  if (CurTok!=tok_number)
    LogError("asyncs expression expect the number of asynchrnonous functions.");

  int async_count = NumVal;
  getNextToken();
  std::cout << "Cur tok is " << ReverseToken(CurTok) << ".\n";
  
  std::vector<std::unique_ptr<ExprAST>> Bodies;
  
  //std::cout << "Pre expression token: " << ReverseToken(CurTok) << "\n";

  Bodies.push_back(std::make_unique<IncThreadIdExprAST>());
  if (CurTok != tok_space)
    Bodies.push_back(std::move(ParseExpression(parser_struct, class_name)));
  else
    Bodies = ParseIdentedBodies(parser_struct, cur_level_tabs, class_name);
  
  
  
  //std::cout << "Post async: " << ReverseToken(CurTok) << "\n";

  return std::make_unique<AsyncsExprAST>(std::move(Bodies), async_count);
}



std::unique_ptr<ExprAST> ParseFinishExpr(Parser_Struct parser_struct, std::string class_name) {

  int cur_level_tabs = SeenTabs;
  //std::cout << "Finish tabs level: " << cur_level_tabs <<  "\n";

  getNextToken(); // eat the finish.


  std::vector<std::unique_ptr<ExprAST>> Bodies;
  std::vector<bool> IsAsync;
  

  if (CurTok!=tok_space)
    LogError("Finish requires line break.");
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

  // getNextToken(); // [
  // std::vector<std::unique_ptr<ExprAST>> Elements = ParseIdx(class_name);
  // getNextToken(); // ]

  getNextToken(); // [
  std::vector<std::unique_ptr<ExprAST>> Elements;
  if (CurTok != ']') {
    while (true) {
      // std::cout << "CURRENT TOKEN: " << ReverseToken(CurTok) << ".\n";
      std::string element_type;
      if (CurTok==tok_number)
        element_type = "float";
      else
      {
        std::cout << "IDENTIFIER STR IS " << IdentifierStr << ".\n";
        if (typeVars.count(IdentifierStr)>0)
          element_type = typeVars[IdentifierStr];
        else
          LogError(IdentifierStr + " variable was not found on the Tuple definition scope.");
      }
      Elements.push_back(std::make_unique<StringExprAST>(element_type));

      if (auto element = ParseExpression(parser_struct, class_name, false))
      {
        Elements.push_back(std::move(element));
      } 
      else
        return nullptr;

      if (CurTok == ']')
        break;
      if (CurTok != ',')
      {
        LogError("Expected ']' or ',' on the Tuple elements list.");
      }
      getNextToken();
    }
  }   
  getNextToken(); // ]


  
  Elements.push_back(std::make_unique<StringExprAST>("TERMINATE_VARARG"));

  //TODO: vector for other types
  return std::make_unique<NewVecExprAST>(std::move(Elements), "tensor");
}





inline std::vector<std::unique_ptr<ExprAST>> Parse_Argument_List(Parser_Struct parser_struct, std::string class_name, std::string expression_name)
{
  std::vector<std::unique_ptr<ExprAST>> Args;
  if(CurTok!='(')
  {
    LogError("Expected ( afther the method name of the " + expression_name + " Expression.");
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
        LogError("Expected ')' or ',' on the Function Call arguments list.");
        return std::move(Args);
      }
      getNextToken();
    }
  } 
  
  // Eat the ')'.
  getNextToken();

  return std::move(Args);
}



std::unique_ptr<ExprAST> ParseSelfExpr(Parser_Struct parser_struct, std::string class_name) {

  std::string pre_dot = "";
  std::string type = "None";
  std::string object_class;
  bool is_class_attr=false;
  bool is_self=false;
  bool is_vec=false;
  std::vector<std::tuple<std::string, int, std::vector<std::unique_ptr<ExprAST>>>> Names;
  std::string Prev_IdName = "";
  std::string IdName = "";



  if (CurTok!=tok_self)
  {
    is_class_attr = true;
    pre_dot="";
  } else 
  {
    is_self=true;
    getNextToken(); // eat self
    Names.push_back(std::make_tuple("_", type_self, std::vector<std::unique_ptr<ExprAST>>{}));
  }


  
  int i=0;
  while (CurTok==tok_identifier||CurTok==tok_class_attr||CurTok==tok_post_class_attr_attr||CurTok==tok_post_class_attr_identifier)
  {
    int _type = type_attr;
    
    if (in_str(IdentifierStr, objectVars))
    {
      
      //std::cout << "\n\n" << IdentifierStr << " IS ON OBJECT VARS" <<  "\n\n";
      _type = type_object_name;
    }
      
    if (i==0&&CurTok==(tok_class_attr))
    {
      Prev_IdName = IdentifierStr;
      Names.push_back(std::make_tuple(IdentifierStr, _type, std::vector<std::unique_ptr<ExprAST>>{}));
      _type = type_attr;
    }
      
    if (i>0)
      is_class_attr = true;

    if (CurTok!=tok_identifier&&CurTok!=tok_post_class_attr_identifier)
    {
      Prev_IdName = IdentifierStr;
      object_class=IdentifierStr;
      pre_dot+=IdentifierStr;
    }

    is_vec=false;
    if (CurTok==tok_identifier||CurTok==tok_post_class_attr_identifier) // Need to handle vector
    {
      // std::cout << "OVERWRITE PREV NAME WITH " << IdName << ".\n";
      if(IdName!="")
        Prev_IdName = IdName;
      IdName = IdentifierStr;
    } 


    getNextToken(); // eat attr/identifier

      
    if (CurTok=='[')
    {
      //std::cout << "tokvec: " << ReverseToken(CurTok) << "\n";
      getNextToken(); // eat [
      std::vector<std::unique_ptr<ExprAST>> idx = ParseIdx(parser_struct, class_name);
      getNextToken(); // eat ]
      _type = type_vec;
      //int _type = (Object_toClassVec.count(IdName)>0) ? type_object_vec : type_vec;
      Names.push_back(std::make_tuple(IdName, _type, std::move(idx)));
      is_vec=true;
    } else
      Names.push_back(std::make_tuple(IdName, _type, std::vector<std::unique_ptr<ExprAST>>{}));

    //std::cout << "tok: " << ReverseToken(CurTok) << "\n";
    i+=1;
  }
    

  
  //std::cout << "Post tok: " << ReverseToken(CurTok) << "\n";
  
  // Turns string from object model of class type Model into Model
  if (Object_toClass.count(object_class)>0)
    object_class = Object_toClass[object_class]; 
  
  
  
  
  //std::cout << "\n\nParseSelfExpr of " << IdentifierStr << " HAS CLASS: " << class_name << " and pre-dot: " << pre_dot << "\n\n\n";
  

  if (!is_vec&&CurTok!='(') // Simple variable ref.
  { 
    if (typeVars.find(IdName) != typeVars.end())
      type = typeVars[IdName];
    else if (in_str(IdName, objectVars))
      type = "object";
    else if (functionVars.find(IdName) != functionVars.end())
      type = "tensor";
    else if (stringMethods.find(IdName) != stringMethods.end())
      type = "str";
    else {
      std::string _error = "Self/attribute variable " + IdName + " was not found in scope.";
      return LogError(_error);
      // type = "none";
    }

      

    std::cout << "Var type: " << type << "\n";

    auto name_solver_expr = std::make_unique<NameSolverAST>(std::move(Names));
    auto aux = std::make_unique<VariableExprAST>(std::move(name_solver_expr), type, IdName, parser_struct);

    aux->SetSelf(is_self);
    aux->SetIsAttribute(is_class_attr);
    aux->SetPreDot(pre_dot);
    
    

    return aux;
  }



  if (is_vec)
  {
    std::unique_ptr<ExprAST> aux;
    std::vector<std::unique_ptr<ExprAST>> Idx;
    
    
    
    if (typeVars.find(IdName) != typeVars.end())
      type = typeVars[IdName];
    if (in_str(IdName, objectVars))
      type = "object_vec";

    if(type=="object_vec")
      Idx = std::vector<std::unique_ptr<ExprAST>>{};
    else
      Idx = std::move(std::get<2>(Names[Names.size()-1]));

    auto name_solver_expr = std::make_unique<NameSolverAST>(std::move(Names));
    aux = std::make_unique<VecIdxExprAST>(std::move(name_solver_expr), std::move(Idx), type);
    aux->SetIsVec(true);


    aux->SetSelf(is_self);
    aux->SetIsAttribute(is_class_attr);
    aux->SetPreDot(pre_dot);


    return std::move(aux);
  }







  // PARSE CALL.

  getNextToken(); // eat (
  std::vector<std::unique_ptr<ExprAST>> Args;
  if (CurTok != ')') {
    while (true) {
      if (CurTok=='>')
      {
        getNextToken();
        auto inner_vec = ParseExpression(parser_struct, class_name, false);
        
        auto Arg = std::make_unique<SplitParallelExprAST>(std::move(inner_vec));
        Args.push_back(std::move(Arg));
      }
      else if (CurTok==':')
      {
        getNextToken(); // eat >
        auto inner_vec = ParseExpression(parser_struct, class_name, false);
        
        auto Arg = std::make_unique<SplitStridedParallelExprAST>(std::move(inner_vec));
        Args.push_back(std::move(Arg));
      }
      else if (auto Arg = ParseExpression(parser_struct, class_name, false))
      {
        //std::cout << "Parsed arg " << Arg->GetName() << "\n";
        Args.push_back(std::move(Arg));
      }
        
      else
        return nullptr;

      if (CurTok == ')')
        break;

      if (CurTok != ',')
        return LogError("Expected ')' or ',' on the Function Call arguments list.");
      getNextToken();
    }
  }

  
  
  

  // Eat the ')'.
  getNextToken();




  std::string callee_override = "none";
  std::string load_type = "none";
  std::string load_of = "none";
  bool name_solve_to_last = false;
  // x.view()
  if(typeVars.count(Prev_IdName)>0)
  {
    name_solve_to_last = false;
    callee_override = typeVars[Prev_IdName] + "_" + IdName;
    load_type = typeVars[Prev_IdName];
    std::cout << "Triggered from Prev_IdName, override as: " << callee_override << ".\n";
    load_of = Prev_IdName;
  }
  // model.linear_1(x)
  if(typeVars.count(IdName)>0)
  {
    std::cout << "Triggered from IdName" << ".\n";
    name_solve_to_last = true;
    callee_override = typeVars[IdName];
    load_of = IdName;
  }
  std::cout << "typeVars.count(Prev_IdName)>0: " << Prev_IdName << " is " <<  std::to_string(typeVars.count(Prev_IdName)>0) << ".\n";
  std::string callee = IdName;
  bool is_var_forward = false;
  bool return_tensor = false;
  bool return_string = false;
  // Override function calls: e.g: conv1 -> Conv2d
  if (functionVars.count(IdName) > 0)
  {
    is_var_forward = true;
    return_tensor = true;
    name_solve_to_last = true;
    callee_override = functionVars[IdName];

  } else if (floatFunctions.find(IdName) != floatFunctions.end())
  {
    is_var_forward = true;
    callee_override = floatFunctions[IdName];

  } else if (stringMethods.find(IdName) != stringMethods.end())
  {
    is_var_forward = true;
    return_string = true;
    callee_override = stringMethods[IdName];

  } else {
    if (is_self && !is_class_attr)
    {
      callee = class_name + IdName;
    }
  }


  std::string call_of = (callee_override!="none") ? callee_override : IdName;
  
  // varargs
  if (in_str(call_of, vararg_methods))
    Args.push_back(std::make_unique<NumberExprAST>(TERMINATE_VARARG));

  // std::cout << "\nCalling method: " << IdName << "/" << callee << " for pre-dot: " << pre_dot << "\n\n";
  // std::cout << "LOAD TYPE IS: " << load_type << ".\n";

  std::string scope_random_string = RandomString(14);

  auto name_solver_expr = std::make_unique<NameSolverAST>(std::move(Names));
  name_solver_expr->SetNameSolveToLast(name_solve_to_last);
  auto aux = std::make_unique<CallExprAST>(std::move(name_solver_expr), callee, IdName, std::move(Args),
                                        object_class, pre_dot, load_type, is_var_forward, callee_override, scope_random_string, load_of, parser_struct);


  if (functions_return_type.count(call_of)>0)
    aux->SetType(functions_return_type[call_of]);
  if (return_tensor)
    aux->SetType("tensor");  
  if (return_string)
    aux->SetType("str");


  if (is_self)
    aux->SetSelf(true);    
  if (is_class_attr)
    aux->SetIsAttribute(true);
  aux->SetPreDot(pre_dot);

  if (CurTok == tok_post_class_attr_identifier)  
    return ParseChainCallExpr(parser_struct, std::move(aux), class_name);

  return aux;
}


std::unique_ptr<ExprAST> ParseChainCallExpr(Parser_Struct parser_struct, std::unique_ptr<ExprAST> previous_call_expr, std::string class_name) {

  std::string IdName = IdentifierStr;
  getNextToken();



  std::vector<std::unique_ptr<ExprAST>> Args = Parse_Argument_List(parser_struct, class_name, "Chain Function Call");
    
  
  std::string type = previous_call_expr->Type;
  std::string call_of = type + "_" + IdName;
  
  // varargs
  if (in_str(call_of, vararg_methods))
    Args.push_back(std::make_unique<NumberExprAST>(TERMINATE_VARARG));
  
  
  

  auto aux = std::make_unique<ChainCallExprAST>(call_of, std::move(Args), std::move(previous_call_expr));
  if (functions_return_type.count(call_of)>0)
    aux->SetType(functions_return_type[call_of]);
  

  if (CurTok == tok_post_class_attr_identifier)  
    return ParseChainCallExpr(parser_struct, std::move(aux), class_name);


  return std::move(aux);
}








std::unique_ptr<ExprAST> ParseDataExpr(Parser_Struct parser_struct, std::string class_name) {

  // std::cout << "Parsing data with data type: " << IdentifierStr << ".\n";

  std::string data_type = IdentifierStr;

  getNextToken(); // eat data token.
  
  std::vector<std::pair<std::string, std::unique_ptr<ExprAST>>> VarNames;
  std::vector<std::unique_ptr<ExprAST>> notes;

  if (CurTok == '[')
  {
    getNextToken();
    
    //std::make_unique<NumberExprAST>(NumVal)
    
    while (true) {
      if (CurTok != tok_number && CurTok != tok_identifier && CurTok != tok_self)
        return LogError("Expected a number or var on the tensor dimension.");
      
      if (CurTok==tok_number)
      {
        if (std::fmod(NumVal, 1.0) != 0)
          LogWarning("A tensor's dimension should be int, not float.");
      
        notes.push_back(std::make_unique<NumberExprAST>( (float)((int)round(NumVal)) ));
        getNextToken();
      } else if (CurTok==tok_identifier)
        notes.push_back(std::move(ParseIdentifierExpr(parser_struct, class_name, true, false)));
      else {
        //notes.push_back(std::move(ParseExpression(parser_struct, class_name)));
        notes.push_back(std::move(ParsePrimary(parser_struct, class_name, false)));
      }

      
      if (CurTok != ',')
        break;
      getNextToken(); // eat the ','.
    }

    
    if (CurTok != ']')
      return LogError("] not found at tensor declaration.");
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
    return LogError("Expected tensor identifier name.");



  while (true) {
    std::string Name = IdentifierStr;

    typeVars[IdentifierStr] = data_type;
    getNextToken(); // eat identifier.

    

    // std::unique_ptr<ExprAST> Init = nullptr;
    // VarNames.push_back(std::make_pair(Name, std::move(Init))); 

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
      return LogError("Expected tensor identifier names.");
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
    pthread_mutex_t* _mutex = new pthread_mutex_t;
    if (pthread_mutex_init(_mutex, NULL) != 0) {
      printf("Mutex initialization failed\n");
      return nullptr;
    }
    
    lockVars[IdentifierStr] = _mutex;  
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
    expr = ParseSelfExpr(parser_struct, class_name);
  else if (CurTok==tok_identifier)
    expr = ParseIdentifierExpr(parser_struct, class_name, false, false);
  else
  {
    std::string _error = expr_name + " expression expected a simple identifier, not another expression.";
    LogError(_error);
  }

  return std::move(expr);
}



std::unique_ptr<ExprAST> ParseGlobalExpr(Parser_Struct parser_struct, std::string class_name) {
  getNextToken(); // eat global
  std::cout << "Parsing global expr\n";


  // At least one variable name is required.
  if (CurTok != tok_identifier)
    return LogError("Expected identifier after global.");

  while (true) {


    if (CurTok!=tok_identifier)
      return LogError("Global expression must contain identifiers only.");

    ParseIdentifierExpr(parser_struct, class_name);
    globalVars.push_back(IdentifierStr);

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

  std::vector<std::unique_ptr<ExprAST>> Vars, Destiny;

  std::unique_ptr<ExprAST> expr;
  

  if (CurTok != tok_identifier && CurTok != tok_class_attr && CurTok != tok_self && CurTok != tok_number)
    return LogError("Expected identifier after return.");

  
  while(true) {
    
    if (CurTok==tok_number)
    {
      expr = std::make_unique<NumberExprAST>(NumVal);
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
///   ::= varexpr
std::unique_ptr<ExprAST> ParsePrimary(Parser_Struct parser_struct, std::string class_name, bool can_be_list) {
  // std::cout << "ParsePrimary: " << ReverseToken(CurTok) << "can be list: " << can_be_list << ".\n";
  switch (CurTok) {
  default:
    //return std::move(std::make_unique<NumberExprAST>(0.0f));
    return LogErrorT(CurTok);
  case tok_identifier:
    return ParseIdentifierExpr(parser_struct, class_name, false, can_be_list);
  case tok_class_attr:
    return ParseSelfExpr(parser_struct, class_name);
  case tok_self:
    return ParseSelfExpr(parser_struct, class_name);
  case tok_number:
    return ParseNumberExpr(parser_struct);
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
    return ParseDataExpr(parser_struct, class_name);
  case tok_global:
    return ParseGlobalExpr(parser_struct);
  case '[':
    return ParseNewVector(parser_struct, class_name);
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
  if (!isascii(CurTok) || CurTok == '(' || CurTok == ',' || CurTok == '[' || CurTok=='>' || CurTok==':')
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
    auto expr = std::make_unique<UnaryExprAST>(Opc, std::move(Operand));
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
    

    if (TokPrec==BinopPrecedence[':'])
    {
      getNextToken();
      return std::make_tuple(std::move(LHS), L_cuda, L_type);
    }
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



    if(CurTok==':')
    {
      getNextToken();
      return std::make_tuple(std::move(LHS), L_cuda, L_type);
    }

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
      LogError("Tuple elements type are unknow during parsing type. Please load the element into a static type variable first.");
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
        // std::cout << "Operation: " << op_type << ".\n";
        // std::cout << "op: " << BinOp << ".\n";

        LHS = std::make_unique<BinaryExprAST>(BinOp, op_elements, op_type, std::move(LHS), std::move(RHS), parser_struct);
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
  std::string FnName = parser_struct.class_name;
  std::string _class, method;
  method = "";
  _class = parser_struct.class_name;



  std::string return_type;

  if (CurTok==tok_data) {
    return_type = IdentifierStr;
  }

  
  // std::cout << "Token " << ReverseToken(CurTok) << ".\n";
  getNextToken(); // eat return_type
  // std::cout << "Token " << ReverseToken(CurTok) << ".\n"; 

  unsigned Kind = 0; // 0 = identifier, 1 = unary, 2 = binary.
  unsigned BinaryPrecedence = 30;

  switch (CurTok) {
  default:
    return LogErrorP("Expected prototype function name");
  case tok_identifier:
    FnName += IdentifierStr;
    method = IdentifierStr;
    Kind = 0;
    getNextToken();
    break;
  case tok_unary:
    getNextToken();
    if (!isascii(CurTok))
      return LogErrorP("Esperado operador unário");
    FnName += "unary";
    FnName += (char)CurTok;
    Kind = 1;
    getNextToken();
    break;
  case tok_binary:
    getNextToken();
    if (!isascii(CurTok))
      return LogErrorP("Esperado operador binário");
    FnName += "binary";
    FnName += (char)CurTok;
    Kind = 2;
    getNextToken();

    // Read the precedence if present.
    if (CurTok == tok_number) {
      if (NumVal < 1 || NumVal > 100)
        return LogErrorP("Precedência inválida: deve ser entre 1 e 100");
      BinaryPrecedence = (unsigned)NumVal;
      getNextToken();
    }
    break;
  }

  if (CurTok != '(')
    return LogErrorP("Esperado '(' no protótipo");


  getNextToken();


  std::string type;
  std::vector<std::string> ArgNames, Types;



  Types.push_back("s");
  ArgNames.push_back("scope_struct");
 

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

    if (IdentifierStr!="s" && IdentifierStr!="t" && IdentifierStr!="f" && IdentifierStr!="i" && (!in_str(IdentifierStr, data_tokens)))
      LogErrorP_to_comma("Prototype var type must be t, f or a data type. Got " + IdentifierStr);
    else {
      Types.push_back(IdentifierStr);
      getNextToken(); // eat arg type

      ArgNames.push_back(IdentifierStr);

      typeVars[IdentifierStr] = type;
      
      getNextToken(); // eat arg name
    }
    


    if (CurTok == ')')
        break;
      
    if (CurTok != ',')
    {
      return LogErrorP("Expected ')' or ',' at prototype arguments list.");
    }
    getNextToken();
  }

  // std::cout << "CREATING PROTO WITH " << ArgNames.size() << " PARAMETERS.\n";

  // success.
  getNextToken(); // eat ')'.

  // Verify right number of names for operator.
  if (Kind && ArgNames.size() != Kind)
    return LogErrorP("Número inválido de operandos para o operador");

  if (CurTok!=tok_space)
    LogError("Post prototype parsing requires a line break.");
  getNextToken();

  functions_return_type[FnName] = return_type;

  return std::make_unique<PrototypeAST>(FnName, return_type, _class, method, ArgNames, Types, Kind != 0,
                                         BinaryPrecedence);
}



/// definition ::= 'def' prototype expression
std::unique_ptr<FunctionAST> ParseDefinition(Parser_Struct parser_struct, std::string class_name) {
  

  int cur_level_tabs = SeenTabs;

  getNextToken(); // eat def.


  auto Proto = ParsePrototype(parser_struct);
  if (!Proto)
  {
    std::string _error = "Error defining " + parser_struct.class_name + " prototype.";  
    LogError(_error);
    return nullptr;
  } 

  // std::cout << "PROTO NAME IS " << Proto->getName() << ".\n";
  parser_struct.function_name = Proto->getName();
  
  
  std::vector<std::unique_ptr<ExprAST>> Body;

  while(!in_char(CurTok, terminal_tokens))
  {
    if (SeenTabs <= cur_level_tabs && CurTok != tok_space)
      break;
      

    if (CurTok==tok_space)
      getNextToken();

    if (SeenTabs <= cur_level_tabs)
      break;

    Body.push_back(std::move(ParseExpression(parser_struct, class_name)));
  }

  //std::cout << "function number of expressions: " << Body.size() << "\n";

  if (Body.size()==0)
  {
    std::string _error = "Function " + class_name + "'s body was not declared.";  
    LogError(_error);
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



std::unique_ptr<ExprAST> ParseClass() {
  getNextToken(); // eat class.

  if (CurTok != tok_identifier)
    return LogError("Expected class name");
  std::string Name = IdentifierStr;

  Classes.push_back(Name);

  getNextToken();

  if(CurTok==tok_space)
    getNextToken();
  

  if (CurTok!=tok_def)
    return LogError("A class definition requires it's functions.");

  int i=0;

  Parser_Struct parser_struct;
  parser_struct.class_name = Name;

  while(CurTok==tok_def)
  {
    if(SeenTabs==0)
      break;
    
    auto Func = ParseDefinition(parser_struct, Name);
    if (!Func)
      return nullptr;
      //return LogError("Falha no parsing da função da Classe.");
    if (!ends_with(Func->getProto().getName(),"__init__") && i==0)
      return LogError("Class requires __init__ method");
    
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