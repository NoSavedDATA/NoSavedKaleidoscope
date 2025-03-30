#include "llvm/IR/Value.h"


#include <map>
#include <vector>
#include <thread>

#include <random>


#include "../common/include.h"
#include "include.h"

#include <iostream>



using namespace llvm;







/// numberexpr ::= number
std::unique_ptr<ExprAST> ParseNumberExpr() {
  auto Result = std::make_unique<NumberExprAST>(NumVal);
  getNextToken(); // consume the number
  return std::move(Result);
}

std::unique_ptr<ExprAST> ParseStringExpr() {
  auto Result = std::make_unique<StringExprAST>(IdentifierStr);
  getNextToken(); // consume the "
  return std::move(Result);
}



/// parenexpr ::= '(' expression ')'
std::unique_ptr<ExprAST> ParseParenExpr(std::string class_name) {
  getNextToken(); // eat (.
  auto V = ParseExpression(class_name);
  if (!V)
    return nullptr;

  if (CurTok != ')')
    return LogError("Expected ')' on parenthesis expression.");
  

  getNextToken(); // eat ).
  return V;
}



std::unique_ptr<ExprAST> ParseObjectInstantiationExpr(std::string _class, std::string class_name) {
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
      VecInitSize = ParsePrimary(class_name);
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

  if (CurTok==tok_space)
    getNextToken();

  return aux;
}


std::vector<std::unique_ptr<ExprAST>> ParseIdx(std::string class_name) {

  std::vector<std::unique_ptr<ExprAST>> Idx;
    
  Idx.push_back(ParseExpression(class_name));
  while(CurTok==',')
  {
    getNextToken(); // eat ,
    Idx.push_back(ParseExpression(class_name));
  }
  Idx.push_back(std::make_unique<NumberExprAST>(TERMINATE_VARARG));

  return Idx;
}



/// identifierexpr
///   ::= identifier
///   ::= identifier '(' expression* ')'
std::unique_ptr<ExprAST> ParseIdentifierExpr(std::string class_name) {
  
  for(int i=0; i<Classes.size(); i++)
    if(IdentifierStr==Classes[i])  // Object object
      return ParseObjectInstantiationExpr(Classes[i], class_name);

  std::vector<std::tuple<std::string, int, std::vector<std::unique_ptr<ExprAST>>>> Names;
  std::string IdName, type;
  IdName = IdentifierStr;
  //std::cout << "Identifier " << IdName <<  "\n";
  getNextToken(); // eat identifier.
  
  Names.push_back(std::make_tuple(IdName, type_var, std::vector<std::unique_ptr<ExprAST>>{}));
  if (CurTok != '(' && CurTok != '[') // Simple variable ref.
  {
    if (in_str(IdName, pinnedTensorVars))
      type = "pinned_tensor";
    else if (in_str(IdName, tensorVars))
      type = "tensor";
    else if (in_str(IdName, floatVars))
      type = "float";
    else if (in_str(IdName, strVars))
      type = "str";
    else if (in_str(IdName, objectVars))
      type = "object";
    else
      type = "none";

    std::unique_ptr<ExprAST> aux;
    if (type!="none")
    {
      auto name_solver_expr = std::make_unique<NameSolverAST>(std::move(Names));
      aux = std::make_unique<VariableExprAST>(std::move(name_solver_expr), type);
    } else {
      std::cout << "Returning ParseIdentifierExpr as a String Expression: " << IdName << "\n";
      aux = std::make_unique<StringExprAST>(IdName);
    }
    
    
    if (CurTok==tok_space)
      getNextToken();

    return std::move(aux);
  }



  if (CurTok=='[')
  {
    getNextToken(); // eat [
    
    std::vector<std::unique_ptr<ExprAST>> Idx;
    Idx = ParseIdx(class_name);
    
    if (in_str(IdName, str_vecVars))
      type = "str_vec";
    if (in_str(IdName, float_vecVars))
      type = "float_vec";
    if (in_str(IdName, tensorVars))
      type = "tensor";

    auto name_solver_expr = std::make_unique<NameSolverAST>(std::move(Names));
    auto aux = std::make_unique<VecIdxExprAST>(std::move(name_solver_expr), std::move(Idx), type);
    aux->SetIsVec(true);
    
    getNextToken(); // eat ]
    
    return std::move(aux);
  }
  

  // Call.
  getNextToken(); // eat (
  
  std::vector<std::unique_ptr<ExprAST>> Args;
  if (CurTok != ')') {
    while (true) {
      
      if (auto Arg = ParseExpression(class_name))
        Args.push_back(std::move(Arg));
      else
        return nullptr;
      

      if (CurTok == ')')
        break;

      if (CurTok != ',')
        return LogError("Esperado ')' ou ',' na lista de argumentos");
      getNextToken();
    }
  }

  // varargs
  if (in_str(IdName, vararg_methods))
    Args.push_back(std::make_unique<NumberExprAST>(TERMINATE_VARARG));
  
  

  // Eat the ')'.
  getNextToken();

  bool is_var_forward = false;
  bool return_tensor = false;
  std::string callee_override = "none";
  if (functionVars.find(IdName) != functionVars.end()) // if found
  {
    is_var_forward = true;
    return_tensor = true;
    callee_override = functionVars[IdName];
  }
  if (floatFunctions.find(IdName) != floatFunctions.end()) // if found
  {
    is_var_forward = true;
    callee_override = floatFunctions[IdName];
  }
  if (IdName=="to_float")
  {
    callee_override = "ToFloat";
    is_var_forward = true;
  }
  
  auto name_solver_expr = std::make_unique<NameSolverAST>(std::move(Names));
  name_solver_expr->SetNameSolveToLast(false);
  auto aux = std::make_unique<CallExprAST>(std::move(name_solver_expr), IdName, std::move(Args),
                                                "None", "None", is_var_forward, callee_override);

  
  
  if (in_str(IdName, return_tensor_fn) || return_tensor)
    aux->SetType("tensor");
  if (in_str(IdName, return_pinned_methods))
    aux->SetType("pinned_tensor");
  if (in_str(IdName, return_string_fn))
    aux->SetType("str");

  return aux;
}





/// ifexpr ::= 'if' expression 'then' expression 'else' expression
std::unique_ptr<ExprAST> ParseIfExpr(std::string class_name) {
  
  int cur_level_tabs = SeenTabs;

  getNextToken(); // eat the if.

  
  //std::cout << "If tabs level: " << cur_level_tabs <<  "\n";
  

  // condition.
  auto Cond = ParseExpression(class_name);
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
    
    auto body = ParseExpression(class_name);
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
      
      auto body = ParseExpression(class_name);
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




std::vector<std::unique_ptr<ExprAST>> ParseIdentedBodies(int cur_level_tabs, std::string class_name)
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


    auto body = ParseExpression(class_name);
    if (!body)
      return std::move(NullBody);
    Body.push_back(std::move(body));
    //getNextToken();
  }

  if (CurTok==tok_space)
    getNextToken();

  return std::move(Body);
}


/// forexpr ::= 'for' identifier '=' expr ',' expr (',' expr)? 'in' expression
std::unique_ptr<ExprAST> ParseForExpr(std::string class_name) {

  int cur_level_tabs = SeenTabs;

  //std::cout << "\nSeen tabs on for: " << SeenTabs << "\n\n";

  getNextToken(); // eat the for.


  if (CurTok != tok_identifier)
    return LogError("Expected for's control variable identifier.");

  std::string IdName = IdentifierStr;
  getNextToken(); // eat identifier.

  floatVars.push_back(IdName);

  if (CurTok != '=')
    return LogError("Expected for's control variable initial value.");
  getNextToken(); // eat '='.

  auto Start = ParseExpression(class_name);
  if (!Start)
    return nullptr;
  if (CurTok != ',')
    return LogError("Expected ',' after for's control variable initial value.");
  getNextToken();



  auto End = ParseExpression(class_name);
  if (!End)
    return nullptr;

  


  std::unique_ptr<ExprAST> Step = std::make_unique<NumberExprAST>(1.0);
  if (CurTok == ',') { // The step value is optional.
    getNextToken();
    auto aux = ParseExpression(class_name);
    if (aux)
      Step = std::move(aux);
  }
  
  std::vector<std::unique_ptr<ExprAST>> Body;

  Body = ParseIdentedBodies(cur_level_tabs, class_name);

  return std::make_unique<ForExprAST>(IdName, std::move(Start), std::move(End),
                                       std::move(Step), std::move(Body));
}



/// whileexpr ::= 'while' identifier '=' expr ',' expr (',' expr)? 'in' expression
std::unique_ptr<ExprAST> ParseWhileExpr(std::string class_name) {

  int cur_level_tabs = SeenTabs;

  getNextToken(); // eat the while.


  //if (CurTok != tok_identifier)
  //  return LogError("Identificador da variável de controle esperado depois do while.");


  auto Cond = ParseExpression(class_name);
  if (!Cond)
    return nullptr;
  
  std::vector<std::unique_ptr<ExprAST>> Body = ParseIdentedBodies(cur_level_tabs, class_name);

  return std::make_unique<WhileExprAST>(std::move(Cond), std::move(Body));
}



std::unique_ptr<ExprAST> ParseAsyncExpr(std::string class_name) {

  int cur_level_tabs = SeenTabs;

  getNextToken(); // eat the async.

  
  std::vector<std::unique_ptr<ExprAST>> Bodies;
  
  //std::cout << "Pre expression token: " << ReverseToken(CurTok) << "\n";

  if (CurTok != tok_space)
    Bodies.push_back(std::move(ParseExpression(class_name)));
  else
    Bodies = ParseIdentedBodies(cur_level_tabs, class_name);
  
  
  
  //std::cout << "Post async: " << ReverseToken(CurTok) << "\n";

  return std::make_unique<AsyncExprAST>(std::move(Bodies));
}



std::unique_ptr<ExprAST> ParseFinishExpr(std::string class_name) {

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
      Bodies.push_back(std::move(ParseAsyncExpr(class_name)));
      IsAsync.push_back(true);
    }
    else
    {
      Bodies.push_back(std::move(ParseExpression(class_name)));
      IsAsync.push_back(false);
    }
    */
    Bodies.push_back(std::move(ParseExpression(class_name)));
    IsAsync.push_back(false);
  }


  return std::make_unique<FinishExprAST>(std::move(Bodies),
                                         std::move(IsAsync));
}


/// varexpr ::= 'var' identifier ('=' expression)?
//                    (',' identifier ('=' expression)?)* 'in' expression
std::unique_ptr<ExprAST> ParseVarExpr(std::string class_name) {
  getNextToken(); // eat the var.
  

  // mem2reg is alloca-driven: it looks for allocas and if it can handle them, it promotes them. It DOES NOT APPLY TO GLOBAL variables or heap allocations.
  // mem2reg only promotes allocas whose uses are direct loads and stores. If the address of the stack object is passed to a function,
  //or if any funny pointer arithmetic is involved, the alloca will not be promoted.

  std::vector<std::pair<std::string, std::unique_ptr<ExprAST>>> VarNames;

  // At least one variable name is required.
  if (CurTok != tok_identifier)
    return LogError("Expected an identifier after float.");

  while (true) {
    std::string Name = IdentifierStr;
    floatVars.push_back(IdentifierStr);
    getNextToken(); // eat identifier.

    // Read the optional initializer.
    std::unique_ptr<ExprAST> Init = nullptr;
    if (CurTok == '=')
    {
      getNextToken(); // eat the '='.

      Init = ParseExpression(class_name);
      if (!Init)
        return nullptr;
    }

    VarNames.push_back(std::make_pair(Name, std::move(Init)));

    // End of var list, exit loop.
    if (CurTok != ',')
      break;
    getNextToken(); // eat the ','.

    if (CurTok != tok_identifier)
      return LogError("Expected one or more identifiers after float.");
  }

  if (CurTok==tok_space)
    getNextToken();


  return std::make_unique<VarExprAST>(std::move(VarNames), "var");
}




std::unique_ptr<ExprAST> ParseStrExpr() {
  getNextToken(); // eat str
  

  std::vector<std::pair<std::string, std::unique_ptr<ExprAST>>> VarNames;

  // At least one variable name is required.
  if (CurTok != tok_identifier)
    return LogError("Esperado identificador após var.");

  
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

  while (true) {
    std::string Name = IdentifierStr;
    strVars.push_back(Name);
    getNextToken(); // eat identifier.

    // Read the optional initializer.
    std::unique_ptr<ExprAST> Init = nullptr;
    if (CurTok == '=') {
      getNextToken(); // eat the '='.

      Init = ParseStringExpr();
      if (!Init)
        return nullptr;
    }

    VarNames.push_back(std::make_pair(Name, std::move(Init)));

    // End of var list, exit loop.
    if (CurTok != ',')
      break;
    getNextToken(); // eat the ','.

    if (CurTok != tok_identifier)
      return LogError("Esperado um ou mais identificadores após var.");
  }

  auto aux = std::make_unique<StrExprAST>(std::move(VarNames));

  aux->SetSelf(is_self);
  aux->SetIsAttribute(is_attr);
  aux->SetPreDot(pre_dot);
  

  if (CurTok==tok_space)
    getNextToken();


  return aux;
}

std::unique_ptr<ExprAST> ParseNewVector(std::string class_name) {
  std::cout << "Parsing new vector" << ReverseToken(CurTok)  << "\n";
  getNextToken(); // [
  std::vector<std::unique_ptr<ExprAST>> values = ParseIdx(class_name);
  getNextToken(); // ]


  if (CurTok==tok_space)
    getNextToken();
  
  values.push_back(std::make_unique<NumberExprAST>(TERMINATE_VARARG));

  //TODO: vector for other types
  return std::make_unique<NewVecExprAST>(std::move(values), "tensor");
}


std::unique_ptr<ExprAST> ParseStrVecExpr() {
  int vec_type = CurTok;
  std::string vec_type_str;

  if (vec_type==tok_str_vec)
    vec_type_str = "str";
  if (vec_type==tok_float_vec)
    vec_type_str = "float";
    
  getNextToken(); // eat str_vec

  
  
  
  std::vector<std::pair<std::string, std::unique_ptr<ExprAST>>> VarNames;

  // At least one variable name is required.
  if (CurTok != tok_identifier)
    return LogError("Expected identifier after vector var.");

  

  while (true) {
    std::string Name = IdentifierStr;
    getNextToken(); // eat identifier.

    // Read the optional initializer.
    std::unique_ptr<ExprAST> Init = nullptr;
    if (CurTok == '=') {
      getNextToken(); // eat the '='.

      Init = ParseStringExpr();
      if (!Init)
        return nullptr;
    }

    VarNames.push_back(std::make_pair(Name, std::move(Init)));

    if (vec_type==tok_str_vec)
      str_vecVars.push_back(Name);
    if (vec_type==tok_float_vec)
      float_vecVars.push_back(Name);
    

    // End of var list, exit loop.
    if (CurTok != ',')
      break;
    getNextToken(); // eat the ','.

    if (CurTok != tok_identifier)
      return LogError("Esperado um ou mais identificadores após var.");
  }

  if (CurTok==tok_space)
    getNextToken();

  return std::make_unique<StrVecExprAST>(std::move(VarNames), vec_type_str);
}



std::unique_ptr<ExprAST> ParseSelfExpr(std::string class_name) {

  std::string pre_dot = "";
  std::string type = "None";
  std::string object_class;
  bool is_class_attr=false;
  bool is_self=false;
  bool is_vec=false;
  std::vector<std::tuple<std::string, int, std::vector<std::unique_ptr<ExprAST>>>> Names;
  std::string IdName;



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
      Names.push_back(std::make_tuple(IdentifierStr, _type, std::vector<std::unique_ptr<ExprAST>>{}));
      _type = type_attr;
    }
      
    if (i>0)
      is_class_attr = true;

    if (CurTok!=tok_identifier&&CurTok!=tok_post_class_attr_identifier)
    {
      object_class=IdentifierStr;
      pre_dot+=IdentifierStr;
    }

    is_vec=false;
    if (CurTok==tok_identifier||CurTok==tok_post_class_attr_identifier) // Need to handle vector
      IdName = IdentifierStr;

    //std::cout << "\n\ntok pre vec: " << ReverseToken(CurTok) << "\n";

    getNextToken(); // eat attr/identifier

      
    if (CurTok=='[')
    {
      //std::cout << "tokvec: " << ReverseToken(CurTok) << "\n";
      getNextToken(); // eat [
      std::vector<std::unique_ptr<ExprAST>> idx = ParseIdx(class_name);
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
    //std::cout << "Parsing a var" << "\n";
    
    if (in_str(IdName, pinnedTensorVars))
      type = "pinned_tensor";
    if (in_str(IdName, tensorVars))
      type = "tensor";
    if (in_str(IdName, objectVars))
      type = "object";
    if (functionVars.find(IdName) != functionVars.end())
      type = "tensor";
    if (stringMethods.find(IdName) != stringMethods.end())
      type = "str";
    if (in_str(IdName, floatVars))
      type = "float";
    if (in_str(IdName, float_vecVars))
      type = "float_vec";
    if (in_str(IdName, strVars))
      type = "str";
    if (in_str(IdName, str_vecVars))
      type = "str_vec";

    std::cout << "Var type: " << type << "\n";

    auto name_solver_expr = std::make_unique<NameSolverAST>(std::move(Names));
    auto aux = std::make_unique<VariableExprAST>(std::move(name_solver_expr), type);

    aux->SetSelf(is_self);
    aux->SetIsAttribute(is_class_attr);
    aux->SetPreDot(pre_dot);
    
    
    if (CurTok==tok_space)
      getNextToken();

    return aux;
  }



  if (is_vec)
  {
    std::unique_ptr<ExprAST> aux;
    std::vector<std::unique_ptr<ExprAST>> Idx;
    
    
    
    if (in_str(IdName, str_vecVars))
      type= "str_vec";
    if (in_str(IdName, float_vecVars))
      type = "float_vec";
    if (in_str(IdName, pinnedTensorVars))
      type = "pinned_tensor";
    if (in_str(IdName, tensorVars))
      type= "tensor";
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
      if (auto Arg = ParseExpression(class_name))
      {
        //std::cout << "Parsed arg " << Arg->GetName() << "\n";
        Args.push_back(std::move(Arg));
      }
        
      else
        return nullptr;

      if (CurTok == ')')
        break;

      if (CurTok != ',')
        return LogError("Esperado ')' ou ',' na lista de argumentos");
      getNextToken();
    }
  }

  
  // varargs
  if (in_str(IdName, vararg_methods))
    Args.push_back(std::make_unique<NumberExprAST>(TERMINATE_VARARG));
  
  

  // Eat the ')'.
  getNextToken();




  // Override function calls: e.g: conv1 -> Conv2d
  bool is_var_forward = false;
  bool return_tensor = false;
  bool return_string = false;
  std::string callee_override = "none";
  if (functionVars.count(IdName) > 0)
  {
    is_var_forward = true;
    return_tensor = true;
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
      IdName = class_name + IdName;
  }

  
  //std::cout << "\nCalling method: " << IdName << " for pre-dot: " << pre_dot << "\n\n";

  //if (IdName == "len")


  auto name_solver_expr = std::make_unique<NameSolverAST>(std::move(Names));
  name_solver_expr->SetNameSolveToLast(false);
  auto aux = std::make_unique<CallExprAST>(std::move(name_solver_expr), IdName, std::move(Args),
                                        object_class, pre_dot, is_var_forward, callee_override);

  if (in_str(IdName, return_tensor_fn) || return_tensor)
    aux->SetType("tensor");
  
  if (return_string||in_str(IdName, return_string_fn))
    aux->SetType("str");

  if (CurTok==tok_space)
    getNextToken();

  if (is_self)
    aux->SetSelf(true);    
  if (is_class_attr)
    aux->SetIsAttribute(true);
  aux->SetPreDot(pre_dot);
  
  return aux;
}


//
std::unique_ptr<ExprAST> ParsePinnedTensorExpr() {
  
  getNextToken(); // eat pinned_tensor.
  
  if (CurTok != '[')
    return LogError("pinned tensor declaration expected [");
    getNextToken();

  std::vector<std::unique_ptr<ExprAST>> dims;
  std::vector<std::pair<std::string, std::unique_ptr<ExprAST>>> VarNames;
  std::string init = "zeros";
  //std::make_unique<NumberExprAST>(NumVal)
  
  while (true) {
    if (CurTok != tok_number && CurTok != tok_identifier && CurTok != tok_self)
    {
      std::cout << "Got token: " << ReverseToken(CurTok) << "\n";
      return LogError("Expected a number or var on the tensor dimension.");
    }
    


    if (CurTok==tok_number)
    {
      if (std::fmod(NumVal, 1.0) != 0)
        LogWarning("A tensor's dimension should be int, not float.");
    
      dims.push_back(std::make_unique<NumberExprAST>( (float)((int)round(NumVal)) ));
      getNextToken();
    } else if (CurTok==tok_identifier)
      if (in_str(IdentifierStr, tensor_inits))
      {
        init = IdentifierStr;
        getNextToken();
      } else
        dims.push_back(std::move(ParseIdentifierExpr()));
    else {
      dims.push_back(std::move(ParseSelfExpr()));
    }

    
    if (CurTok != ',')
      break;
    getNextToken(); // eat the ','.
  }

  
  if (CurTok != ']')
    return LogError("] not found.");
    getNextToken();


  std::string pre_dot="";
  bool is_self = false;
  bool is_attr = false;
  if (CurTok == tok_self)
  {
    is_self=true;
    getNextToken();
  }
  if (CurTok == tok_class_attr)
  {
    is_attr=true;
    pre_dot = IdentifierStr;
    std::cout << "Obj attr pinned_tensor: " << pre_dot << ".\n";
    getNextToken();
  }

  if (CurTok != tok_identifier)
    return LogError("Expected pinned tensor identifier name.");

  while (true) {
    std::string Name = IdentifierStr;
    pinnedTensorVars.push_back(IdentifierStr);
    getNextToken(); // eat identifier.

    
    std::unique_ptr<ExprAST> Init = nullptr;
    VarNames.push_back(std::make_pair(Name, std::move(Init)));

    // End of var list, exit loop.
    if (CurTok != ',')
      break;
    getNextToken(); // eat the ','.

    if (CurTok != tok_identifier)
      return LogError("Expected pinned tensor identifier names.");
  }



  auto aux = std::make_unique<PinnedTensorExprAST>(std::move(VarNames), "pinned_tensor",
                                             std::move(dims), init);
  aux->SetSelf(is_self);
  aux->SetIsAttribute(is_attr);
  aux->SetPreDot(pre_dot);

  
  if (CurTok==tok_space)
    getNextToken();
  
  return aux;
}










std::unique_ptr<ExprAST> ParseDataExpr(std::string class_name) {
  // bool is_weight;
  // if (CurTok==tok_tensor)
  //   is_weight=false;
  // if (CurTok==tok_param)
  //   is_weight=true;

  std::cout << "Parsing data with data type: " << IdentifierStr << ".\n";

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
      {
        std::cout << "Notes with tok identifier\n";
        notes.push_back(std::move(ParseIdentifierExpr()));
        std::cout << "Push back\n";
      }
      else {
        //notes.push_back(std::move(ParseExpression(class_name)));
        notes.push_back(std::move(ParsePrimary(class_name)));
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
    tensorVars.push_back(IdentifierStr);
    getNextToken(); // eat identifier.

    
    std::unique_ptr<ExprAST> Init = nullptr;
    VarNames.push_back(std::make_pair(Name, std::move(Init))); 

    // End of var list, exit loop.
    if (CurTok != ',')
      break;
    getNextToken(); // eat the ','.

    if (CurTok != tok_identifier)
      return LogError("Expected tensor identifier names.");
  }



  auto aux = std::make_unique<DataExprAST>(std::move(VarNames), data_type,
                                             std::move(notes));
  aux->SetSelf(is_self);
  aux->SetIsAttribute(is_attr);
  aux->SetPreDot(pre_dot);
  
  if (CurTok==tok_space)
    getNextToken();
  
  return aux;
}



//
std::unique_ptr<ExprAST> ParseConv2dExpr() {
  
  getNextToken(); // eat the Conv2d.
  
  if (CurTok != '[')
    return LogError("Conv2d declaration expected [");
    getNextToken();

  std::vector<std::unique_ptr<ExprAST>> dims;
  std::vector<std::pair<std::string, std::unique_ptr<ExprAST>>> VarNames;
  std::string init = "xavu_relu";
  //std::make_unique<NumberExprAST>(NumVal)
  
  while (true) {
    if (CurTok != tok_number && CurTok != tok_identifier && CurTok != tok_self)
      return LogError("Expected tensor dimension number.");
    
    if (CurTok==tok_number)
    {
      if (std::fmod(NumVal, 1.0) != 0)
        LogWarning("Tensor dimensions must be of type int. They are not supposed to be float.");
    
      dims.push_back(std::make_unique<NumberExprAST>( (float)((int)round(NumVal)) ));
      getNextToken();
    } else if (CurTok==tok_identifier)
      if (in_str(IdentifierStr, tensor_inits))
      {
        init = IdentifierStr;
        getNextToken();
      } else
        dims.push_back(std::move(ParseIdentifierExpr()));
    else {
      dims.push_back(std::move(ParseSelfExpr()));
    }

    
    if (CurTok != ',')
      break;
    getNextToken(); // eat the ','.
  }

  

  if (CurTok != ']')
    return LogError("Expected ].");
    getNextToken();

  if (dims.size()<5)
    return LogError("Convolution declaration requires input and output channels, kernel size, stride and padding.");


  std::string pre_dot="";
  bool is_self = false;
  bool is_attr = false;
  if (CurTok == tok_self)
  {
    is_self=true;
    getNextToken();
  }
  if (CurTok == tok_class_attr)
  {
    is_attr=true;
    pre_dot = IdentifierStr;
    std::cout << "Obj attr pinned_tensor: " << pre_dot << ".\n";
    getNextToken();
  }

  if (CurTok != tok_identifier)
    return LogError("Expected conv identifier name.");



  while (true) {
    std::string Name = IdentifierStr;
    
    getNextToken(); // eat identifier.

    
    std::unique_ptr<ExprAST> Init = nullptr;
    VarNames.push_back(std::make_pair(Name, std::move(Init)));
    functionVars[Name] = "ConvForward2d";

    // End of var list, exit loop.
    if (CurTok != ',')
      break;
    getNextToken(); // eat the ','.

    if (CurTok != tok_identifier)
      return LogError("Esperado um ou mais identificadores após var.");
  }



  auto aux = std::make_unique<Conv2dExprAST>(std::move(VarNames), "conv2d",
                                             std::move(dims[0]), std::move(dims[1]), std::move(dims[2]),
                                             std::move(dims[3]), std::move(dims[4]),
                                             init);
  aux->SetSelf(is_self);
  aux->SetIsAttribute(is_attr);
  aux->SetPreDot(pre_dot);

  
  if (CurTok==tok_space)
    getNextToken();
  
  return aux;
}



//
std::unique_ptr<ExprAST> ParseLSTMExpr() {
  
  getNextToken(); // eat the LSTM.
  
  if (CurTok != '[')
    return LogError("LSTM declaration expected [");
    getNextToken();

  std::vector<std::unique_ptr<ExprAST>> dims;
  std::vector<std::pair<std::string, std::unique_ptr<ExprAST>>> VarNames;
  std::string init = "xavu_relu";
  //std::make_unique<NumberExprAST>(NumVal)
  
  while (true) {
    if (CurTok != tok_number && CurTok != tok_identifier && CurTok != tok_self)
      return LogError("Expected tensor dimension number.");
    
    if (CurTok==tok_number)
    {
      if (std::fmod(NumVal, 1.0) != 0)
        LogWarning("Tensor dimensions must be of type int. They are not supposed to be float.");
    
      dims.push_back(std::make_unique<NumberExprAST>( (float)((int)round(NumVal)) ));
      getNextToken();
    } else if (CurTok==tok_identifier)
      if (in_str(IdentifierStr, tensor_inits))
      {
        init = IdentifierStr;
        getNextToken();
      } else
        dims.push_back(std::move(ParseIdentifierExpr()));
    else {
      dims.push_back(std::move(ParseSelfExpr()));
    }

    
    if (CurTok != ',')
      break;
    getNextToken(); // eat the ','.
  }

  

  if (CurTok != ']')
    return LogError("Expected ].");
    getNextToken();

  if (dims.size()!=2 && dims.size()!=3)
    return LogError("LSTM requires input and output hiddens count.");


  std::string pre_dot="";
  bool is_self = false;
  bool is_attr = false;
  if (CurTok == tok_self)
  {
    is_self=true;
    getNextToken();
  }
  if (CurTok == tok_class_attr)
  {
    is_attr=true;
    pre_dot = IdentifierStr;
    std::cout << "Obj attr pinned_tensor: " << pre_dot << ".\n";
    getNextToken();
  }

  if (CurTok != tok_identifier)
    return LogError("Expected LSTM identifier name.");



  while (true) {
    std::string Name = IdentifierStr;
    
    getNextToken(); // eat identifier.

    
    std::unique_ptr<ExprAST> Init = nullptr;
    VarNames.push_back(std::make_pair(Name, std::move(Init)));
    functionVars[Name] = "LSTMForward";

    // End of var list, exit loop.
    if (CurTok != ',')
      break;
    getNextToken(); // eat the ','.

    if (CurTok != tok_identifier)
      return LogError("Expected one or more identifiers after LSTM.");
  }



  auto aux = std::make_unique<LSTMExprAST>(std::move(VarNames), "lstm",
                                             std::move(dims[0]), std::move(dims[1]),
                                             init);
  aux->SetSelf(is_self);
  aux->SetIsAttribute(is_attr);
  aux->SetPreDot(pre_dot);

  
  if (CurTok==tok_space)
    getNextToken();
  
  return aux;
}


std::unique_ptr<ExprAST> ParseEmbeddingExpr() {
  
  getNextToken(); // eat the Embedding.
  
  if (CurTok != '[')
    return LogError("Embedding declaration expected [");
    getNextToken();

  std::vector<std::unique_ptr<ExprAST>> dims;
  std::vector<std::pair<std::string, std::unique_ptr<ExprAST>>> VarNames;
  std::string init = "xavu_relu";
  //std::make_unique<NumberExprAST>(NumVal)
  
  while (true) {
    if (CurTok != tok_number && CurTok != tok_identifier && CurTok != tok_self)
      return LogError("Expected tensor dimension number.");
    
    if (CurTok==tok_number)
    {
      if (std::fmod(NumVal, 1.0) != 0)
        LogWarning("Tensor dimensions must be of type int. They are not supposed to be float.");
    
      dims.push_back(std::make_unique<NumberExprAST>( (float)((int)round(NumVal)) ));
      getNextToken();
    } else if (CurTok==tok_identifier)
      if (in_str(IdentifierStr, tensor_inits))
      {
        init = IdentifierStr;
        getNextToken();
      } else
        dims.push_back(std::move(ParseIdentifierExpr()));
    else {
      dims.push_back(std::move(ParseSelfExpr()));
    }

    
    if (CurTok != ',')
      break;
    getNextToken(); // eat the ','.
  }

  

  if (CurTok != ']')
    return LogError("Expected ].");
    getNextToken();

  if (dims.size()!=2 && dims.size()!=3)
    return LogError("Embedding requires input vocabulary size and output hiddens count.");


  std::string pre_dot="";
  bool is_self = false;
  bool is_attr = false;
  if (CurTok == tok_self)
  {
    is_self=true;
    getNextToken();
  }
  if (CurTok == tok_class_attr)
  {
    is_attr=true;
    pre_dot = IdentifierStr;
    std::cout << "Obj attr pinned_tensor: " << pre_dot << ".\n";
    getNextToken();
  }

  if (CurTok != tok_identifier)
    return LogError("Expected Embedding identifier name.");



  while (true) {
    std::string Name = IdentifierStr;
    
    getNextToken(); // eat identifier.

    
    std::unique_ptr<ExprAST> Init = nullptr;
    VarNames.push_back(std::make_pair(Name, std::move(Init)));
    functionVars[Name] = "EmbeddingForward";

    // End of var list, exit loop.
    if (CurTok != ',')
      break;
    getNextToken(); // eat the ','.

    if (CurTok != tok_identifier)
      return LogError("Expected one or more identifiers after Embedding.");
  }



  auto aux = std::make_unique<EmbeddingExprAST>(std::move(VarNames), "embedding",
                                             std::move(dims[0]), std::move(dims[1]),
                                             init);
  aux->SetSelf(is_self);
  aux->SetIsAttribute(is_attr);
  aux->SetPreDot(pre_dot);

  
  if (CurTok==tok_space)
    getNextToken();
  
  return aux;
}




std::unique_ptr<ExprAST> ParseLinearExpr() {
  
  getNextToken(); // eat the Linear.
  
  if (CurTok != '[')
    return LogError("Linear declaration expected [");
    getNextToken();

  std::vector<std::unique_ptr<ExprAST>> dims;
  std::vector<std::pair<std::string, std::unique_ptr<ExprAST>>> VarNames;
  std::string init = "xavu";
  std::vector<int> notators_vec;
  //std::make_unique<NumberExprAST>(NumVal)
  
  while (true) {
    if (CurTok != tok_number && CurTok != tok_identifier && CurTok != tok_self)
      return LogError("Expected tensor dimension number.");
    
    if (CurTok==tok_number)
    {
      if (std::fmod(NumVal, 1.0) != 0)
        LogWarning("Tensor dimensions must be of type int. They are not supposed to be float.");
    
      dims.push_back(std::make_unique<NumberExprAST>( (float)((int)round(NumVal)) ));
      getNextToken();
    } else if (CurTok==tok_identifier)
      if (in_str(IdentifierStr, tensor_inits))
      {
        init = IdentifierStr;
        getNextToken();
      } else if (in_str(IdentifierStr, notators_str)){
        notators_vec.push_back(NotatorsMap[IdentifierStr]);
        getNextToken();
      } else
        dims.push_back(std::move(ParseIdentifierExpr()));
    else {
      dims.push_back(std::move(ParseSelfExpr()));
    }

    
    if (CurTok != ',')
      break;
    getNextToken(); // eat the ','.
  }


  if (CurTok != ']')
    return LogError("Expected ] at Linear.");
    getNextToken();

  if (dims.size()<2)
    return LogError("Linear requires input and output dimensions.");


  std::string pre_dot="";
  bool is_self = false;
  bool is_attr = false;
  if (CurTok == tok_self)
  {
    is_self=true;
    getNextToken();
  }
  if (CurTok == tok_class_attr)
  {
    is_attr=true;
    pre_dot = IdentifierStr;
    std::cout << "Obj attr pinned_tensor: " << pre_dot << ".\n";
    getNextToken();
  }

  if (CurTok != tok_identifier)
    return LogError("Expected Linear identifier name.");


  while (true) {
    std::string Name = IdentifierStr;
    
    getNextToken(); // eat identifier.

    
    std::unique_ptr<ExprAST> Init = nullptr;
    VarNames.push_back(std::make_pair(Name, std::move(Init)));
    functionVars[Name] = "LinearForward";

    // End of var list, exit loop.
    if (CurTok != ',')
      break;
    getNextToken(); // eat the ','.

    if (CurTok != tok_identifier)
      return LogError("Expected one or more identifiers after Linear.");
  }


  auto aux = std::make_unique<LinearExprAST>(std::move(VarNames), "linear",
                                             std::move(dims[0]), std::move(dims[1]),
                                             std::move(notators_vec),
                                             init);
  aux->SetSelf(is_self);
  aux->SetIsAttribute(is_attr);
  aux->SetPreDot(pre_dot);

  
  if (CurTok==tok_space)
    getNextToken();
  
  return aux;
}







std::unique_ptr<ExprAST> ParseMHSAExpr() {
  
  getNextToken(); // eat the MHSA.
  
  if (CurTok != '[')
    return LogError("MHSA declaration expected [");
    getNextToken();

  std::vector<std::unique_ptr<ExprAST>> dims;
  std::vector<std::pair<std::string, std::unique_ptr<ExprAST>>> VarNames;
  std::string init = "init_gpt";
  std::vector<int> notators_vec;
  //std::make_unique<NumberExprAST>(NumVal)
  
  while (true) {
    if (CurTok != tok_number && CurTok != tok_identifier && CurTok != tok_self)
      return LogError("Expected tensor dimension number.");
    
    if (CurTok==tok_number)
    {
      if (std::fmod(NumVal, 1.0) != 0)
        LogWarning("Tensor dimensions must be of type int. They are not supposed to be float.");
    
      dims.push_back(std::make_unique<NumberExprAST>( (float)((int)round(NumVal)) ));
      getNextToken();
    } else if (CurTok==tok_identifier)
      if (in_str(IdentifierStr, tensor_inits))
      {
        init = IdentifierStr;
        getNextToken();
      } else if (in_str(IdentifierStr, notators_str)){
        notators_vec.push_back(NotatorsMap[IdentifierStr]);
        getNextToken();
      } else
        dims.push_back(std::move(ParseIdentifierExpr()));
    else {
      dims.push_back(std::move(ParseSelfExpr()));
    }

    
    if (CurTok != ',')
      break;
    getNextToken(); // eat the ','.
  }

  

  if (CurTok != ']')
    return LogError("Expected ] at MHSA.");
    getNextToken();

  if (dims.size()!=3)
    return LogError("MHSA requires input vocabulary size and output hiddens count.");


  std::string pre_dot="";
  bool is_self = false;
  bool is_attr = false;
  if (CurTok == tok_self)
  {
    is_self=true;
    getNextToken();
  }
  if (CurTok == tok_class_attr)
  {
    is_attr=true;
    pre_dot = IdentifierStr;
    std::cout << "Obj attr pinned_tensor: " << pre_dot << ".\n";
    getNextToken();
  }

  if (CurTok != tok_identifier)
    return LogError("Expected MHSA identifier name.");



  while (true) {
    std::string Name = IdentifierStr;
    
    getNextToken(); // eat identifier.

    
    std::unique_ptr<ExprAST> Init = nullptr;
    VarNames.push_back(std::make_pair(Name, std::move(Init)));
    functionVars[Name] = "MHSAForward";

    // End of var list, exit loop.
    if (CurTok != ',')
      break;
    getNextToken(); // eat the ','.

    if (CurTok != tok_identifier)
      return LogError("Expected one or more identifiers after MHSA.");
  }



  auto aux = std::make_unique<MHSAExprAST>(std::move(VarNames), "mhsa",
                                             std::move(dims[0]), std::move(dims[1]), std::move(dims[2]),
                                             std::move(notators_vec),
                                             init);
  aux->SetSelf(is_self);
  aux->SetIsAttribute(is_attr);
  aux->SetPreDot(pre_dot);

  
  if (CurTok==tok_space)
    getNextToken();
  
  return aux;
}






//
std::unique_ptr<ExprAST> ParseMaxPool2dExpr() {
  std::string type;
  if (CurTok==tok_maxpool2d)
    type = "max";
  if (CurTok==tok_avgpool2d)
    type = "avg";

  getNextToken(); // eat the MaxPool2d.
  
  if (CurTok != '[')
    return LogError("MaxPool2d declaration expected [");
    getNextToken();

  std::vector<std::unique_ptr<ExprAST>> dims;
  std::vector<std::pair<std::string, std::unique_ptr<ExprAST>>> VarNames;
  std::string init = "";
  //std::make_unique<NumberExprAST>(NumVal)
  
  while (true) {
    if (CurTok != tok_number && CurTok != tok_identifier && CurTok != tok_self)
      return LogError("Expected tensor dimension number.");
    
    if (CurTok==tok_number)
    {
      if (std::fmod(NumVal, 1.0) != 0)
        LogWarning("Tensor dimensions must be of type int. They are not supposed to be float.");
    
      dims.push_back(std::make_unique<NumberExprAST>( (float)((int)round(NumVal)) ));
      getNextToken();
    } else if (CurTok==tok_identifier)
      if (in_str(IdentifierStr, tensor_inits))
      {
        init = IdentifierStr;
        getNextToken();
      } else
        dims.push_back(std::move(ParseIdentifierExpr()));
    else {
      dims.push_back(std::move(ParseSelfExpr()));
    }

    
    if (CurTok != ',')
      break;
    getNextToken(); // eat the ','.
  }

  

  if (CurTok != ']')
    return LogError("Expected ].");
    getNextToken();

  if (dims.size()<3)
    return LogError("MaxPool2d requires kernel size, stride and padding.");


  std::string pre_dot="";
  bool is_self = false;
  bool is_attr = false;
  if (CurTok == tok_self)
  {
    is_self=true;
    getNextToken();
  }
  if (CurTok == tok_class_attr)
  {
    is_attr=true;
    pre_dot = IdentifierStr;
    std::cout << "Obj attr pinned_tensor: " << pre_dot << ".\n";
    getNextToken();
  }

  if (CurTok != tok_identifier)
    return LogError("Expected conv identifier name.");



  while (true) {
    std::string Name = IdentifierStr;
    
    getNextToken(); // eat identifier.

    
    std::unique_ptr<ExprAST> Init = nullptr;
    VarNames.push_back(std::make_pair(Name, std::move(Init)));
    functionVars[Name] = "MaxPoolForward2d";

    // End of var list, exit loop.
    if (CurTok != ',')
      break;
    getNextToken(); // eat the ','.

    if (CurTok != tok_identifier)
      return LogError("Esperado um ou mais identificadores após var.");
  }



  auto aux = std::make_unique<MaxPool2dExprAST>(std::move(VarNames), type,
                                             std::move(dims[0]), std::move(dims[1]), std::move(dims[2]));
  aux->SetSelf(is_self);
  aux->SetIsAttribute(is_attr);
  aux->SetPreDot(pre_dot);

  
  if (CurTok==tok_space)
    getNextToken();
  
  return aux;
}





//
std::unique_ptr<ExprAST> ParseBatchNorm2dExpr() {
  std::string type;

  getNextToken(); // eat the BatchNorm2d.
  
  if (CurTok != '[')
    return LogError("BatchNorm2d declaration expected [");
    getNextToken();

  std::vector<std::unique_ptr<ExprAST>> dims;
  std::vector<std::pair<std::string, std::unique_ptr<ExprAST>>> VarNames;
  std::string init = "";
  //std::make_unique<NumberExprAST>(NumVal)
  
  while (true) {
    if (CurTok != tok_number && CurTok != tok_identifier && CurTok != tok_self)
      return LogError("Expected tensor dimension number.");
    
    if (CurTok==tok_number)
    {
      if (std::fmod(NumVal, 1.0) != 0)
        LogWarning("Tensor dimensions must be of type int. They are not supposed to be float.");
    
      dims.push_back(std::make_unique<NumberExprAST>( (float)((int)round(NumVal)) ));
      getNextToken();
    } else if (CurTok==tok_identifier)
      if (in_str(IdentifierStr, tensor_inits))
      {
        init = IdentifierStr;
        getNextToken();
      } else
        dims.push_back(std::move(ParseIdentifierExpr()));
    else {
      dims.push_back(std::move(ParseSelfExpr()));
    }

    
    if (CurTok != ',')
      break;
    getNextToken(); // eat the ','.
  }

  

  if (CurTok != ']')
    return LogError("Expected ].");
    getNextToken();

  if (dims.size()<1)
    return LogError("BatchNorm2d requires input channels, kernel size, stride and padding.");


  std::string pre_dot="";
  bool is_self = false;
  bool is_attr = false;
  if (CurTok == tok_self)
  {
    is_self=true;
    getNextToken();
  }
  if (CurTok == tok_class_attr)
  {
    is_attr=true;
    pre_dot = IdentifierStr;
    std::cout << "Obj attr pinned_tensor: " << pre_dot << ".\n";
    getNextToken();
  }

  if (CurTok != tok_identifier)
    return LogError("Expected BatchNorm2d identifier name.");



  while (true) {
    std::string Name = IdentifierStr;
    
    getNextToken(); // eat identifier.

    
    std::unique_ptr<ExprAST> Init = nullptr;
    VarNames.push_back(std::make_pair(Name, std::move(Init)));
    functionVars[Name] = "BatchNormForward2d";

    // End of var list, exit loop.
    if (CurTok != ',')
      break;
    getNextToken(); // eat the ','.

    if (CurTok != tok_identifier)
      return LogError("Expected one or more identifiers at BatchNorm2d.");
  }



  auto aux = std::make_unique<BatchNorm2dExprAST>(std::move(VarNames), type,
                                             std::move(dims[0]));
  aux->SetSelf(is_self);
  aux->SetIsAttribute(is_attr);
  aux->SetPreDot(pre_dot);

  
  if (CurTok==tok_space)
    getNextToken();
  
  return aux;
}




//
std::unique_ptr<ExprAST> ParseBN2dReluExpr() {
  std::string type;

  getNextToken(); // eat the BatchNorm2d.
  
  if (CurTok != '[')
    return LogError("BN2dRelu declaration expected [");
    getNextToken();

  std::vector<std::unique_ptr<ExprAST>> dims;
  std::vector<std::pair<std::string, std::unique_ptr<ExprAST>>> VarNames;
  std::string init = "";
  //std::make_unique<NumberExprAST>(NumVal)
  
  while (true) {
    if (CurTok != tok_number && CurTok != tok_identifier && CurTok != tok_self)
      return LogError("Expected tensor dimension number.");
    
    if (CurTok==tok_number)
    {
      if (std::fmod(NumVal, 1.0) != 0)
        LogWarning("Tensor dimensions must be of type int. They are not supposed to be float.");
    
      dims.push_back(std::make_unique<NumberExprAST>( (float)((int)round(NumVal)) ));
      getNextToken();
    } else if (CurTok==tok_identifier)
      if (in_str(IdentifierStr, tensor_inits))
      {
        init = IdentifierStr;
        getNextToken();
      } else
        dims.push_back(std::move(ParseIdentifierExpr()));
    else {
      dims.push_back(std::move(ParseSelfExpr()));
    }

    
    if (CurTok != ',')
      break;
    getNextToken(); // eat the ','.
  }

  

  if (CurTok != ']')
    return LogError("Expected ].");
    getNextToken();

  if (dims.size()<1)
    return LogError("BN2dRelu requires input channels, kernel size, stride and padding.");


  std::string pre_dot="";
  bool is_self = false;
  bool is_attr = false;
  if (CurTok == tok_self)
  {
    is_self=true;
    getNextToken();
  }
  if (CurTok == tok_class_attr)
  {
    is_attr=true;
    pre_dot = IdentifierStr;
    std::cout << "Obj attr pinned_tensor: " << pre_dot << ".\n";
    getNextToken();
  }

  if (CurTok != tok_identifier)
    return LogError("Expected BatchNorm2d identifier name.");



  while (true) {
    std::string Name = IdentifierStr;
    
    getNextToken(); // eat identifier.

    
    std::unique_ptr<ExprAST> Init = nullptr;
    VarNames.push_back(std::make_pair(Name, std::move(Init)));
    functionVars[Name] = "BN2dReluForward";

    // End of var list, exit loop.
    if (CurTok != ',')
      break;
    getNextToken(); // eat the ','.

    if (CurTok != tok_identifier)
      return LogError("Expected one or more identifiers at BN2dRelu.");
  }



  auto aux = std::make_unique<BN2dReluExprAST>(std::move(VarNames), type,
                                             std::move(dims[0]));
  aux->SetSelf(is_self);
  aux->SetIsAttribute(is_attr);
  aux->SetPreDot(pre_dot);

  
  if (CurTok==tok_space)
    getNextToken();
  
  return aux;
}



//
std::unique_ptr<ExprAST> ParseReluExpr() {
  std::string type;

  getNextToken(); // eat the Relu.
  
  std::vector<std::pair<std::string, std::unique_ptr<ExprAST>>> VarNames;
  

  std::string pre_dot="";
  bool is_self = false;
  bool is_attr = false;
  if (CurTok == tok_self)
  {
    is_self=true;
    getNextToken();
  }
  if (CurTok == tok_class_attr)
  {
    is_attr=true;
    pre_dot = IdentifierStr;
    std::cout << "Obj attr pinned_tensor: " << pre_dot << ".\n";
    getNextToken();
  }

  if (CurTok != tok_identifier)
    return LogError("Expected Relu identifier name.");



  while (true) {
    std::string Name = IdentifierStr;
    
    getNextToken(); // eat identifier.

    
    std::unique_ptr<ExprAST> Init = nullptr;
    VarNames.push_back(std::make_pair(Name, std::move(Init)));
    functionVars[Name] = "ReluForward";

    // End of var list, exit loop.
    if (CurTok != ',')
      break;
    getNextToken(); // eat the ','.

    if (CurTok != tok_identifier)
      return LogError("Expected one or more identifiers at Relu.");
  }



  auto aux = std::make_unique<ReluExprAST>(std::move(VarNames), type);
  aux->SetSelf(is_self);
  aux->SetIsAttribute(is_attr);
  aux->SetPreDot(pre_dot);

  
  if (CurTok==tok_space)
    getNextToken();
  
  return aux;
}





std::unique_ptr<ExprAST> ParseLockExpr(std::string class_name) {
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
  
  
  std::vector<std::unique_ptr<ExprAST>> Body = ParseIdentedBodies(cur_level_tabs, class_name);


  return std::make_unique<LockExprAST>(std::move(Body), Name);
}



std::unique_ptr<ExprAST> ParseNoGradExpr(std::string class_name) {
  int cur_level_tabs = SeenTabs;
  getNextToken(); // eat no_grad
  
  std::vector<std::unique_ptr<ExprAST>> Body = ParseIdentedBodies(cur_level_tabs, class_name);

  return std::make_unique<NoGradExprAST>(std::move(Body));
}



std::unique_ptr<ExprAST> ParseMustBeVar(std::string class_name, std::string expr_name) {

  std::unique_ptr<ExprAST> expr;

  if (CurTok==tok_class_attr||CurTok==tok_self)
    expr = ParseSelfExpr(class_name);
  else if (CurTok==tok_identifier)
    expr = ParseIdentifierExpr(class_name);
  else
  {
    std::string _error = expr_name + " expression expected a simple identifier, not another expression.";
    LogError(_error);
  }

  return std::move(expr);
}



std::unique_ptr<ExprAST> ParseGlobalExpr(std::string class_name) {
  getNextToken(); // eat global
  std::cout << "Parsing global expr\n";


  // At least one variable name is required.
  if (CurTok != tok_identifier)
    return LogError("Expected identifier after global.");

  while (true) {


    if (CurTok!=tok_identifier)
      return LogError("Global expression must contain identifiers only.");

    ParseIdentifierExpr(class_name);
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


std::unique_ptr<ExprAST> ParseReturnExpr(std::string class_name) {
  getNextToken(); // eat return
  std::cout << "Parsing var expr\n";

  std::vector<std::unique_ptr<ExprAST>> Vars, Destiny;
  std::vector<bool> IsAs;

  // At least one variable name is required.
  if (CurTok != tok_identifier)
    return LogError("Expected identifier after return.");

  while (true) {
    std::unique_ptr<ExprAST> expr, aux;


    expr = ParseMustBeVar(class_name, "return");

    
    if (CurTok == tok_as)
    {
      getNextToken(); // eat as
      aux = std::move(expr);

      expr = ParseMustBeVar(class_name, "return");

      IsAs.push_back(true);
      Vars.push_back(std::move(aux));
      Destiny.push_back(std::move(expr));

    } else {
      Destiny.push_back(std::move(expr));

      expr = std::make_unique<NumberExprAST>(0.0f);
      IsAs.push_back(false);
      Vars.push_back(std::move(expr));
    }

    // End of var list, exit loop.
    if (CurTok != ',')
      break;
    getNextToken(); // eat the ','.

    
  }

  if (CurTok==tok_space)
    getNextToken();


  return std::make_unique<ReturnExprAST>(std::move(Vars), std::move(IsAs), std::move(Destiny));
}




/// primary
///   ::= identifierexpr
///   ::= numberexpr
///   ::= parenexpr
///   ::= ifexpr
///   ::= forexpr
///   ::= varexpr
std::unique_ptr<ExprAST> ParsePrimary(std::string class_name) {
  switch (CurTok) {
  default:
    //return std::move(std::make_unique<NumberExprAST>(0.0f));
    return LogErrorT(CurTok);
  case tok_identifier:
    return ParseIdentifierExpr();
  case tok_class_attr:
    return ParseSelfExpr(class_name);
  case tok_self:
    return ParseSelfExpr(class_name);
  case tok_number:
    return ParseNumberExpr();
  case tok_str:
    return ParseStringExpr();
  case tok_var_str:
    return ParseStrExpr();
  case tok_str_vec:
    return ParseStrVecExpr();
  case tok_float_vec:
    return ParseStrVecExpr();
  case '(':
    return ParseParenExpr();
  case tok_if:
    return ParseIfExpr(class_name);
  case tok_for:
    return ParseForExpr(class_name);
  case tok_while:
    return ParseWhileExpr(class_name);
  case tok_async_finish:
    return ParseFinishExpr(class_name);
  case tok_async:
    return ParseAsyncExpr(class_name);
  case tok_lock:
    return ParseLockExpr(class_name);
  case tok_no_grad:
    return ParseNoGradExpr(class_name);
  case tok_return:
    return ParseReturnExpr(class_name);
  case tok_var:
    return ParseVarExpr(class_name);
  case tok_data:
    return ParseDataExpr(class_name);
  case tok_tensor:
    return ParseDataExpr(class_name);
  case tok_pinned_tensor:
    return ParsePinnedTensorExpr();
  case tok_conv2d:
    return ParseConv2dExpr();
  case tok_global:
    return ParseGlobalExpr();
  case tok_lstm:
    return ParseLSTMExpr();
  case tok_embedding:
    return ParseEmbeddingExpr();
  case tok_mhsa:
    return ParseMHSAExpr();
  case tok_linear:
    return ParseLinearExpr();
  case tok_maxpool2d:
    return ParseMaxPool2dExpr();
  case tok_avgpool2d:
    return ParseMaxPool2dExpr();
  case tok_batchnorm2d:
    return ParseBatchNorm2dExpr();
  case tok_bn2drelu:
    return ParseBN2dReluExpr();
  case tok_relu:
    return ParseReluExpr();
  case '[':
    return ParseNewVector(class_name);
  case tok_space:
    getNextToken();
    return ParsePrimary(class_name);
  }
}

/// unary
///   ::= primary
///   ::= '!' unary
std::unique_ptr<ExprAST> ParseUnary(std::string class_name) {
  //std::cout <<"Parse unary\n";
  if(CurTok==tok_space)
    getNextToken();
  // If the current token is not an operator, it must be a primary expr.
  
  //std::cout << "Unary current token " << ReverseToken(CurTok) << "\n";
  if (!isascii(CurTok) || CurTok == '(' || CurTok == ',' || CurTok == '[')
  {
    //std::cout << "Returning, non-ascii found.\n";
    return ParsePrimary(class_name);
  }
  
  
  // If this is a unary operator, read it.
  int Opc = CurTok;
  
  //std::cout << "Unary expr\n";
  getNextToken();
  if (auto Operand = ParseUnary(class_name))
    return std::make_unique<UnaryExprAST>(Opc, std::move(Operand));
  return nullptr;
}


/// binoprhs
///   ::= ('+' unary)*
std::tuple<std::unique_ptr<ExprAST>, int> ParseBinOpRHS(int ExprPrec,
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

  while (true)
  {
    // If this is a binop, find its precedence.
    int TokPrec = get_tokenPrecedence();

    // If this is a binop that binds at least as tightly as the current binop,
    // consume it, otherwise we are done.
    

    if (TokPrec==BinopPrecedence[':'])
    {
      getNextToken();
      return std::make_tuple(std::move(LHS), L_cuda);
    }
    if (TokPrec < ExprPrec)
      return std::make_tuple(std::move(LHS), L_cuda);
    
      


    if (CurTok == tok_space)
    {
      //std::cout << "Returning tok space with " << SeenTabs << " tabs. \n\n\n";
      getNextToken();
      return std::make_tuple(std::move(LHS), L_cuda);
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
      return std::make_tuple(std::move(LHS),L_cuda);
    }

    if (CurTok==')')
      return std::make_tuple(std::move(LHS),L_cuda);

    
    getNextToken(); // eat binop
    if (CurTok==tok_number)
      RName = std::to_string(NumVal);
    else
      RName = IdentifierStr;


    // Get the Right Hand Side token for debugging only
    RhsTok = CurTok;

    
    auto RHS = ParseUnary(class_name); // Returns an identifier, number or expression result
    if (!RHS)
      return std::make_tuple(nullptr,0);


    if (RHS->GetType()=="tensor")
      R_cuda=type_tensor;
    if (RHS->GetType()=="pinned_tensor")
      R_cuda=type_pinned_tensor;
    if (RHS->GetType()=="object"||RHS->GetType()=="object_vec")
      R_cuda=type_object;
    if (RHS->GetType()=="str")
      R_cuda = type_string;
    
    
    

    // If BinOp binds less tightly with RHS than the operator after RHS, let
    // the pending operator take RHS as its LHS.
    int NextPrec = get_tokenPrecedence();
    

    if (TokPrec < NextPrec)
    {
      //std::cout << NextPrec << " Next Prec\n";
        
      auto tuple = ParseBinOpRHS(TokPrec + 1, std::move(RHS));
      RHS = std::move(std::get<0>(tuple));
      R_cuda = std::get<1>(tuple);


      if (!RHS)
        return std::make_tuple(nullptr,0);
      
    }


    //std::cout << "\nBinary expression of BinOp and Rhs:" << "\n";
    //std::cout << ReverseToken(BinOp) << " " << ReverseToken(RhsTok) << "\n";
    //std::cout << "L type: " << L_cuda << " R type: " << R_cuda << "\n\n";


    if (L_cuda==type_string||R_cuda==type_string)
    {
      LHS = std::make_unique<ConcatStringsExprAST>(BinOp, std::move(LHS), std::move(RHS));
      LHS->SetType("str");
    }
    else if (R_cuda==type_object)
    {
      LHS = std::make_unique<BinaryObjExprAST>(BinOp, std::move(LHS), std::move(RHS));
      LHS->SetType("object");
    }
    else if (L_cuda==type_tensor && R_cuda==type_pinned_tensor)
    {
      //std::cout << "\nParse BinaryTensorPinned " << ReverseToken(BinOp) <<  "\n";
      LHS = std::make_unique<BinaryTensorPinnedExprAST>(BinOp,
                                                      std::move(LHS), std::move(RHS));
      LHS->SetType("tensor");
    }
    else if (L_cuda==type_pinned_tensor && R_cuda==type_float)
    {
      LHS = std::make_unique<BinaryPinnedScalarExprAST>(BinOp,
                                                      std::move(LHS), std::move(RHS));
      LHS->SetType("pinned_tensor");
    }
    else if (L_cuda==type_pinned_tensor && R_cuda==type_tensor)
    {
      LHS = std::make_unique<BinaryPinnedAndTensorExprAST>(BinOp,
                                                      std::move(LHS), std::move(RHS));
      LHS->SetType("pinned_tensor");
    }
    else if (L_cuda==type_tensor && R_cuda==type_float)
    {
      LHS = std::make_unique<BinaryTensorScalarExprAST>(BinOp,
                                                      std::move(LHS), std::move(RHS));
      LHS->SetType("tensor");  
    }
    else if (L_cuda==type_float && R_cuda==type_tensor)
    {
      std::cout << "Reverse LHS and RHS\n";
      //std::cout << "Bin op: " << BinOp << "\n";


      if (BinOp=='/')
        BinOp = 77; // scalar / tensor

      if (BinOp=='-') // inversion of 1 - tensor
      {
        RHS = std::make_unique<BinaryTensorScalarExprAST>('*',
                                                    std::move(RHS),
                                                    std::move(std::make_unique<NumberExprAST>(-1.0f)));
                                                    //std::move(LHS)
                                                    
        LHS = std::make_unique<BinaryTensorScalarExprAST>('+',
                                                    std::move(RHS), std::move(LHS));
      } else {
        if (BinOp!=':') // Avoid codegen reversing //todo: is this necessary anymore?
          LHS = std::make_unique<BinaryTensorScalarExprAST>(BinOp,
                                                    std::move(RHS), std::move(LHS));
        else
          LHS = std::make_unique<BinaryTensorScalarExprAST>(BinOp,
                                                    std::move(LHS), std::move(RHS));
      }
      LHS->SetType("tensor");  
      L_cuda=type_tensor;
      R_cuda=type_float;
    }
    else if (L_cuda==type_tensor && R_cuda==type_tensor)
    { 
      LHS = std::make_unique<BinaryTensorTensorExprAST>(BinOp,
                                                      std::move(LHS), std::move(RHS));
      R_cuda=type_float;
      LHS->SetType("tensor");
    }
    else
      LHS = std::make_unique<BinaryExprAST>(BinOp, std::move(LHS), std::move(RHS));
     

    LhsTok = RhsTok;    
  
  }
}


/// expression
///   ::= unary binoprhs
///
std::unique_ptr<ExprAST> ParseExpression(std::string class_name) {
  
  //std::cout << "Parse Expression\n";
  
  auto LHS = ParseUnary(class_name);
  if (!LHS)
    return nullptr;

  return std::get<0>(ParseBinOpRHS(0, std::move(LHS), class_name));
}

/// prototype
///   ::= id '(' id* ')'
///   ::= binary LETTER number? (id, id)
///   ::= unary LETTER (id)
std::unique_ptr<PrototypeAST> ParsePrototype(std::string class_name) {
  std::string FnName = class_name;
  std::string _class, method;
  method = "";
  _class = class_name;

  unsigned Kind = 0; // 0 = identifier, 1 = unary, 2 = binary.
  unsigned BinaryPrecedence = 30;

  switch (CurTok) {
  default:
    return LogErrorP("Esperado nome da função no protótipo");
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


  if (class_name!="") // If it is a class method, add self
  {
    Types.push_back("s");
    ArgNames.push_back("self");
  }
  Types.push_back("s");
  ArgNames.push_back("scope_str");
  Types.push_back("s");
  ArgNames.push_back("previous_scope");
  Types.push_back("i");
  ArgNames.push_back("thread_id");
  Types.push_back("i");
  ArgNames.push_back("has_grad");

  while (CurTok != ')')
  {
    type="str";
    if (IdentifierStr=="t")
      type="tensor";
    if (IdentifierStr=="c")
      type="function";
    if (IdentifierStr=="f")
      type="float";

    if (IdentifierStr!="t" && IdentifierStr!="f" && IdentifierStr!="s" && IdentifierStr!="c")
      LogErrorP_to_comma("Prototype var type must be t, f, s or c");
    else {
      Types.push_back(IdentifierStr);
      getNextToken(); // eat arg type

      ArgNames.push_back(IdentifierStr);

      if (type=="float")
        floatVars.push_back(IdentifierStr);
      else if (type=="tensor")
        tensorVars.push_back(IdentifierStr);
      else if (type=="function")
        functionVars[IdentifierStr] = "ConvForward2d";
      else
        strVars.push_back(IdentifierStr);
      
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

  // success.
  getNextToken(); // eat ')'.

  // Verify right number of names for operator.
  if (Kind && ArgNames.size() != Kind)
    return LogErrorP("Número inválido de operandos para o operador");

  if (CurTok!=tok_space)
    LogError("Post prototype parsing requires a line break.");
  getNextToken();


  return std::make_unique<PrototypeAST>(FnName, _class, method, ArgNames, Types, Kind != 0,
                                         BinaryPrecedence);
}



/// definition ::= 'def' prototype expression
std::unique_ptr<FunctionAST> ParseDefinition(std::string class_name) {

  int cur_level_tabs = SeenTabs;

  getNextToken(); // eat def.


  auto Proto = ParsePrototype(class_name);
  if (!Proto)
  {
    std::string _error = "Error defining " + class_name + " prototype.";  
    LogError(_error);
    return nullptr;
  } 
  
  
  std::vector<std::unique_ptr<ExprAST>> Body;

  while(!in_char(CurTok, terminal_tokens))
  {
    if (SeenTabs <= cur_level_tabs && CurTok != tok_space)
      break;
      

    if (CurTok==tok_space)
      getNextToken();

    if (SeenTabs <= cur_level_tabs)
      break;

    Body.push_back(std::move(ParseExpression(class_name)));
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
std::unique_ptr<FunctionAST> ParseTopLevelExpr() {
  //std::cout << "Top Level Expression\n";

  
  std::vector<std::unique_ptr<ExprAST>> Body;
  while(!in_char(CurTok, terminal_tokens))
  {
    Body.push_back(std::move(ParseExpression()));
    //std::cout << "\n\nTop level expr cur tok: " << ReverseToken(CurTok) <<  ".\n";
    //std::cout << "Top level expr number of expressions: " << Body.size() <<  ".\n\n\n";
  }
  

  // Make an anonymous proto.
  auto Proto = std::make_unique<PrototypeAST>("__anon_expr", "", "",
                                                std::vector<std::string>(),
                                                std::vector<std::string>());
    
  return std::make_unique<FunctionAST>(std::move(Proto), std::move(Body));
  
  return nullptr;
}



/// external ::= 'extern' prototype
std::unique_ptr<PrototypeAST> ParseExtern() {
  getNextToken(); // eat extern.
  return ParsePrototype();
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
  while(CurTok==tok_def)
  {
    
    auto Func = ParseDefinition(Name);
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