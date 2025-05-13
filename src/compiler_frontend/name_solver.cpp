#include "llvm/IR/Value.h"

#include <string>
#include <map>
#include <vector>

#include "../common/extension_functions.h"
#include "../data_types/include.h"
#include "expressions.h"
#include "logging.h"
#include "modules.h"

Value *NameSolverAST::codegen(Value *scope_struct) {
  // std::cout << "\n\n\nName solver type: " << Type << "\n\n\n\n";
  if (not ShallCodegen)
    return ConstantFP::get(*TheContext, APFloat(0.0f));



  Value *name;
  int type;
  std::vector<std::unique_ptr<ExprAST>> idx;

  
  bool include_scope = GetSolverIncludeScope();
  Value *var_name = Builder->CreateCall(TheModule->getFunction("GetEmptyChar"), {});

  
  // std::cout << "NAME SOLVER HAS SIZE: " << Names.size() << ".\n";


  if(Names.size()>1)
    for (int i=0; i<Names.size()-1;i++)
    {
      name = Builder->CreateGlobalString(std::get<0>(Names[i]));
      type = std::get<1>(Names[i]);
      idx = std::move(std::get<2>(Names[i]));

    //  std::cout << "NameSolver[" << i<< "]:  " << std::get<0>(Names[i]) << ", type: " << type << "\n";

      if (i==0)
      {
        if (type==type_self)
          var_name = callret("ConcatStrFreeLeft", {var_name, callret("get_scope_first_arg", {scope_struct})});
        else
        {
          // p2t("concat scope, scope is: ");
          // call("scope_struct_Print", {scope_struct});

          // std::cout << "888888888888888 INCLUDE SCOPE " << include_scope << " TYPE: " << Type << ".\n";
          // if((Type=="object"||Type=="tensor"||Type=="float"||type==type_object_name||Type=="str")&&include_scope)
          if (include_scope)
            var_name = callret("ConcatScopeStr", {var_name, callret("get_scope_scope", {scope_struct})});
          else
            p2t("DO NOT CONCATENATE SCOPE");
        }
      }

      if (type==type_object_name)
      {
        var_name = Builder->CreateCall(TheModule->getFunction("ConcatStrFreeLeft"),
                                                        {var_name, name});
        var_name = callret("LoadObject", {var_name});

      }

      if (type==type_attr||type==type_var)
        var_name = Builder->CreateCall(TheModule->getFunction("ConcatStrFreeLeft"),
                                                        {var_name, name});

      if (type==type_vec)
      {
        Value *_idx = idx[0]->codegen(scope_struct);
        var_name = Builder->CreateCall(TheModule->getFunction("ConcatStrFreeLeft"),
                                                        {var_name, name});
        var_name = Builder->CreateCall(TheModule->getFunction("ConcatNumToStrFree"),
                                                        {var_name, _idx});
        var_name = Builder->CreateCall(TheModule->getFunction("LoadObjectScopeName"),
                                                        {var_name});
      }
    }


  if(Names.size()==1)// Concat scope only
    if((Type=="object"||Type=="tensor"||Type=="float"||Type=="str"||Type=="float_vec")&&include_scope)
      var_name = callret("ConcatScopeStr", {var_name, callret("get_scope_scope", {scope_struct})});


  if(NameSolveToLast)
  {
    name = Builder->CreateGlobalString(std::get<0>(Names[Names.size()-1]));
    idx = std::move(std::get<2>(Names[Names.size()-1]));

    //std::cout << "\n\n\nNAMESOLVER TYPE OF LAST: " << type << "\n";
    //std::cout << "For: " << std::get<0>(Names[Names.size()-1]) << "\n";
    //std::cout << "Type: " << Type << "\n";
    var_name = callret("ConcatStrFreeLeft", {var_name, name});
    if (Type=="object_vec")
    {
      Value *_idx = idx[0]->codegen(scope_struct);
      var_name = callret("ConcatNumToStrFree", {var_name, _idx});
    }
  }

  return var_name;
}