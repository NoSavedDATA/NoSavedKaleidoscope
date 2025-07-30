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
  if (not ShallCodegen)
    return ConstantFP::get(*TheContext, APFloat(0.0f));



  Value *name;
  int type;
  std::vector<std::unique_ptr<ExprAST>> idx;

  
  bool include_scope = GetSolverIncludeScope();
  Value *var_name = Builder->CreateCall(TheModule->getFunction("GetEmptyChar"), {});

  


  if(Names.size()>1)
    for (int i=0; i<Names.size()-1;i++)
    {
      name = Builder->CreateGlobalString(std::get<0>(Names[i]));
      type = std::get<1>(Names[i]);
      idx = std::move(std::get<2>(Names[i]));


      if (i==0)
      {
        if (type==type_self)
          var_name = callret("ConcatStrFreeLeft", {var_name, callret("get_scope_first_arg", {scope_struct})});
        else
        {
          // if (include_scope)
          //   var_name = callret("ConcatScopeStr", {var_name, callret("get_scope_scope", {scope_struct})});
          // else
          //   p2t("DO NOT CONCATENATE SCOPE");
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


  // if(Names.size()==1)// Concat scope only
  //   if((Type=="object"||Type=="tensor"||Type=="float"||Type=="str"||Type=="float_vec"||Type=="int_vec")&&include_scope)
  //     var_name = callret("ConcatScopeStr", {var_name, callret("get_scope_scope", {scope_struct})});


  if(NameSolveToLast)
  {
    name = Builder->CreateGlobalString(std::get<0>(Names[Names.size()-1]));
    idx = std::move(std::get<2>(Names[Names.size()-1]));

    var_name = callret("ConcatStrFreeLeft", {var_name, name});
    if (Type=="object_vec")
    {
      Value *_idx = idx[0]->codegen(scope_struct);
      var_name = callret("ConcatNumToStrFree", {var_name, _idx});
    }
  }

  return var_name;
}