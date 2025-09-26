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
  Value *var_name = Builder->CreateCall(TheModule->getFunction("GetEmptyChar"), {scope_struct});

  


  if(Names.size()>1)
    for (int i=0; i<Names.size()-1;i++)
    {
      name = Builder->CreateGlobalString(std::get<0>(Names[i]));
      type = std::get<1>(Names[i]);
      idx = std::move(std::get<2>(Names[i]));


      if (i==0&&type==type_self)
        var_name = callret("ConcatStrFreeLeft", {scope_struct, var_name, callret("get_scope_first_arg", {scope_struct})});
      


      if (type==type_attr||type==type_var)
        var_name = Builder->CreateCall(TheModule->getFunction("ConcatStrFreeLeft"),
                                                        {scope_struct, var_name, name});

    }


  if(NameSolveToLast)
  {
    name = Builder->CreateGlobalString(std::get<0>(Names[Names.size()-1]));
    idx = std::move(std::get<2>(Names[Names.size()-1]));

    var_name = callret("ConcatStrFreeLeft", {scope_struct, var_name, name});
    if (Type=="object_vec")
    {
      Value *_idx = idx[0]->codegen(scope_struct);
      var_name = callret("ConcatNumToStrFree", {scope_struct, var_name, _idx});
    }
  }

  return var_name;
}