
#include "llvm/IR/Value.h"
#include "llvm/IR/Verifier.h"


#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "../compiler_frontend/modules.h"


std::map<std::string, std::string> Lib_Functions_Return;
std::map<std::string, std::vector<std::string>> Lib_Functions_Args;

void Generate_Lib_Functions() {
    
    llvm::Type *int8PtrTy = Type::getInt8Ty(*TheContext)->getPointerTo();
    llvm::Type *boolTy = Type::getInt1Ty(*TheContext);
    llvm::Type *floatTy = Type::getFloatTy(*TheContext);
    llvm::Type *intTy = Type::getInt32Ty(*TheContext);



    for(auto pair : Lib_Functions_Return) {

        llvm::Type *fn_return_type;
        std::vector<llvm::Type *> arg_types;

        if (pair.second=="int")
            fn_return_type = intTy;
        else if (pair.second=="float")
            fn_return_type = floatTy;
        else if (pair.second=="bool")
            fn_return_type = boolTy;
        else
            fn_return_type = int8PtrTy;

        std::vector<std::string> arg_types_str = Lib_Functions_Args[pair.first];
        for (int i=0; i<arg_types_str.size(); ++i)
        {
            if (arg_types_str[i]=="int")
                arg_types.push_back(intTy);
            else if (arg_types_str[i]=="float")
                arg_types.push_back(floatTy);
            else
                arg_types.push_back(int8PtrTy);
        }

        
        FunctionType *llvm_function = FunctionType::get(
            fn_return_type,
            arg_types,
            false
        );

        Function* funcDecl = cast<Function>(
            TheModule->getOrInsertFunction(pair.first, llvm_function).getCallee()
        );
    }
}