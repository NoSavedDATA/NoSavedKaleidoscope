#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "include.h"



ExternFunctionExpr::ExternFunctionExpr(const std::string &ReturnType, const std::string &FunctionName, std::vector<std::string> ArgTypes, bool Vararg)
    : ReturnType(ReturnType), FunctionName(FunctionName), ArgTypes(std::move(ArgTypes)) {this->Vararg=Vararg;};



std::string ExternFunctionExpr::Generate_LLVM(std::string fname, std::string lib_string) {
    // std::cout << "ExternFunctionExpr for file " << fname << ".\n";
    std::cout << "Function:\n\tReturn Type:\t" << ReturnType << "\n\tName:\t\t" << FunctionName << "\n\tArgs:\t\t";

    if(ArgTypes.size()>0)
    {
        for (int i=0;i<ArgTypes.size()-1;++i)
            std::cout << ArgTypes[i] << ", ";
        std::cout << ArgTypes[ArgTypes.size()-1];
    }


    std::cout <<  "\n\n\n";



    std::string fTy = FunctionName+"Ty";

    std::string line1 = "\tFunctionType *" + fTy + "= FunctionType::get(\n";

    std::string line2;

    if (ReturnType=="float")
        line2="\t\tType::getFloatTy(*TheContext),\n";
    else if (ReturnType=="int")
        line2="\t\tType::getInt32Ty(*TheContext),\n";
    else
        line2="\t\tint8PtrTy,\n";


    std::string line3 = "\t\t{";

    if (ArgTypes.size()>0)
    {
        for(int i=0; i<ArgTypes.size(); ++i)
        {
            if (ArgTypes[i]=="float")
                line3 = line3 + "Type::getFloatTy(*TheContext)";
            else if (ArgTypes[i]=="int")
                line3 = line3 + "Type::getInt32Ty(*TheContext)";
            else
                line3 = line3 + "int8PtrTy";

            if (i!=ArgTypes.size()-1)
                line3 = line3 + ", ";
        }
    }
    
    line3 = line3 + "},\n";

    std::string line4;
    if(Vararg)
        line4 = "\t\ttrue //vararg\n";
    else
        line4 = "\t\tfalse\n";
    
    std::string line5="\t);\n";

    std::string line6="\tTheModule->getOrInsertFunction(\"" + FunctionName + "\", " + fTy + ");\n";



    lib_string = lib_string + "\n" + line1 + line2 + line3 + line4 + line5 + line6;

    return lib_string;
}