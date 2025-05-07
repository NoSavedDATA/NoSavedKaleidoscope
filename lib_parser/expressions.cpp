#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "include.h"



ExternFunctionExpr::ExternFunctionExpr(const std::string &ReturnType, const std::string &FunctionName, std::vector<std::string> ArgTypes, bool Vararg)
    : ReturnType(ReturnType), FunctionName(FunctionName), ArgTypes(std::move(ArgTypes)) {this->Vararg=Vararg;};


PlaceholderExpr::PlaceholderExpr() {};
Lib_Info *PlaceholderExpr::Generate_LLVM(std::string fname, Lib_Info *lib_info) {
    // std::cout << "Deal with placeholder" << ".\n";
    return lib_info;
}


CppFunctionExpr::CppFunctionExpr(const std::string & FunctionName) : FunctionName(FunctionName) {};

Lib_Info *CppFunctionExpr::Generate_LLVM(std::string fname, Lib_Info *lib_info) {
    // std::cout << "Deal with cpp function " << FunctionName << ".\n";

    std::string dict_key="";

    if(ends_with(FunctionName, "_Clean_Up"))
    {
        dict_key=FunctionName; 
        size_t pos = dict_key.rfind("_Clean_Up");
        if (pos != std::string::npos) {
            dict_key.replace(pos, 9, "");
        }
        lib_info->clean_up_functions = lib_info->clean_up_functions + "\tclean_up_functions[\"" + dict_key + "\"] = " + FunctionName+";\n";
    } else if (ends_with(FunctionName, "_backward"))
        lib_info->backward_functions = lib_info->backward_functions + "\tbackward_functions[\"" + FunctionName + "\"] = " + FunctionName+";\n";
    else {

    }




    return lib_info;
}




Lib_Info *Generate_Function_Dict(Lib_Info *lib_info, std::string in_return_type, std::string function_name) {

    std::string return_type="";
    if(in_return_type=="char*")
        return_type = "str";
    else if (in_return_type=="std::vector<char*>*")
        return_type = "str_vec";
    else if (in_return_type=="std::vector<float>*")
        return_type = "float_vec";
    else if (begins_with(in_return_type, "DT_")) {
        return_type = remove_substring(in_return_type, "DT_");
        if (ends_with(return_type, "*"))
            return_type = remove_substring(return_type, "*");
    }
    else {

    }


    // std::cout << in_return_type << " --> " << return_type << ".\n";


    if (in_return_type!="float"&&in_return_type!="void"&&in_return_type!="void*") 
        lib_info->dict_string = lib_info->dict_string + "{\"" + function_name + "\", \"" + return_type + "\"}, ";
    lib_info->functions_string = lib_info->functions_string + "\"" +  function_name + "\", ";

    return lib_info;
}

Lib_Info *ExternFunctionExpr::Generate_LLVM(std::string fname, Lib_Info *lib_info) {
    // std::cout << "ExternFunctionExpr for file " << fname << ".\n";
    // std::cout << "Function:\n\tReturn Type:\t" << ReturnType << "\n\tName:\t\t" << FunctionName << "\n\tArgs:\t\t";

    // if(ArgTypes.size()>0)
    // {
    //     for (int i=0;i<ArgTypes.size()-1;++i)
    //         std::cout << ArgTypes[i] << ", ";
    //     std::cout << ArgTypes[ArgTypes.size()-1];
    // }


    // std::cout <<  "\n\n\n";
    

    lib_info = Generate_Function_Dict(lib_info, ReturnType, FunctionName); 


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



    lib_info->llvm_string = lib_info->llvm_string + "\n" + line1 + line2 + line3 + line4 + line5 + line6;

    return lib_info;
}