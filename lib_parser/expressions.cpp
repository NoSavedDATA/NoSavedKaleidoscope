#include <algorithm>
#include <cassert>
#include <cctype>
#include <complex>
#include <iostream>
#include <iomanip>
#include <map>
#include <string>
#include <string_view>
#include <vector>

#include "include.h"


void removeSpacesAndAsterisks(std::string& s) {
    s.erase(std::remove_if(s.begin(), s.end(),
                           [](char c) { return c == ' ' || c == '*'; }),
            s.end());
}

std::string remove_suffix(const std::string& input_string, std::string suffix) {
    if (input_string.length() >= suffix.length() && 
        input_string.substr(input_string.length() - suffix.length()) == suffix) {
        return input_string.substr(0, input_string.length() - suffix.length());
    }
    return input_string;
}



ExternFunctionExpr::ExternFunctionExpr(const std::string &ReturnType, const std::string &FunctionName, std::vector<std::string> ArgTypes, bool Vararg)
    : ReturnType(ReturnType), FunctionName(FunctionName), ArgTypes(std::move(ArgTypes)) {this->Vararg=Vararg;};


PlaceholderExpr::PlaceholderExpr() {};
Lib_Info *PlaceholderExpr::Generate_LLVM(std::string fname, Lib_Info *lib_info) {
    // std::cout << "Deal with placeholder" << ".\n";
    return lib_info;
}


CppFunctionExpr::CppFunctionExpr(const std::string & FunctionName) : FunctionName(FunctionName) {};

Lib_Info *CppFunctionExpr::Generate_LLVM(std::string fname, Lib_Info *lib_info) {
    // Build std::maps<> for _backward and _Clean_Up functions

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
    else if (in_return_type=="int")
        return_type = "int";
    else if (in_return_type=="uint64_t")
        return_type = "uint64_t";
    else if (in_return_type=="int64_t")
        return_type = "int64_t";
    else if (in_return_type=="float")
        return_type = "float";
    else if (in_return_type=="bool")
        return_type = "bool";
    else if (in_return_type=="std::vector<char*>*")
        return_type = "str_vec";
    else if (in_return_type=="std::vector<float>*")
        return_type = "float_vec";
    else if (begins_with(in_return_type, "DT_")) {
        return_type = remove_substring(in_return_type, "DT_");
        if (ends_with(return_type, "*"))
            return_type = remove_substring(return_type, "*");
    }
    else {}

    

    if (in_return_type!="void"&&in_return_type!="void*") {
        if (ends_with(return_type, "_vec")) {
            std::string vec_type = remove_substring(return_type, "_vec");
            std::string data_str = "\n\tData_Tree " + function_name+"_vec = Data_Tree(\"vec\");\n";
            data_str += "\t"+function_name+"_vec.Nested_Data.push_back(Data_Tree(\"" + vec_type + "\"));\n";
            data_str += "\tfunctions_return_data_type[\""+function_name+"\"] = " + function_name+"_vec;\n";

            lib_info->return_data_string = lib_info->return_data_string + data_str; 
        }
        else
            lib_info->return_data_string = lib_info->return_data_string + "\tfunctions_return_data_type[\"" + function_name \
                                                                        + "\"] = Data_Tree(\"" + return_type + "\");\n";
        lib_info->dict_string = lib_info->dict_string + "{\"" + function_name + "\", \"" + return_type + "\"}, "; // {"tensor_tensor_add", "tensor"}
    }
    lib_info->functions_string = lib_info->functions_string + "\"" +  function_name + "\", "; // user_cpp_functions


    return lib_info;
}


Lib_Info *PlaceholderExpr::Generate_Args_Dict(Lib_Info *lib_info) {}
Lib_Info *CppFunctionExpr::Generate_Args_Dict(Lib_Info *lib_info) {}

Lib_Info *ExternFunctionExpr::Generate_Args_Dict(Lib_Info *lib_info) {

    // Handle Arg Types
    // if (ends_with(FunctionName, "_Create")||ends_with(FunctionName, "_Load")||ends_with(FunctionName, "_Copy")\
    //     ||ends_with(FunctionName, "_New")||ends_with(FunctionName, "_CopyArg")||ends_with(FunctionName, "_Idx")\
    //     ||ends_with(FunctionName, "_Idx_num")||ends_with(FunctionName, "_Slice")||ends_with(FunctionName, "_Split_Parallel")\
    //     ||ends_with(FunctionName, "_Split_Strided_Parallel")||ends_with(FunctionName, "_CalculateSliceIdx")\
    //     ||ends_with(FunctionName, "_CalculateIdx")||ends_with(FunctionName, "_Store_Idx")||begins_with(FunctionName, "scope_struct")\
    //     ||contains_str(FunctionName, "Attr_on_Offset")||contains_str(FunctionName, "Load_on_Offset")||begins_with(FunctionName, "set_scope")\
    //     ||begins_with(FunctionName, "get_scope")||FunctionName=="FirstArgOnDemand"||begins_with(FunctionName, "MarkToSweep"))
    //     return lib_info;


    std::string arg_names_line = "\n\t";
    std::string arg_types_line = "\n\t";
    std::string arg_data_types_line = "\n\t";


    if (ArgTypes.size()>0)
    {
        for(int i=0; i<ArgTypes.size(); ++i)
        {
            
            std::string arg_type = ArgTypes[i];
            removeSpacesAndAsterisks(arg_type);

            if(i==0&&arg_type!="Scope_Struct")
                return lib_info;

            std::string arg_name = std::to_string(i);

            
            
            if(arg_type=="std::vector<char>")
                arg_type = "str_vec";
            if (arg_type=="char")
                arg_type = "str";
            if (begins_with(arg_type, "DT_"))
                arg_type = remove_substring(arg_type, "DT_");
            
            arg_names_line = arg_names_line + "\n\tFunction_Arg_Names[\"" + FunctionName + "\"].push_back(\"" + arg_name + "\");";

            arg_types_line = arg_types_line + "\n\tFunction_Arg_Types[\"" + FunctionName + "\"][\"" + arg_name + "\"] = \"" + arg_type + "\";";



            if(ends_with(arg_type, "vec")) {
                std::string data_tree_type = FunctionName+"_"+arg_name;
                arg_data_types_line = arg_data_types_line + "\n\tData_Tree " + data_tree_type + " = Data_Tree(\"vec\");";
                arg_data_types_line = arg_data_types_line + "\n\t"+ data_tree_type \
                                                          + ".Nested_Data.push_back(Data_Tree(\"" +remove_suffix(arg_type,"_vec")+ "\"));";
                arg_data_types_line = arg_data_types_line + "\n\tFunction_Arg_DataTypes[\"" + FunctionName \
                                                          + "\"][\"" + arg_name + "\"] = " + data_tree_type + ";";
            } else if (arg_type=="list") {
                std::string data_tree_type = FunctionName+"_"+arg_name;
                arg_data_types_line = arg_data_types_line + "\n\tData_Tree " + data_tree_type + " = Data_Tree(\"list\");";
                arg_data_types_line = arg_data_types_line + "\n\t"+ data_tree_type + ".Nested_Data.push_back(Data_Tree(\"any\"));";
                arg_data_types_line = arg_data_types_line + "\n\tFunction_Arg_DataTypes[\"" + FunctionName + "\"][\"" \
                                                          + arg_name + "\"] = " + data_tree_type + ";";
            } else if (arg_type=="array") {
                std::string data_tree_type = FunctionName+"_"+arg_name;
                // std::cout << arg_data_types_line << "\n\tData_Tree " << data_tree_type << " = Data_Tree(\"array\");";
                // std::cout << arg_data_types_line << "\n\t"<<  data_tree_type << ".Nested_Data.push_back(Data_Tree(\"any\"));";

                arg_data_types_line = arg_data_types_line + "\n\tData_Tree " + data_tree_type + " = Data_Tree(\"array\");";
                arg_data_types_line = arg_data_types_line + "\n\t"+ data_tree_type + ".Nested_Data.push_back(Data_Tree(\"any\"));";
                arg_data_types_line = arg_data_types_line + "\n\tFunction_Arg_DataTypes[\"" + FunctionName + "\"][\"" + arg_name + "\"] = " + data_tree_type + ";";
            } else
                arg_data_types_line = arg_data_types_line + "\n\tFunction_Arg_DataTypes[\"" + FunctionName + "\"][\"" + arg_name + "\"] = Data_Tree(\"" + arg_type + "\");";
        }
    }


    lib_info->arg_types_string = lib_info->arg_types_string + arg_types_line + arg_data_types_line + arg_names_line;

    // std::cout << "Got lib_info " << arg_types_line << "\n\n" << arg_names_line << ".\n";

    return lib_info;
}



Lib_Info *ExternFunctionExpr::Generate_LLVM(std::string fname, Lib_Info *lib_info) {
    

    // std::cout << "generate function dict for " << FunctionName << ".\n";
    lib_info = Generate_Function_Dict(lib_info, ReturnType, FunctionName); 

    if (!(FunctionName=="pthread_create_aux"||FunctionName=="pthread_join_aux"))
    {



        std::string fTy = FunctionName+"Ty";

        std::string line1 = "\tFunctionType *" + fTy + "= FunctionType::get(\n";

        std::string line2;

        if (ReturnType=="float")
            line2="\t\tType::getFloatTy(*TheContext),\n";
        else if (ReturnType=="int")
            line2="\t\tType::getInt32Ty(*TheContext),\n";
        else if (ReturnType=="uint64_t")
            line2="\t\tType::getInt64Ty(*TheContext),\n";
        else if (ReturnType=="int64_t")
            line2="\t\tType::getInt64Ty(*TheContext),\n";
        else if (ReturnType=="bool")
            line2="\t\tType::getInt1Ty(*TheContext),\n";
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
                else if (ArgTypes[i]=="uint64_t")
                    line3 = line3 + "Type::getInt64Ty(*TheContext)";
                else if (ArgTypes[i]=="int64_t")
                    line3 = line3 + "Type::getInt64Ty(*TheContext)";
                else if (ArgTypes[i]=="bool")
                    line3 = line3 + "Type::getInt1Ty(*TheContext)";
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
    } else 
        lib_info->llvm_string = lib_info->llvm_string + "\n";




    lib_info = Generate_Args_Dict(lib_info);

    
    return lib_info;
}
