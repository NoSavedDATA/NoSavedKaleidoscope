#pragma once



#include <map>
#include <string>
#include <vector>



struct Lib_Info{
    std::string llvm_string="";
    std::string dict_string="";
    std::string functions_string = "";
    Lib_Info();
};


std::string Mangle_Lib_File_Name(std::string fname);


void Parse_Libs();