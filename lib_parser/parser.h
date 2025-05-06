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



void Write_Txt(std::string fname, std::string content);

void Parse_Libs();