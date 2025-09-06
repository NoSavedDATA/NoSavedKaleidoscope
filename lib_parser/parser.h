#pragma once



#include <map>
#include <string>
#include <vector>



struct Lib_Info{
    std::string llvm_string="";
    std::string arg_types_string="";
    std::string return_data_string = "";
    std::string dict_string="";
    std::string functions_string = "";
    std::string clean_up_functions = "";
    std::string backward_functions = "";
    Lib_Info();
};



void Write_Txt(std::string fname, std::string content);
void Write_Append(std::string fname, std::string content);

void Parse_Libs();

extern std::string file_name;