#pragma once



#include <map>
#include <string>


struct Parser_Struct {
  std::string class_name="";
  std::string function_name="";
  bool can_be_string=false;
  bool can_be_list=false;
  int line=0;
};
