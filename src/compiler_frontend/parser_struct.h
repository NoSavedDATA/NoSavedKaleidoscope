#pragma once



#include <map>
#include <string>


struct Parser_Struct {
  std::string class_name="";
  std::string function_name="";
  std::string prev_function_name="";
  bool can_be_string=false;
  bool can_be_list=false;
  int line=0;
  int loop_depth=0;
  int control_flow_depth=0;
};
