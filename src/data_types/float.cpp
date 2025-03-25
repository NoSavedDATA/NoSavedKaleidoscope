#include <iostream>
#include <vector>
#include <string>

#include "../common/extension_functions.h"


extern "C" void *to_string(float v)
{
  //todo: allow float instead of int only
  return str_to_char(std::to_string((int)v));
}


extern "C" void PrintFloat(float value){
  std::cout << "Printing float.\n";
  std::cout << "Float value: " << value << "\n";
}


extern "C" float UnbugFloat(float value){
    return value;
}