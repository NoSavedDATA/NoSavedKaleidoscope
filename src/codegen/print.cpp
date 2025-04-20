#include <string>
#include <iostream>

#include "../mangler/scope_struct.h"

extern "C" float print(Scope_Struct *scope_struct, char* str){
  // std::string _str = str;
  // std::cout << "\n" << str  << "\n";
  std::cout << str  << "\n";
  return 0;
}