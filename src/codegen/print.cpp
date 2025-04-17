#include <string>
#include <iostream>

extern "C" float print(char* str){
  // std::string _str = str;
  // std::cout << "\n" << str  << "\n";
  std::cout << str  << "\n";
  return 0;
}