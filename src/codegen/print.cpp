#include <string>
#include <iostream>

extern "C" float print(char* str, float x){
  std::string _str = str;
  std::cout << "\n" << _str << " " << x << "\n";
  return 0;
}