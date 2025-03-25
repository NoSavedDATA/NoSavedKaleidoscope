#include <iostream>
#include "str.h"


extern "C" float PrintStr(char* value){
  std::cout << "Str: " << value << "\n";
  return 0;
}