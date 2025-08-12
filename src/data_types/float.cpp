#include <iostream>
#include <vector>
#include <string>

#include "../codegen/string.h"
#include "../common/extension_functions.h"
#include "../mangler/scope_struct.h"
#include "include.h"







extern "C" float print_float(float value){
  std::cout << "print_float: " << std::to_string(value) << ".\n";
  return value;
}






