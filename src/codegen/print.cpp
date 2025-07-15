#include "../mangler/scope_struct.h"

extern "C" float print(Scope_Struct *scope_struct, char* str){
  printf("%s\n", str);
  return 0;
}