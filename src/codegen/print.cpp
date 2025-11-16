#include "../mangler/scope_struct.h"

extern "C" float print(Scope_Struct *scope_struct, char* str){
  printf("%s\n", str);
  return 0;
}

extern "C" void print_void_ptr(void *x) {
    std::cout << "--->GOT void*: " << x << ".\n";
}

extern "C" void print_int(int x) {
    // if(x!=0)
    std::cout << "GOT INT: " << x << ".\n";
}

extern "C" void print_uint64(uint64_t x) {
    std::cout << "GOT U INT 64: " << x << ".\n";
}