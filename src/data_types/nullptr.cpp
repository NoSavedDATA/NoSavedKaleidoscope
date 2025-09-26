#include <iostream>
#include <vector>
#include <map>

#include "../codegen/random.h"
#include "../mangler/scope_struct.h"
#include "include.h"



extern "C" void *nullptr_get() {
  return nullptr;
}

extern "C" bool is_null(void *ptr) {
  return ptr==nullptr;
}