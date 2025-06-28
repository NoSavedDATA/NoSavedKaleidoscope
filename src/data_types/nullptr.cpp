#include <iostream>
#include <vector>
#include <map>

#include "../codegen/random.h"
#include "../mangler/scope_struct.h"
#include "include.h"



extern "C" void *nullptr_get() {
  return nullptr;
}

extern "C" void check_is_null(void *ptr) {
  if (ptr!=nullptr)
    printf("Not null");

}