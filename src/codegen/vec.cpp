#include <vector>

#include "../data_types/include.h"
#include "../mangler/scope_struct.h"
#include "include.h"



extern "C" float first_nonzero(Scope_Struct *scope_struct)
{
  
  std::string self = scope_struct->first_arg;
  std::vector<float> vec;
  vec = ClassFloatVecs[self];
  
  /*
  std::cout << "[";
  for (int i=0; i<vec.size(); i++)
    std::cout << vec[i] << ", ";
  std::cout << "]" << "\n";
  */


  float idx = -1;
  for (int i=0; i<vec.size(); i++)
    if (vec[i]!=0)
    {
      idx = i;
      break;
    }

  return idx;
}