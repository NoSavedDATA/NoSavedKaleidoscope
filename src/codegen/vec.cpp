#include <vector>

#include "../data_types/include.h"
#include "include.h"



extern "C" float first_nonzero(char *self)
{
  //std::cout << "first_nonzero call of: " << self <<"\n";

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

  delete[] self;
  return idx;
}