#include <iostream>
#include <vector>
#include "str_vec.h"

extern "C" float PrintStrVec(std::vector<char*> vec)
{
  for (int i=0; i<vec.size(); i++)
    std::cout << vec[i] << "\n";

  return 0;
}