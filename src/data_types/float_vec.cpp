#include <iostream>
#include <vector>

#include "float_vec.h"


extern "C" float PrintFloatVec(std::vector<float> vec)
{

  std::cout << "Float vector:\n[";
  for (int i=0; i<vec.size()-1; i++)
    std::cout << "" << vec[i] << ", ";
  std::cout << "" << vec[vec.size()-1];
  std::cout << "]\n\n";

  return 0;
}