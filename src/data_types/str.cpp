#include <iostream>

#include "../common/extension_functions.h"
#include "str.h"


extern "C" float PrintStr(char* value){
  std::cout << "Str: " << value << "\n";
  return 0;
}


extern "C" float *split_str_to_float(char *in_string, int gather_position)
{
  std::vector<std::string> splitted = split_str(in_string, '/');

  float * ret = new float[1];

  if(gather_position<0)
    gather_position = splitted.size()+gather_position;

  ret[0] = std::stof(splitted[gather_position]);

  return ret;
}




extern "C" void *cat_str_float(char *c, float v)
{

  std::string s = c;
  std::string tmp = std::to_string((int)v);

  s = s + c;

  return str_to_char(s);
}