#include <string>
#include <iostream>

#include "../common/extension_functions.h"
#include "../data_types/include.h"


extern "C" char * FirstArgOnDemand(char *first_arg, char *pre_dotc, char *_class, char *method, int nested_function, int isSelf, int isAttribute)
{

  std::string _first_arg = first_arg;
  std::string pre_dot = pre_dotc;

  delete[] pre_dotc;


  //std::cout << "\n\n\nIncoming first arg: " << first_arg << " from pre-dot: " << pre_dot << ";\n   class: " << _class << ", method: " << method << "\n   is nested: " << nested_function <<".\n";
  //std::cout << "   is self: " << isSelf << ", is attribute: " << isAttribute << "\n\n\n";

  //std::cout << "\n\n\n";
  //for(auto& pair : NamedObjects)
  //  std::cout << "NamedObjects: " << pair.first << ": " << pair.second<< "\n";

  //TODO:
  
  if (!isSelf && isAttribute)
  {
    std::string ret = NamedObjects[pre_dot];
    //std::cout << "\nReturning " << ret << "\n\n\n\n";
    return str_to_char(ret);
  }
  
  
  
  
  if (pre_dot!="self")
  {
    if (nested_function)
      _first_arg = _first_arg+pre_dot;
    else
      _first_arg = pre_dot; 
  }

  return str_to_char(_first_arg);
}