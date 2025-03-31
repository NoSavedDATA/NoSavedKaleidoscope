
#include "../codegen/string.h"
#include "include.h"


extern "C" void InstantiateObject(char *scope, char *obj_name)
{
  //std::cout << "\n\n\n\nInstantiateObject of: " << scope << obj_name << "\n\n\n";
  std::string _obj_name = obj_name;

  NamedObjects[scope+_obj_name] = _obj_name + RandomString(13);
  //std::cout << "Saving " << NamedObjects[scope+_obj_name]  << "\n\n";
}


extern "C" char *objHash(char *scope, char *obj_name)
{
  std::string _obj_name = obj_name;
  std::string ret = NamedObjects[scope+_obj_name];
  return str_to_char(ret);
}


extern "C" char *LoadObject(char *obj_name)
{
  //std::cout << "LOADING OBJECT FROM " << obj_name << "\n";
  std::string ret = NamedObjects[obj_name];
  delete[] obj_name;
  //std::cout << "Load object of: " << ret << "\n";
  return str_to_char(ret);
}