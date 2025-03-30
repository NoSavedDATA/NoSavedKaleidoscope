#include <iostream>
#include <map>
#include <string>

#include "../common/extension_functions.h"
#include "include.h"

std::map<std::string, std::string> AuxRandomStrs;



  
extern "C" float str_Create(char *name, char *scopeless_name, char *init_val, AnyVector *notes_vector, int thread_id, char *scope) {

  // std::cout << "Creating string"  << ".\n";
  // std::cout << "Val: " << init_val << ".\n";
  
  pthread_mutex_lock(&clean_scope_mutex);
  NamedStrs[name] = init_val;
  //std::cout << "Store " << value << " at " << name << "\n";
  pthread_mutex_unlock(&clean_scope_mutex);
  move_to_char_pool(strlen(name)+1, name, "free");
  move_to_char_pool(strlen(name)+1, scopeless_name, "free");
  //delete[] name;

  return 0;
}


extern "C" float StoreStrOnDemand(char *name, char *value){
  
  //NamedStrs[name] = CopyString(value); //TODO: Break?
  
  pthread_mutex_lock(&clean_scope_mutex);
  NamedStrs[name] = value;
  //std::cout << "Store " << value << " at " << name << "\n";
  pthread_mutex_unlock(&clean_scope_mutex);
  move_to_char_pool(strlen(name)+1, name, "free");
  //delete[] name;

  return 0;
}
extern "C" void *LoadStrOnDemand(char *name){
  
  //char *ret = CopyString(NamedStrs[name]);
  
  pthread_mutex_lock(&clean_scope_mutex);
  char *ret = NamedStrs[name];
  pthread_mutex_unlock(&clean_scope_mutex);
  move_to_char_pool(strlen(name)+1, name, "free");
  //delete[] name;

  return ret;
}





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