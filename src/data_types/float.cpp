#include <iostream>
#include <vector>
#include <string>

#include "../common/extension_functions.h"
#include "include.h"


extern "C" void *to_string(float v)
{
  //todo: allow float instead of int only
  return str_to_char(std::to_string((int)v));
}


extern "C" void PrintFloat(float value){
  std::cout << "Printing float.\n";
  std::cout << "Float value: " << value << "\n";
}


extern "C" float UnbugFloat(float value){
    return value;
}




extern "C" float float_Create(char *name, char *scopeless_name, float init_val, AnyVector *notes_vector, int thread_id, char *scope)
{
  pthread_mutex_lock(&clean_scope_mutex);
  NamedClassValues[name] = init_val;
  pthread_mutex_unlock(&clean_scope_mutex);
  
  // move_to_char_pool(strlen(name)+1, name, "StoreOnDemand");

  delete[] name;
  delete[] scopeless_name;

  return 0;
}


extern "C" void StoreOnDemand(char *name, float value){
  
  pthread_mutex_lock(&clean_scope_mutex);
  NamedClassValues[name] = value;
  pthread_mutex_unlock(&clean_scope_mutex);
  move_to_char_pool(strlen(name)+1, name, "StoreOnDemand");
  //delete[] name;
}