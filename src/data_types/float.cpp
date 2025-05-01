#include <iostream>
#include <vector>
#include <string>

#include "../codegen/string.h"
#include "../common/extension_functions.h"
#include "../mangler/scope_struct.h"
#include "include.h"




extern "C" void PrintFloat(float value){
  std::cout << "Printing float.\n";
  std::cout << "Float value: " << value << "\n";
}


extern "C" float UnbugFloat(float value){
    return value;
}



extern "C" float print_float(float value){
  std::cout << "print_float: " << std::to_string(value) << ".\n";
  return value;
}


extern "C" float float_Create(char *name, char *scopeless_name, float init_val, AnyVector *notes_vector, Scope_Struct *scope_struct)
{
  pthread_mutex_lock(&clean_scope_mutex);
  NamedClassValues[name] = init_val;
  pthread_mutex_unlock(&clean_scope_mutex);
  
  // move_to_char_pool(strlen(name)+1, name, "float_Store");

  delete[] scopeless_name;

  return 0;
}

extern "C" float float_Load(char *object_var_name, Scope_Struct *scope_struct) {
  
  // std::cout << "Loading float " << object_var_name << ".\n";

  pthread_mutex_lock(&clean_scope_mutex);
  float ret = NamedClassValues[object_var_name];
  pthread_mutex_unlock(&clean_scope_mutex);
  

  return ret;
}


extern "C" void float_Store(char *name, float value, Scope_Struct *scope_struct){
  // std::cout << "Float store: " << name << " on thread id: " << thread_id << ".\n";
  pthread_mutex_lock(&clean_scope_mutex);
  NamedClassValues[name] = value;
  pthread_mutex_unlock(&clean_scope_mutex);
  

}


extern "C" void float_MarkToSweep(Scope_Struct *scope_struct, char *name, float value) {
  scope_struct->mark_sweep_map->append(name, value, "float");
}


extern "C" void StoreOnDemandNoFree(char *name, float value){
  
  pthread_mutex_lock(&clean_scope_mutex);
  NamedClassValues[name] = value;
  pthread_mutex_unlock(&clean_scope_mutex);
}



extern "C" float LoadOnDemandNoFree(char *object_var_name) {
  
  pthread_mutex_lock(&clean_scope_mutex);
  float ret = NamedClassValues[object_var_name];
  pthread_mutex_unlock(&clean_scope_mutex);
  
  return ret;
}