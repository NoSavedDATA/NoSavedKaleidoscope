#include <iostream>
#include <vector>
#include <string>

#include "../codegen/string.h"
#include "../common/extension_functions.h"
#include "../mangler/scope_struct.h"
#include "include.h"









extern "C" int int_Create(Scope_Struct *scope_struct, char *name, char *scopeless_name, int init_val, DT_list *notes_vector)
{
  
  // std::cout << "Storing int " << init_val << ".\n";
  // pthread_mutex_lock(&clean_scope_mutex);
  NamedInts[name] = init_val;
  // pthread_mutex_unlock(&clean_scope_mutex);
  
  // move_to_char_pool(strlen(name)+1, name, "int_Store");


  return init_val;
}

extern "C" int int_Load(Scope_Struct *scope_struct, char *object_var_name) {
  


  // std::cout << "Loading int " << object_var_name << ".\n";
  // pthread_mutex_lock(&clean_scope_mutex);
  int ret = NamedInts[object_var_name];
  // pthread_mutex_unlock(&clean_scope_mutex);
  

  std::cout << "Loading int: " << ret << ".\n";


  return ret;
}


extern "C" void int_Store(char *name, int value, Scope_Struct *scope_struct) {
  // std::cout << "STORE OF " << name << " ON THREAD " << scope_struct->thread_id << ".\n";
  // pthread_mutex_lock(&clean_scope_mutex);
  NamedInts[name] = value;
  // pthread_mutex_unlock(&clean_scope_mutex);
}



void int_Clean_Up(void *data_ptr) {
  // pthread_mutex_lock(&clean_scope_mutex);
  // pthread_mutex_unlock(&clean_scope_mutex);
}

