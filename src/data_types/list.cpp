#include <iostream>
#include <map>
#include <string>
#include <cstdarg>

#include <sstream>
#include <iomanip>

#include "../common/extension_functions.h"
#include "../codegen/random.h"
#include "../codegen/string.h"
#include "../compiler_frontend/logging.h"
#include "../mangler/scope_struct.h"
#include "../tensor/tensor_dim_functions.h"
#include "include.h"






extern "C" data_type_list *list_New(Scope_Struct *scope_struct, char *type, ...)
{
  // std::cout << "list_New. First type: " << type << ".\n";
  va_list args;
  va_start(args, type);

  data_type_list *notes_vector = new data_type_list();  

  bool is_type = false;
  for (int i=0; i<10; i++)
  {
    if (is_type)
    {
      type = va_arg(args, char *);
      // std::cout << "Got list type: " << type << ".\n";
      is_type = false;
      if (strcmp(type, "TERMINATE_VARARG")==0)
        break;
    } else {   
      if (strcmp(type, "float")==0)
      {
        // std::cout << "appending float" << ".\n";
        float value = va_arg(args, float);
        notes_vector->append(value, type);
      } else {
        // std::cout << "appending void *: " << type << ".\n";
        void *value = va_arg(args, void *);
        // std::cout << "decoded"  << ".\n";
        notes_vector->append(std::any(value), type);
        // std::cout << "appended"  << ".\n";
      }
      is_type = true;
    }
  }
  va_end(args);
  
  // std::cout << "" << ".\n";
  // notes_vector->print();
  
  return notes_vector;
}



extern "C" float list_Store(char *name, data_type_list *vector, Scope_Struct *scope_struct)
{
  // std::cout << "list_Store of " << name << ".\n";

  NamedVectors[name] = vector;

  return 0;
}


extern "C" data_type_list *list_Load(char *name, Scope_Struct *scope_struct){
  std::cout << "list_Load"  << ".\n";
  data_type_list *ret = NamedVectors[name];
  //delete[] tensor_name;
  return ret;
}



extern "C" float list_print(Scope_Struct *scope_struct, data_type_list *list) {
  // std::cout << "\n";
  list->print();
  return 0;
}


extern "C" float list_test(Scope_Struct *scope_struct, data_type_list *list) {
  std::cout << "REACHED list TEST" << ".\n";
  return 0;
}


extern "C" data_type_list *list_Create(char *name, char *scopeless_name, data_type_list *init_val, data_type_list *notes_vector, Scope_Struct *scope_struct)
{
  std::cout << "list_Create"  << ".\n";


  if (init_val!=nullptr)
    NamedVectors[name] = init_val;


  // std::string list_type = "";
  // for (int i=0; i<notes_vector->data->size(); i++)
  // {
  //   if(notes_vector->data_types->at(i)=="float")
  //   {}
  //   if(notes_vector->data_types->at(i)=="string")
  //   {
  //     char *note = notes_vector->get<char *>(i);
  //     if (i==0)
  //       list_type = note;
  //     else
  //       list_type = list_type + "_" + note;
  //   }
  // }
  // std::cout << "Building list from type: " << list_type << ".\n";


  // auto list = std::make_list(3.14f, (char *)"Hello", new Tensor());
  // using Mylist = listFromString<"float_str_tensor">;



  return init_val;
}

extern "C" void list_MarkToSweep(Scope_Struct *scope_struct, char *name, data_type_list *value) {
  scope_struct->mark_sweep_map->append(name, static_cast<void *>(value), "list");
}


void list_Clean_Up(std::string name, void *data_ptr) {
  if (data_ptr==nullptr)
    return;
  std::cout << "list cleanup" << ".\n";
}






// extern "C" void *list_Idx(Scope_Struct *scope_struct, char *name, float _idx)
// {
//   int idx = (int)_idx;


  
//   data_type_list *vec = NamedVectors[name];
//   move_to_char_pool(strlen(name)+1, name, "free");

  
//   std::string type = vec->data_types->at(idx);
//   // std::cout << "list_Idx on index " << idx << " for data type " << type << ".\n";

//   if (type=="float")
//   {
//     float* float_ptr = new float(vec->get<float>(idx));
//     return (void*)float_ptr;
//     // return (void *)vec->get<float>(idx);
//   }
//   return std::any_cast<void *>((*vec->data)[idx]);
// }


extern "C" void *list_Idx(data_type_list *vec, int idx)
{
 
  std::string type = vec->data_types->at(idx);
  // std::cout << "data_type_list_Idx on index " << idx << " for data type " << type << ".\n";

  if (type=="float")
  {
    float* float_ptr = new float(vec->get<float>(idx));
    return (void*)float_ptr;
    // return (void *)vec->get<float>(idx);
  }

  return std::any_cast<void *>((*vec->data)[idx]);
}


