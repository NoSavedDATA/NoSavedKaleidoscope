#include <iostream>
#include <map>
#include <string>
#include <cstdarg>

#include <sstream>
#include <iomanip>

#include "../char_pool/include.h"
#include "../common/extension_functions.h"
// #include "../codegen/random.h"
// #include "../codegen/string.h"
#include "../compiler_frontend/logging.h"
#include "../mangler/scope_struct.h"






extern "C" DT_list *list_New(Scope_Struct *scope_struct, char *type, ...)
{
  // std::cout << "list_New. First type: " << type << ".\n";
  va_list args;
  va_start(args, type);

  DT_list *notes_vector = new DT_list();  

  bool is_type = false;
  for (int i=0; i<1000; i++)
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
      } else if (strcmp(type, "int")==0) {
        // std::cout << "appending float" << ".\n";
        int value = va_arg(args, int);
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



extern "C" float list_Store(char *name, DT_list *vector, Scope_Struct *scope_struct)
{
  // std::cout << "list_Store of " << name << ".\n";

  NamedVectors[name] = vector;

  return 0;
}

extern "C" float list_print(Scope_Struct *scope_struct, DT_list *list) {
  // std::cout << "\n";
  list->print();
  return 0;
}

extern "C" DT_list *list_Load(Scope_Struct *scope_struct, char *name){
  std::cout << "list_Load of " << name << ".\n";
  DT_list *ret = NamedVectors[name];
  // list_print(scope_struct, ret);
  return ret;
}







extern "C" DT_list *list_Create(Scope_Struct *scope_struct, char *name, char *scopeless_name, DT_list *init_val, DT_list *notes_vector)
{
  std::cout << "list_Create"  << ".\n";


  // if (init_val!=nullptr)
  //   NamedVectors[name] = init_val;


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





  return init_val;
}



void list_Clean_Up(void *data_ptr) {
  if (data_ptr==nullptr)
    return;
  // std::cout << "list cleanup" << ".\n";
}








extern "C" void *list_Idx(Scope_Struct *scope_struct, DT_list *vec, int idx)
{
  std::cout << "INDEX AT " << idx << ".\n";
  

  
  std::string type = vec->data_types->at(idx);
  // std::cout << "list_Idx on index " << idx << " for data type " << type << ".\n";

  if (type=="float")
  {
    float* float_ptr = new float(vec->get<float>(idx));
    return (void*)float_ptr;
    // return (void *)vec->get<float>(idx);
  }
  if (type=="int")
  {
    int* ptr = new int(vec->get<int>(idx));
    return (void*)ptr;
  }
  return std::any_cast<void *>((*vec->data)[idx]);
}



extern "C" void *assign_wise_list_Idx(DT_list *vec, int idx)
{
 
  std::string type = vec->data_types->at(idx);
  // std::cout << "DT_list_Idx on index " << idx << " for data type " << type << ".\n";

  if (type=="float")
  {
    float* float_ptr = new float(vec->get<float>(idx));
    return (void*)float_ptr;
    // return (void *)vec->get<float>(idx);
  }
  if (type=="int")
  {
    int* ptr = new int(vec->get<int>(idx));
    return (void*)ptr;
    // return (void *)vec->get<float>(idx);
  }

  return std::any_cast<void *>((*vec->data)[idx]);
}


