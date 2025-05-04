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






extern "C" AnyVector *tuple_New(Scope_Struct *scope_struct, char *type, ...)
{
  // std::cout << "tuple_New. First type: " << type << ".\n";
  va_list args;
  va_start(args, type);

  AnyVector *notes_vector = new AnyVector();  

  bool is_type = false;
  for (int i=0; i<10; i++)
  {
    if (is_type)
    {
      type = va_arg(args, char *);
      // std::cout << "Got tuple type: " << type << ".\n";
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



extern "C" float tuple_Store(char *name, AnyVector *vector, Scope_Struct *scope_struct)
{
  std::cout << "tuple_Store of " << name << ".\n";

  NamedVectors[name] = vector;

  return 0;
}


extern "C" void *tuple_Load(char *name, Scope_Struct *scope_struct){
  std::cout << "tuple_Load"  << ".\n";
  AnyVector *ret = NamedVectors[name];
  move_to_char_pool(strlen(name)+1, name, "free");
  //delete[] tensor_name;
  return ret;
}



extern "C" float tuple_print(Scope_Struct *scope_struct, AnyVector *tuple) {
  // std::cout << "\n";
  tuple->print();
  return 0;
}


extern "C" void *tuple_Create(char *name, char *scopeless_name, AnyVector *init_val, AnyVector *notes_vector, Scope_Struct *scope_struct)
{
  std::cout << "tuple_Create"  << ".\n";


  if (init_val!=nullptr)
    NamedVectors[name] = init_val;


  // std::string tuple_type = "";
  // for (int i=0; i<notes_vector->data->size(); i++)
  // {
  //   if(notes_vector->data_types->at(i)=="float")
  //   {}
  //   if(notes_vector->data_types->at(i)=="string")
  //   {
  //     char *note = notes_vector->get<char *>(i);
  //     if (i==0)
  //       tuple_type = note;
  //     else
  //       tuple_type = tuple_type + "_" + note;
  //   }
  // }
  // std::cout << "Building tuple from type: " << tuple_type << ".\n";


  // auto tuple = std::make_tuple(3.14f, (char *)"Hello", new Tensor());
  // using MyTuple = TupleFromString<"float_str_tensor">;



  return init_val;
}

extern "C" void tuple_MarkToSweep(Scope_Struct *scope_struct, char *name, AnyVector *value) {
  scope_struct->mark_sweep_map->append(name, static_cast<void *>(value), "tuple");
}


void tuple_Clean_Up(std::string name, void *data_ptr) {
  if (data_ptr==nullptr)
    return;
  std::cout << "Tuple cleanup" << ".\n";
}






extern "C" void *tuple_Idx(Scope_Struct *scope_struct, char *name, float _idx)
{
  int idx = (int)_idx;


  
  AnyVector *vec = NamedVectors[name];
  move_to_char_pool(strlen(name)+1, name, "free");

  
  std::string type = vec->data_types->at(idx);
  // std::cout << "tuple_Idx on index " << idx << " for data type " << type << ".\n";

  if (type=="float")
  {
    float* float_ptr = new float(vec->get<float>(idx));
    return (void*)float_ptr;
    // return (void *)vec->get<float>(idx);
  }
  return std::any_cast<void *>((*vec->data)[idx]);
}


extern "C" void *AnyVector_Idx(AnyVector *vec, int idx)
{
 
  std::string type = vec->data_types->at(idx);
  // std::cout << "AnyVector_Idx on index " << idx << " for data type " << type << ".\n";

  if (type=="float")
  {
    float* float_ptr = new float(vec->get<float>(idx));
    return (void*)float_ptr;
    // return (void *)vec->get<float>(idx);
  }
  return std::any_cast<void *>((*vec->data)[idx]);
}


