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





extern "C" float tuple_New(Scope_Struct *scope_struct, char *type, ...)
{


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
    } else 
    {
      
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
  // std::cout << "Printing notes_vector" << ".\n";

  notes_vector->print();
  

  return 0;
}


extern "C" float tuple_Create(char *name, char *scopeless_name, float init_val, AnyVector *notes_vector, Scope_Struct *scope_struct)
{
  std::cout << "tuple_Create"  << ".\n";

  std::string tuple_type = "";
  for (int i=0; i<notes_vector->data->size(); i++)
  {
    if(notes_vector->data_types->at(i)=="float")
    {}
    if(notes_vector->data_types->at(i)=="string")
    {
      char *note = notes_vector->get<char *>(i);
      if (i==0)
        tuple_type = note;
      else
        tuple_type = tuple_type + "_" + note;
    }
  }

  std::cout << "Building tuple from type: " << tuple_type << ".\n";

  // auto tuple = std::make_tuple(3.14f, (char *)"Hello", new Tensor());
  // using MyTuple = TupleFromString<"float_str_tensor">;



  return 0;
}