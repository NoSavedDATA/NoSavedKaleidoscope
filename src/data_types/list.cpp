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
#include "../compiler_frontend/global_vars.h"
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




extern "C" float list_print(Scope_Struct *scope_struct, DT_list *list) {
  // std::cout << "\n";
  list->print();
  return 0;
}






extern "C" DT_list *list_Create(Scope_Struct *scope_struct, char *name, char *scopeless_name, DT_list *init_val, DT_list *notes_vector)
{
  std::cout << "list_Create"  << ".\n";

  if (init_val==nullptr)
    init_val = new DT_list();  

  return init_val;
}



void list_Clean_Up(void *data_ptr) {
  if (data_ptr==nullptr)
    return;
  // std::cout << "list cleanup" << ".\n";
}





extern "C" int list_CalculateIdx(DT_list *list, int first_idx, ...) {
  if (first_idx<0)
    first_idx = list->size+first_idx;
  return first_idx;
}




extern "C" void *list_Idx(Scope_Struct *scope_struct, DT_list *vec, int idx)
{
  // std::cout << "INDEX AT " << idx << ".\n";
    
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


extern "C" DT_int_vec *list_CalculateSliceIdx(DT_list *vec, int first_idx, ...) {

  DT_int_vec *slices;

  int second_idx;

  va_list args;
  va_start(args, first_idx);
  va_arg(args, int); // get terminate symbol
  second_idx = va_arg(args, int); // get second idx
  va_end(args);


  int size = vec->size;
  if (first_idx<0)
    first_idx = size + first_idx;
  if (second_idx<0)
    second_idx = size + second_idx;

  slices = new DT_int_vec(2);
  slices->vec[0] = first_idx;
  slices->vec[1] = second_idx;

  std::cout << "Slice list from " << first_idx << " to " << second_idx << ".\n";
 
  return slices;
}



extern "C" DT_list *list_Slice(Scope_Struct *scope_struct, DT_list *vec, DT_int_vec *slices) {

  int start=slices->vec[0], end=slices->vec[1];

  if (end==COPY_TO_END_INST)
    end = vec->size;


  int size = end-start;

  DT_list *out_vec = new DT_list();

  for (int i=0; i<size; ++i)
  {
    int idx = start+i;
    std::string data_type = vec->data_types->at(idx);

    if (data_type=="float")
      out_vec->append(vec->get<float>(idx), data_type);
    if (data_type=="int")
      out_vec->append(vec->get<int>(idx), data_type);
    else 
      out_vec->append(vec->get<void *>(idx), data_type);

  }

  return out_vec;
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


