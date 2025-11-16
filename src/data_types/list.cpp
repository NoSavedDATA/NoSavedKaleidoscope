#include <iostream>
#include <map>
#include <string>
#include <cstdarg>

#include <sstream>
#include <iomanip>

#include "../char_pool/include.h"
#include "../common/extension_functions.h"
#include "../codegen/random.h"
// #include "../codegen/string.h"
#include "../compiler_frontend/global_vars.h"
#include "../compiler_frontend/logging.h"
#include "../compiler_frontend/logging_v.h"
#include "../mangler/scope_struct.h"
#include "../pool/include.h"

#include "float_vec.h"

#include "codegen_notes.h"





extern "C" DT_list *list_New(Scope_Struct *scope_struct, char *type, ...)
{
  va_list args;
  va_start(args, type);

  // DT_list *notes_vector = new DT_list();  
  DT_list *notes_vector = newT<DT_list>(scope_struct, "list");

  bool is_type = false;
  for (int i=0; i<1000; i++)
  {
    if (is_type)
    {
      type = va_arg(args, char *);
      // std::cout << "Got list type: " << type << ".\n";
      is_type = false;
      if (!strcmp(type, "TERMINATE_VARARG"))
        break;
    } else {   
      if (!strcmp(type, "float")) {
        float value = va_arg(args, float);
        notes_vector->append(value, type);
      } else if (!strcmp(type, "int")) {
        int value = va_arg(args, int);
        notes_vector->append(value, type);
      } else if (!strcmp(type, "str")) {
        char *value = va_arg(args, char *);
        notes_vector->append(value, type);
      } else if (!strcmp(type, "bool")) {
        bool value = va_arg(args, bool);
        notes_vector->append(value, type);
      } else {
        void *value = va_arg(args, void *);
        notes_vector->append(std::any(value), type);
      }
      is_type = true;
    }
  }
  va_end(args);
  
  // std::cout << "" << ".\n";
  // notes_vector->print();
  
  return notes_vector;
}



extern "C" void list_append_int(Scope_Struct *scope_struct, DT_list *list, int x) {
  list->append(x, "int");
}

extern "C" void list_append_float(Scope_Struct *scope_struct, DT_list *list, float x) {
  list->append(x, "float");
}

extern "C" void list_append_bool(Scope_Struct *scope_struct, DT_list *list, bool x) {
  list->append(x, "bool");
}

extern "C" float list_append(Scope_Struct *scope_struct, DT_list *list, void *x, char *type) {
  // std::cout << "Appending x of type " << type << ".\n";  
  list->append(std::any(x), type);
  // std::cout << "new list size: " << list->size << ".\n";
  return 0;
}




extern "C" float list_print(Scope_Struct *scope_struct, DT_list *list) {
  std::cout << "print list " << list << "\n";
  list->print();
  return 0;
}


extern "C" float tuple_print(Scope_Struct *scope_struct, DT_list *list) {
  // std::cout << "\n";
  list->print();
  return 0;
}







extern "C" DT_list *list_Create(Scope_Struct *scope_struct, char *name, char *scopeless_name, DT_list *init_val, DT_list *notes_vector)
{
  if (init_val==nullptr)
    init_val = newT<DT_list>(scope_struct, "list");


  return init_val;
}


void DT_list_shuffle_pair(std::vector<std::any>& a,
                  std::vector<std::string>& b)
{
    const size_t n = a.size();
    if (b.size() != n) return; // or throw

    std::vector<size_t> idx(n);
    for (size_t i = 0; i < n; ++i) idx[i] = i;

    
    std::shuffle(idx.begin(), idx.end(), MAIN_PRNG);

    std::vector<std::any> a2(n);
    std::vector<std::string> b2(n);

    for (size_t i = 0; i < n; ++i) {
        a2[i] = std::move(a[idx[i]]);
        b2[i] = std::move(b[idx[i]]);
    }

    a = std::move(a2);
    b = std::move(b2);
}

extern "C" float list_shuffle(Scope_Struct *scope_struct, DT_list *list) {
  DT_list_shuffle_pair(*list->data, *list->data_types);
  return 0;
}

extern "C" int list_size(Scope_Struct *scope_struct, DT_list *list) {
  return list->size;
}

extern "C" DT_float_vec *list_as_float_vec(Scope_Struct *scope_struct, DT_list *list) {
  int size = list->size;

  DT_float_vec *vec = newT<DT_float_vec>(scope_struct, "float_vec");
  vec->New(size);

  for (int i=0; i<size; ++i)
    vec->vec[i] = list->get<float>(i);

  return vec;
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







extern "C" int to_int(Scope_Struct *scope_struct, void *ptr) {
  return *static_cast<int*>(ptr);
}

extern "C" float to_float(Scope_Struct *scope_struct, void *ptr) {
  return *static_cast<float*>(ptr);
}


extern "C" Vec_Slices *list_CalculateSliceIdx(DT_list *vec, int first_idx, ...) {


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

  Vec_Slices *vec_slices = new Vec_Slices();

  DT_int_vec slices = DT_int_vec(2);
  slices.vec[0] = first_idx;
  slices.vec[1] = second_idx;

  vec_slices->push_back(slices);

 
  return vec_slices;
}



extern "C" DT_list *list_Slice(Scope_Struct *scope_struct, DT_list *vec, Vec_Slices *slices) {

  int start = slices->slices[0].vec[0], end=slices->slices[0].vec[1];

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
  // std::cout << "DT_list_Idx at index " << idx << " for data type " << type << ".\n";

  if (type=="float")
  {
    float* float_ptr = new float(vec->get<float>(idx));
    return (void*)float_ptr;
  }
  if (type=="int")
  {
    int* ptr = new int(vec->get<int>(idx));
    return (void*)ptr;
  }

  return std::any_cast<void *>((*vec->data)[idx]);
}




extern "C" float int_list_Store_Idx(DT_list *list, int idx, int value, Scope_Struct *scope_struct) {
  std::any& slot = (*list->data)[idx];
  slot = value;
  return 0;
}

extern "C" float float_list_Store_Idx(DT_list *list, int idx, float value, Scope_Struct *scope_struct) {
  std::any& slot = (*list->data)[idx];
  slot = value;
  return 0;
}

extern "C" float list_Store_Idx(DT_list *list, int idx, void *value, Scope_Struct *scope_struct) {
  std::any& slot = (*list->data)[idx];
  slot = value;
  return 0;
}



extern "C" DT_list *zip(Scope_Struct *scope_struct, DT_list *list, ...) {


  va_list args;
  va_start(args, list);


  std::vector<DT_list*> lists = {list};


  DT_list *new_list = va_arg(args, DT_list*);
  if (new_list==nullptr)
  {
    LogError(scope_struct->code_line, "Zipping a single list");
    return list;
  }

  do {
    if (new_list!=nullptr)
      lists.push_back(new_list);
    new_list = va_arg(args, DT_list*);
  } while(new_list!=nullptr);
  
  // std::cout << "got " << lists.size() << " list " << ".\n";


  int list_size = list->size;
  for (int i=1; i<lists.size(); ++i) {
    if (lists[i]->size!=list_size) {
      LogError(scope_struct->code_line, "Zipping lists of different size.");
      return list;
    }
  }

 
  DT_list *out_list = new DT_list();
  for (int i=0; i<list_size; ++i)
  { 
    DT_list *inner_list = new DT_list();
    
    for (DT_list *inner : lists) {
      if(inner->data_types->at(i)=="float")
        inner_list->append(inner->get<float>(i), "float");
      else if(inner->data_types->at(i)=="int")
        inner_list->append(inner->get<int>(i), "int");
      else if(inner->data_types->at(i)=="bool")
        inner_list->append(inner->get<bool>(i), "bool");
      else if(inner->data_types->at(i)=="str")
        inner_list->append(inner->get<char *>(i), "str");
      else
      {
        inner_list->append(std::any(inner->get<void *>(i)), inner->data_types->at(i));
      }
    } 
    out_list->append(std::any(static_cast<void *>(inner_list)), "list");
  }


  return out_list;
}


extern "C" void *list_Idx(Scope_Struct *scope_struct, DT_list *vec, int idx)
{
  std::string type = vec->data_types->at(idx);
  // std::cout << "list_Idx on index " << idx << " & recover data type " << type << ".\n";

  if (type=="float") {
    float* float_ptr = new float(vec->get<float>(idx));
    return (void*)float_ptr;
  } else if (type=="int") {
    int* ptr = new int(vec->get<int>(idx));
    return static_cast<void*>(ptr);
  } else if (type=="bool") {
    bool* ptr = new bool(vec->get<bool>(idx));
    return static_cast<void*>(ptr); 
  } else if (type=="str") {
    char *c = vec->get<char*>(idx);
    int len = strlen(c);
    char *c_copy = allocate<char>(scope_struct, len+1, "str");
    c_copy[len] = '\0';
    memcpy(c_copy, c, len);
    // std::cout << "get str " << vec->get<char*>(idx) << ".\n";
    // return static_cast<char*>(ptr);
    return c_copy;
  } else {
    if (idx >= (*vec->data).size()) {
      LogErrorC(scope_struct->code_line, "list index out of range.");
      return nullptr;
    }
    return std::any_cast<void *>((*vec->data)[idx]);
  }
}



extern "C" void *tuple_Idx(Scope_Struct *scope_struct, DT_list *vec, int idx)
{
  // std::cout << "INDEX AT " << idx << ".\n";
    
  std::string type = vec->data_types->at(idx);
  // std::cout << "tuple_Idx on index " << idx << " for data type " << type << ".\n";

  if (type=="float") {
    float* float_ptr = new float(vec->get<float>(idx));
    return (void*)float_ptr;
  } else if (type=="int") {
    int* ptr = new int(vec->get<int>(idx));
    return static_cast<void*>(ptr);
  } else if (type=="bool") {
    bool* ptr = new bool(vec->get<bool>(idx));
    return static_cast<void*>(ptr); 
  } else
  {
    return std::any_cast<void *>((*vec->data)[idx]);
  }
}