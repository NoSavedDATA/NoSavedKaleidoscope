#include <iostream>
#include <vector>
#include <map>

#include "../codegen/random.h"
#include "../compiler_frontend/global_vars.h"
#include "../mangler/scope_struct.h"
#include "include.h"





extern "C" void *int_vec_Create(Scope_Struct *scope_struct, char *name, char *scopeless_name, void *init_val, DT_list *notes_vector)
{
  // std::cout << "int_vec_Create" << ".\n";

  if (init_val!=nullptr)
  {
    DT_int_vec *vec = static_cast<DT_int_vec *>(init_val);
    NamedIntVecs[name] = vec;
  }


  return init_val;
}

extern "C" DT_int_vec *int_vec_Load(Scope_Struct *scope_struct, char *object_var_name) {
  // std::cout << "Load int_vec On Demand var to load: " << object_var_name << "\n";
  // std::cout << "scope: " << scope_struct->scope << ".\n";
  
  DT_int_vec *vec = NamedIntVecs[object_var_name];
  // std::cout << "vec size is " << vec->size << ".\n";

  return vec;
}

extern "C" int int_vec_Store(char *name, DT_int_vec *value, Scope_Struct *scope_struct){
  // std::cout << "STORING " << name << " on demand as int vec type" << ".\n";

  NamedIntVecs[name] = value;
  return 0;
}

 
void int_vec_Clean_Up(void *data_ptr) {

}


extern "C" int int_vec_Store_Idx(DT_int_vec *vec, int idx, int value, Scope_Struct *scope_struct){
  // std::cout << "int_vec_Store_Idx[" << idx << "]: " << value << ".\n";

  vec->vec[idx] = value;

  return 0;
}







extern "C" DT_int_vec *arange_int(Scope_Struct *scope_struct, int begin, int end) {
  // TODO: turn into python like expression [0]*size

  DT_int_vec *vec = new DT_int_vec(end-begin);
  int c=0;
  for(int i=begin; i<end; ++i)
  {
    vec->vec[c] = i;
    c++;
  }
  return vec; 
}


extern "C" DT_int_vec *zeros_int(Scope_Struct *scope_struct, int size) {

  DT_int_vec *vec = new DT_int_vec(size);
  for(int i=0; i<size; ++i)
    vec->vec[i] = 0;

  return vec;
}


extern "C" DT_int_vec *ones_int(Scope_Struct *scope_struct, int size) {
  DT_int_vec *vec = new DT_int_vec(size);
  for(int i=0; i<size; ++i)
    vec->vec[i] = 1;
   

  return vec;
}


extern "C" int int_vec_Idx(Scope_Struct *scope_struct, DT_int_vec *vec, int idx)
{
  // std::cout << "int_vec_Idx on idx " << idx << " for the vector " << vec << ".\n";

  // std::cout << "Loaded vec" << ".\n";
  int ret = vec->vec[idx];
  // std::cout << "returning" << ".\n"; 
  // std::cout << "got: " << ret << ".\n";
  return ret;
}

extern "C" int int_vec_Idx_num(Scope_Struct *scope_struct, DT_int_vec *vec, int _idx)
{
  int idx = (int) _idx;
  // std::cout << "int_vec_Idx_num on idx " << idx << ".\n";
  // std::cout << "vec idx " << idx << " got: " << vec->vec[idx] << ".\n";


  return vec->vec[idx];
}



extern "C" int int_vec_CalculateIdx(DT_int_vec *vec, int first_idx, ...) {

  if (first_idx<0)
    first_idx = vec->size+first_idx;


  return first_idx;
}


extern "C" DT_int_vec *int_vec_CalculateSliceIdx(DT_int_vec *vec, int first_idx, ...) {

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

 
  return slices;
}



extern "C" DT_int_vec *int_vec_Slice(Scope_Struct *scope_struct, DT_int_vec *vec, DT_int_vec *slices) {

  int start=slices->vec[0], end=slices->vec[1];

  int size = end-start;

  DT_int_vec *out_vec = new DT_int_vec(size);

  for (int i=0; i<size; ++i)
    out_vec->vec[i] = vec->vec[start+i];

  return out_vec;
}




extern "C" int int_vec_first_nonzero(Scope_Struct *scope_struct, DT_int_vec *vec) { 
  // std::cout << "First non zero" << ".\n";
  int idx = -1;
  for (int i=0; i<vec->size; i++)
    if (vec->vec[i]!=0)
    {
      idx = i;
      break;
    }

  return idx;
}



extern "C" int int_vec_print(Scope_Struct *scope_struct, DT_int_vec *vec) {
  // std::cout << "int_vec_print" << ".\n";
  std::cout << "[";
  for (int i=0; i<vec->size-1; i++)
    std::cout << vec->vec[i] << ", ";

  std::cout << vec->vec[vec->size-1] << "]" << "\n";
  return 0;
}









extern "C" DT_int_vec *int_vec_Split_Parallel(Scope_Struct *scope_struct, DT_int_vec *vec)
{
    int threads_count = scope_struct->asyncs_count;
    int thread_id = scope_struct->thread_id-1;
    // std::cout << "SPLITTING int VEC"  << ".\n";
    // std::cout << "Threads count: " << scope_struct->asyncs_count << ".\n";
    // std::cout << "Threads id: " << scope_struct->thread_id << ".\n\n";

    int vec_size = vec->size;
    int segment_size;

    segment_size = ceilf(vec_size/threads_count);

    // std::cout << "SEGMENT SIZE IS " << segment_size << ".\n";

    int size = segment_size;
    if((thread_id+1)==threads_count)
      size = vec_size - segment_size*thread_id;
      

    DT_int_vec *out_vector = new DT_int_vec(size);


    int c=0;
    for (int i = segment_size*thread_id; i<segment_size*(thread_id+1) && i<vec->size; ++i)
    {
      out_vector->vec[c] = vec->vec[i];
      c++;
    }

    return out_vector;
}


extern "C" DT_int_vec *int_vec_Split_Strided_Parallel(Scope_Struct *scope_struct, DT_int_vec *vec)
{
    int threads_count = scope_struct->asyncs_count;
    int thread_id = scope_struct->thread_id-1;
    // std::cout << "SPLITTING int VEC"  << ".\n";
    // std::cout << "Threads count: " << scope_struct->asyncs_count << ".\n";
    // std::cout << "Threads id: " << scope_struct->thread_id << ".\n\n";

    int vec_size = vec->size;
    int segment_size;

    segment_size = ceilf(vec_size/threads_count);

    // std::cout << "SEGMENT SIZE IS " << segment_size << ".\n";

    int size = segment_size;
    if((thread_id+1)==threads_count)
      size = vec_size - segment_size*thread_id;
      

    DT_int_vec *out_vector = new DT_int_vec(size);


    int c=0;

    for (int i = thread_id; c<segment_size && i<vec->size; i=i+threads_count)
    {
      out_vector->vec[c] = vec->vec[i];
      c++;
    }

    return out_vector;
}

extern "C" int int_vec_size(Scope_Struct *scope_struct, DT_int_vec *vec) {
  return vec->size;
}