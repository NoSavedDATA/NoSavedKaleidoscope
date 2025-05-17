#include <iostream>
#include <vector>
#include <map>

#include "../codegen/random.h"
#include "../mangler/scope_struct.h"
#include "include.h"



DT_float_vec::DT_float_vec(int size) : size(size) {
  vec = (float*)malloc(size*sizeof(float));
}


extern "C" void *float_vec_Create(Scope_Struct *scope_struct, char *name, char *scopeless_name, void *init_val, DT_list *notes_vector)
{
  // std::cout << "float_vec_Create" << ".\n";

  if (init_val!=nullptr)
  {
    DT_float_vec *vec = static_cast<DT_float_vec *>(init_val);
    ClassFloatVecs[name] = vec;
  }


  return init_val;
}

extern "C" DT_float_vec *float_vec_Load(Scope_Struct *scope_struct, char *object_var_name) {
  // std::cout << "Load float_vec On Demand var to load: " << object_var_name << "\n";
  // std::cout << "scope: " << scope_struct->scope << ".\n";
  
  DT_float_vec *vec = ClassFloatVecs[object_var_name];

  return vec;
}

extern "C" float float_vec_Store(char *name, DT_float_vec *value, Scope_Struct *scope_struct){
  // std::cout << "STORING " << name << " on demand as float vec type" << ".\n";

  ClassFloatVecs[name] = value;
  return 0;
}

 
void float_vec_Clean_Up(void *data_ptr) {

}


extern "C" float float_vec_Store_Idx(char *name, float idx, float value, Scope_Struct *scope_struct){
  // std::cout << "float_vec_Store_Idx" << ".\n";
  //std::cout << "STORING " << self << "." << object_var_name << " on demand as float vec type" << ".\n";

  DT_float_vec *vec = ClassFloatVecs[name];
  vec->vec[(int)idx] = value;

  return 0;
}







extern "C" DT_float_vec *arange_float(Scope_Struct *scope_struct, float begin, float end) {
  // TODO: turn into python like expression [0]*size

  DT_float_vec *vec = new DT_float_vec(end-begin);
  int c=0;
  for(int i=begin; i<end; ++i)
  {
    vec->vec[c] = i;
    c++;
  }
  return vec; 
}


extern "C" DT_float_vec *zeros_vec(Scope_Struct *scope_struct, float size) {
  DT_float_vec *vec = new DT_float_vec(size);
  for(int i=0; i<size; ++i)
    vec->vec[i] = 0;
   

  return vec;
}


extern "C" DT_float_vec *ones_vec(Scope_Struct *scope_struct, float size) {
  DT_float_vec *vec = new DT_float_vec(size);
  for(int i=0; i<size; ++i)
    vec->vec[i] = 1;
   

  return vec;
}


extern "C" float float_vec_Idx(Scope_Struct *scope_struct, char *vec_name, float _idx)
{
  int idx = (int) _idx;
  // std::cout << "float_vec_Idx on idx " << idx << " for the vector " << vec_name << ".\n";

  DT_float_vec *vec = ClassFloatVecs[vec_name];
  // std::cout << "Loaded vec" << ".\n";
  float ret = vec->vec[idx];
  // std::cout << "got: " << ret << ".\n";
  delete[] vec_name;
  // std::cout << "returning" << ".\n"; 
  return ret;
}

extern "C" float float_vec_Idx_num(Scope_Struct *scope_struct, DT_float_vec *vec, float _idx)
{
  int idx = (int) _idx;
  // std::cout << "float_vec_Idx_num on idx " << idx << ".\n";
  // std::cout << "vec idx " << idx << " got: " << vec->vec[idx] << ".\n";


  return vec->vec[idx];
}



extern "C" float float_vec_CalculateIdx(char *data_name, float first_idx, ...) {
  return first_idx;
}




extern "C" float float_vec_first_nonzero(Scope_Struct *scope_struct, DT_float_vec *vec) { 
  // std::cout << "First non zero" << ".\n";
  float idx = -1;
  for (int i=0; i<vec->size; i++)
    if (vec->vec[i]!=0)
    {
      idx = i;
      break;
    }

  return idx;
}



extern "C" float float_vec_print(Scope_Struct *scope_struct, DT_float_vec *vec) {
  // std::cout << "float_vec_print" << ".\n";
  std::cout << "[";
  for (int i=0; i<vec->size-1; i++)
    std::cout << vec->vec[i] << ", ";

  std::cout << vec->vec[vec->size-1] << "]" << "\n";
  return 0;
}







extern "C" DT_float_vec *float_vec_Split_Parallel(Scope_Struct *scope_struct, DT_float_vec *vec)
{
    float threads_count = scope_struct->asyncs_count;
    float thread_id = scope_struct->thread_id-1;
    // std::cout << "SPLITTING FLOAT VEC"  << ".\n";
    // std::cout << "Threads count: " << scope_struct->asyncs_count << ".\n";
    // std::cout << "Threads id: " << scope_struct->thread_id << ".\n\n";

    int vec_size = vec->size;
    int segment_size;

    segment_size = ceilf(vec_size/threads_count);

    // std::cout << "SEGMENT SIZE IS " << segment_size << ".\n";

    int size = segment_size;
    if((thread_id+1)==threads_count)
      size = vec_size - segment_size*thread_id;
      

    DT_float_vec *out_vector = new DT_float_vec(size);


    int c=0;
    for (int i = segment_size*thread_id; i<segment_size*(thread_id+1) && i<vec->size; ++i)
    {
      out_vector->vec[c] = vec->vec[i];
      c++;
    }

    return out_vector;
}


extern "C" DT_float_vec *float_vec_Split_Strided_Parallel(Scope_Struct *scope_struct, DT_float_vec *vec)
{
    float threads_count = scope_struct->asyncs_count;
    float thread_id = scope_struct->thread_id-1;
    // std::cout << "SPLITTING FLOAT VEC"  << ".\n";
    // std::cout << "Threads count: " << scope_struct->asyncs_count << ".\n";
    // std::cout << "Threads id: " << scope_struct->thread_id << ".\n\n";

    int vec_size = vec->size;
    int segment_size;

    segment_size = ceilf(vec_size/threads_count);

    // std::cout << "SEGMENT SIZE IS " << segment_size << ".\n";

    int size = segment_size;
    if((thread_id+1)==threads_count)
      size = vec_size - segment_size*thread_id;
      

    DT_float_vec *out_vector = new DT_float_vec(size);


    int c=0;

    for (int i = thread_id; c<segment_size && i<vec->size; i=i+threads_count)
    {
      out_vector->vec[c] = vec->vec[i];
      c++;
    }

    return out_vector;
}

extern "C" float float_vec_size(Scope_Struct *scope_struct, DT_float_vec *vec) {
  return (float)vec->size;
}