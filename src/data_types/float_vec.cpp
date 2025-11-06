#include <iostream>
#include <vector>
#include <map>
#include <math.h>

#include "../compiler_frontend/logging_v.h"
#include "../mangler/scope_struct.h"
#include "../pool/include.h"
#include "include.h"

#include "float_vec.h"


void DT_float_vec::New(int size) {
  this->size = size;
  vec = (float*)malloc(size*sizeof(float)); 
}

DT_float_vec::DT_float_vec() : Nsk_Vector() {
}

DT_float_vec::DT_float_vec(int size) : Nsk_Vector(size) {
  vec = (float*)malloc(size*sizeof(float));
}

DT_float_vec::~DT_float_vec() {
    // free(vec);
}



extern "C" void *float_vec_Create(Scope_Struct *scope_struct, char *name, char *scopeless_name, void *init_val, DT_list *notes_vector)
{
  if (init_val!=nullptr)
    return init_val;
  if(notes_vector==nullptr)  
    return nullptr;

  if(notes_vector->size!=1||notes_vector==nullptr) {
    LogErrorC(-1, "float_vec requires size argument");
    return nullptr;
  }

  DT_float_vec *vec = newT<DT_float_vec>(scope_struct, "float_vec");
  vec->New(notes_vector->get<int>(0));

  return init_val;
}

 
void float_vec_Clean_Up(void *data_ptr) {
  // std::cout << "delete float_vec" << ".\n";
  DT_float_vec *vec = static_cast<DT_float_vec*>(data_ptr);
  free(vec->vec);
  delete vec;
}


extern "C" float float_vec_Store_Idx(DT_float_vec *vec, float idx, float value, Scope_Struct *scope_struct){
  vec->vec[(int)idx] = value;
  return 0;
}







extern "C" DT_float_vec *arange_float(Scope_Struct *scope_struct, int begin, int end) {
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


extern "C" DT_float_vec *zeros_float(Scope_Struct *scope_struct, int size) {
  DT_float_vec *vec = newT<DT_float_vec>(scope_struct, "float_vec");
  vec->New(size);
  for(int i=0; i<size; ++i)
    vec->vec[i] = 0;
  return vec;
}


extern "C" DT_float_vec *ones_float(Scope_Struct *scope_struct, int size) {
  DT_float_vec *vec = newT<DT_float_vec>(scope_struct, "float_vec");
  vec->New(size);
  for(int i=0; i<size; ++i)
    vec->vec[i] = 1;
  return vec;
}


extern "C" float float_vec_Idx(Scope_Struct *scope_struct, DT_float_vec *vec, int idx)
{
  // std::cout << "Load vec on idx " << idx << ".\n";
  float ret = vec->vec[idx];
  // std::cout << "got: " << ret << ".\n";
  // std::cout << "returning" << ".\n"; 
  return ret;
}




extern "C" int float_vec_CalculateIdx(DT_float_vec *vec, int first_idx, ...) {
  if (first_idx<0)
    first_idx = vec->size+first_idx;
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
  std::cout << "[";
  for (int i=0; i<vec->size-1; i++)
    std::cout << vec->vec[i] << ", ";

  std::cout << vec->vec[vec->size-1] << "]" << "\n";
  return 0;
}



extern "C" DT_float_vec *float_vec_pow(Scope_Struct *scope_struct, DT_float_vec *vec, float exponent) {
  
  DT_float_vec *out_vec = newT<DT_float_vec>(scope_struct, "float_vec");
  out_vec->New(vec->size);
  
  for (int i=0; i<vec->size; ++i)
    out_vec->vec[i] = std::pow(vec->vec[i], exponent);

  return out_vec;
}


extern "C" float float_vec_sum(Scope_Struct *scope_struct, DT_float_vec *vec) {
  
  float sum=0;
  for (int i=0; i<vec->size; ++i)
    sum += vec->vec[i];

  return sum;
}


extern "C" DT_float_vec *float_vec_int_add(Scope_Struct *scope_struct, DT_float_vec *vec, int rhs) {
  DT_float_vec *out_vec = newT<DT_float_vec>(scope_struct, "float_vec");
  out_vec->New(vec->size);

  for (int i=0; i<vec->size; ++i)
    out_vec->vec[i] = vec->vec[i] + rhs;

  return out_vec;
}


extern "C" DT_float_vec *float_vec_int_div(Scope_Struct *scope_struct, DT_float_vec *vec, int rhs) {
  DT_float_vec *out_vec = newT<DT_float_vec>(scope_struct, "float_vec");
  out_vec->New(vec->size);

  for (int i=0; i<vec->size; ++i)
    out_vec->vec[i] = vec->vec[i] / rhs;

  return out_vec;
}


extern "C" DT_float_vec *float_vec_float_vec_add(Scope_Struct *scope_struct, DT_float_vec *lhs, DT_float_vec *rhs) {
  if(lhs->size!=rhs->size) {
    LogErrorC(scope_struct->code_line, "Tried to add float vectors of different sizes. LHS size: " + std::to_string(lhs->size) + ", RHS size: " + std::to_string(rhs->size));
    return nullptr;
  }
  DT_float_vec *out_vec = newT<DT_float_vec>(scope_struct, "float_vec");
  out_vec->New(lhs->size);
  
  for (int i=0; i<lhs->size; ++i)
    out_vec->vec[i] = lhs->vec[i] + rhs->vec[i];

  return out_vec;
}


extern "C" DT_float_vec *float_vec_float_vec_sub(Scope_Struct *scope_struct, DT_float_vec *lhs, DT_float_vec *rhs) {
  if(lhs->size!=rhs->size) {
    LogErrorC(scope_struct->code_line, "Tried to add float vectors of different sizes. LHS size: " + std::to_string(lhs->size) + ", RHS size: " + std::to_string(rhs->size));
    return nullptr;
  }
  DT_float_vec *out_vec = newT<DT_float_vec>(scope_struct, "float_vec");
  out_vec->New(lhs->size);
  
  for (int i=0; i<lhs->size; ++i)
    out_vec->vec[i] = lhs->vec[i] - rhs->vec[i];

  return out_vec;
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

    segment_size = ceilf(vec_size/(float)threads_count);

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

    segment_size = ceilf(vec_size/(float)threads_count);

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

extern "C" int float_vec_size(Scope_Struct *scope_struct, DT_float_vec *vec) {
  return vec->size;
}