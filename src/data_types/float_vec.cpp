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

DT_float_vec::DT_float_vec() {
}

DT_float_vec::DT_float_vec(int size) {
  vec = (float*)malloc(size*sizeof(float));
}

DT_float_vec::~DT_float_vec() {
    // free(vec);
}



extern "C" void *float_vec_Create(Scope_Struct *scope_struct, int size)
{
  DT_float_vec *vec = newT<DT_float_vec>(scope_struct, "float_vec");
  vec->New(size);

  return vec;
}

 
void float_vec_Clean_Up(void *data_ptr) {
  // std::cout << "delete float_vec" << ".\n";
  DT_float_vec *vec = static_cast<DT_float_vec*>(data_ptr);
  free(vec->vec);
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
