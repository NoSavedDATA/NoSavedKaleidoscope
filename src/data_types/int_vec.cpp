#include <iostream>
#include <vector>
#include <map>

#include "../codegen/random.h"
#include "../compiler_frontend/global_vars.h"
#include "../compiler_frontend/logging_execution.h"
#include "../compiler_frontend/logging_v.h"
#include "../mangler/scope_struct.h"
#include "../pool/include.h"
#include "include.h"


DT_int_vec::DT_int_vec() {}
DT_int_vec::DT_int_vec(int size) {
  vec = (int*)malloc(size*sizeof(int));
}

void DT_int_vec::New(int size)  {
    this->size = size;
    vec = (int*)malloc(size*sizeof(int));
}


extern "C" void *int_vec_Create(Scope_Struct *scope_struct, int size)
{
  DT_int_vec *vec = newT<DT_int_vec>(scope_struct, "int_vec");
  vec->New(size);

  return vec;
}


DT_int_vec::~DT_int_vec() {
    // free(vec);
}




extern "C" int nsk_vec_size(Scope_Struct *scope_struct, Nsk_Vector *vec) {
    return vec->size;
}

 
void int_vec_Clean_Up(void *data_ptr) {
  std::cout << "delete int_vec" << ".\n";
  DT_int_vec *vec = static_cast<DT_int_vec*>(data_ptr);
  free(vec->vec);
  // delete vec;
}






extern "C" Vec_Slices *int_vec_CalculateSliceIdx(DT_int_vec *vec, int first_idx, ...) {


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



extern "C" DT_int_vec *int_vec_Slice(Scope_Struct *scope_struct, DT_int_vec *vec, Vec_Slices *slices) {

    int start=slices->slices[0].vec[0], end=slices->slices[0].vec[1];

    if (end==COPY_TO_END_INST)
      end = vec->size;


    int size = end-start;

    DT_int_vec *out_vec = newT<DT_int_vec>(scope_struct, "int_vec");
    out_vec->New(size);

    for (int i=0; i<size; ++i)
      out_vec->vec[i] = vec->vec[start+i];

    return out_vec;
}







extern "C" int int_vec_print(Scope_Struct *scope_struct, DT_int_vec *vec) {

  // std::cout << "print vec of size " << vec->size << ".\n";
  std::cout << "[";
  for (int i=0; i<vec->size-1; i++)
    std::cout << vec->vec[i] << ", ";

  std::cout << vec->vec[vec->size-1] << "]" << "\n";
  return 0;
}









extern "C" DT_int_vec *int_vec_Split_Parallel(Scope_Struct *scope_struct, DT_int_vec *vec)
{
    // std::cout << "SPLITTING int VEC"  << ".\n";
    // std::cout << "Threads count: " << scope_struct->asyncs_count << ".\n";
    // std::cout << "Thread id: " << scope_struct->thread_id << ".\n\n";
    int threads_count = scope_struct->asyncs_count;
    int thread_id = scope_struct->thread_id-1;

    int vec_size = vec->size;
    int segment_size;

    segment_size = ceilf(vec_size/(float)threads_count);

    // std::cout << "SEGMENT SIZE IS " << segment_size << ".\n";

    int size = segment_size;
    if((thread_id+1)==threads_count)
      size = vec_size - segment_size*thread_id;
      

    DT_int_vec *out_vector = newT<DT_int_vec>(scope_struct, "int_vec");
    out_vector->New(size);


    // std::cout << "Splitting from " << std::to_string(segment_size*thread_id) << " to " << std::to_string(segment_size*(thread_id+1)) << ".\n";

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
    // std::cout << "Thread id: " << scope_struct->thread_id << ".\n\n";

    int vec_size = vec->size;
    int segment_size;

    segment_size = ceilf(vec_size/(float)threads_count);

    // std::cout << "SEGMENT SIZE IS " << segment_size << ".\n";

    int size = segment_size;
    if((thread_id+1)==threads_count)
      size = vec_size - segment_size*thread_id;
      

    DT_int_vec *out_vector = newT<DT_int_vec>(scope_struct, "int_vec");
    out_vector->New(size);


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
