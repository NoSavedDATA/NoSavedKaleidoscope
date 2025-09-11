#include <iostream>
#include <vector>
#include <map>

#include "../codegen/random.h"
#include "../compiler_frontend/global_vars.h"
#include "../compiler_frontend/logging_execution.h"
#include "../mangler/scope_struct.h"
#include "include.h"



extern "C" void *int_vec_Create(Scope_Struct *scope_struct, char *name, char *scopeless_name, void *init_val, DT_list *notes_vector)
{
  // std::cout << "int_vec_Create" << ".\n";

  if (init_val!=nullptr)
    DT_int_vec *vec = static_cast<DT_int_vec *>(init_val);


  return init_val;
}

 
void int_vec_Clean_Up(void *data_ptr) {
}


extern "C" int int_vec_Store_Idx(DT_int_vec *vec, int idx, int value, Scope_Struct *scope_struct){
  vec->vec[idx] = value;
  return 0;
}


extern "C" DT_int_vec *arange_int(Scope_Struct *scope_struct, int begin, int end) {
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
  // TODO: turn into python like expression [0]*size

  DT_int_vec *vec = new DT_int_vec(size);
  for(int i=0; i<size; ++i)
    vec->vec[i] = 0;

  return vec;
}


extern "C" DT_int_vec *rand_int_vec(Scope_Struct *scope_struct, int size, int min_val, int max_val) {
    DT_int_vec *vec = new DT_int_vec(size);

    std::uniform_int_distribution<int> dist(min_val, max_val);

    for (int i = 0; i < size; ++i) {
        int r;
        {
            std::lock_guard<std::mutex> lock(MAIN_PRNG_MUTEX);
            r = dist(MAIN_PRNG);
        }
        vec->vec[i] = r;
    }

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
  if (idx>vec->size)
    LogErrorEE(scope_struct->code_line, "Index " + std::to_string(idx) + " is out of bounds for a vector of size: " + std::to_string(vec->size) + ".");

  return vec->vec[idx];
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
  if (first_idx>vec->size)
    LogErrorEE(-1, "Vector out of bounds. Index: " + std::to_string(first_idx) + " vs size " + std::to_string(vec->size));

  return first_idx;
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
      

    DT_int_vec *out_vector = new DT_int_vec(size);


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