#include "../codegen/random.h"

#include "../compiler_frontend/logging_v.h"
#include "../mangler/scope_struct.h"
#include "../pool/include.h"
#include "array.h"
#include "list.h"
#include "include.h"

DT_array::DT_array() {}

void DT_array::New(int size, int elem_size) {
    this->virtual_size = size;

    size = ((size + 7) / 8)*8;

    this->size = size;
    this->elem_size = elem_size;

    std::cout << "vsize " << virtual_size << " to size: " << size << ".\n";
    
    data = (void*)malloc(size*elem_size);
}


extern "C" DT_array *array_Create(Scope_Struct *scope_struct, char *name, char *scopeless_name, DT_array *init_val,
                                  DT_list *notes_vector, Data_Tree dt)
{
  if (init_val!=nullptr)
    return init_val;

  std::string elem_type = dt.Nested_Data[0].Type;
  int elem_size;
  if(data_name_to_size.count(elem_type)>0)
      elem_size = data_name_to_size[elem_type];
  else
      elem_size = 8;

  std::cout << "Elem size: " << elem_size << ".\n";

  DT_array *vec = newT<DT_array>(scope_struct, "array");
  vec->New(8, elem_size);
  vec->virtual_size = 0;

  return vec;
}

extern "C" int array_size(Scope_Struct *scope_struct, DT_array *vec) {
    return vec->virtual_size;
}



extern "C" int array_bad_idx(int line, int idx, int size) {
    
    LogErrorC(line, "Tried to index array at " + std::to_string(idx) + ", but the array size is: " + std::to_string(size));
}



extern "C" void array_double_size(DT_array *vec, int new_size) {
    // vec->data 
    int vec_size = new_size*vec->elem_size;

    void *new_data = malloc(vec_size);
    memcpy(new_data, vec->data, vec_size);

    vec->virtual_size++;
    vec->size = new_size;
}

extern "C" float array_print_int(Scope_Struct *scope_struct, DT_array *vec) {
    int *ptr = static_cast<int*>(vec->data);
    int size = vec->virtual_size;

    std::cout << "[";
    for (int i=0; i<size-1; ++i)
        std::cout << ptr[i] << ",";
    std::cout << ptr[size-1] << "]\n";
    return 0;
}


extern "C" DT_array *arange_int(Scope_Struct *scope_struct, int begin, int end) {
    DT_array *vec = newT<DT_array>(scope_struct, "array");
    vec->New(end-begin, 4);

    int *ptr = static_cast<int*>(vec->data);
    
    int c=0;
    for(int i=begin; i<end; ++i)
    {
        ptr[c] = i;
        c++;
    }

    return vec; 
} 


extern "C" DT_array *zeros_int(Scope_Struct *scope_struct, int N) {
    DT_array *vec = newT<DT_array>(scope_struct, "array");
    vec->New(N, 4);

    int *ptr = static_cast<int*>(vec->data);
    
    int c=0;
    for(int i=0; i<N; ++i)
    {
        ptr[c] = 0;
        c++;
    }

    return vec; 
} 



extern "C" DT_array *randint_array(Scope_Struct *scope_struct, int size, int min_val, int max_val) {
    DT_array *vec = newT<DT_array>(scope_struct, "array");
    vec->New(size,4);

    std::uniform_int_distribution<int> dist(min_val, max_val);

    int *ptr = static_cast<int*>(vec->data);
    for (int i = 0; i < size; ++i) {
        int r;
        {
            std::lock_guard<std::mutex> lock(MAIN_PRNG_MUTEX);
            r = dist(MAIN_PRNG);
        }
        ptr[i] = r;
    }

    return vec;
}


extern "C" DT_int_vec *ones_int(Scope_Struct *scope_struct, int size) {
    DT_int_vec *vec = newT<DT_int_vec>(scope_struct, "int_vec");
    vec->New(size);

    for(int i=0; i<size; ++i)
      vec->vec[i] = 1;

    return vec;
}



extern "C" DT_array *randfloat_array(Scope_Struct *scope_struct, int size, float min_val, float max_val) {
    DT_array *vec = newT<DT_array>(scope_struct, "array");
    vec->New(size,4);

    std::uniform_real_distribution<float> dist(min_val, max_val);

    float *ptr = static_cast<float*>(vec->data);
    for (int i = 0; i < size; ++i) {
        float r;
        {
            std::lock_guard<std::mutex> lock(MAIN_PRNG_MUTEX);
            r = dist(MAIN_PRNG);
        }
        ptr[i] = r;
    }

    return vec;
}

extern "C" float array_print_float(Scope_Struct *scope_struct, DT_array *vec) {
    float *ptr = static_cast<float*>(vec->data);
    int size = vec->virtual_size;

    std::cout << "[";
    for (int i=0; i<size-1; ++i)
        printf("%.3f, ",ptr[i]);
    printf("%.3f]\n",ptr[size-1]);
    return 0;
}


extern "C" DT_array *arange_float(Scope_Struct *scope_struct, float begin, float end) {
    DT_array *vec = newT<DT_array>(scope_struct, "float_vec");
    vec->New(end-begin, 4);

    float *ptr = static_cast<float*>(vec->data);
    
    int c=0;
    for(int i=begin; i<end; ++i)
    {
        ptr[c] = i;
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


extern "C" DT_array *array_Split_Parallel(Scope_Struct *scope_struct, DT_array *vec) {
    int threads_count = scope_struct->asyncs_count;
    int thread_id = scope_struct->thread_id-1;

    int vec_size = vec->size;
    int elem_size = vec->elem_size;
    int segment_size;

    segment_size = ceilf(vec_size/(float)threads_count);

    // std::cout << "SEGMENT SIZE IS " << segment_size << ".\n";

    int size = segment_size;
    if((thread_id+1)==threads_count)
      size = vec_size - segment_size*thread_id;
      

    DT_array *out_vector = newT<DT_array>(scope_struct, "array");
    out_vector->New(size, elem_size);


    // std::cout << "Splitting from " << std::to_string(segment_size*thread_id) << " to " << std::to_string(segment_size*(thread_id+1)) << ".\n";
    
    int copy_size;
    if(segment_size*(thread_id+1)>vec_size) 
        copy_size = (vec_size - segment_size*thread_id)*elem_size;
    else
        copy_size = segment_size*elem_size;

    memcpy(out_vector->data,
           static_cast<char*>(vec->data) + segment_size*thread_id*elem_size,
           copy_size);

    // int c=0;
    // for (int i = segment_size*thread_id; i<segment_size*(thread_id+1) && i<vec->size; ++i)
    // {
    //   out_vector->vec[c] = vec->vec[i];
    //   c++;
    // }

    return out_vector;

}
