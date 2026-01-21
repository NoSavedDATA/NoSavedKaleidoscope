#include "../codegen/random.h"

#include "../compiler_frontend/logging_v.h"
#include "../mangler/scope_struct.h"
#include "../pool/include.h"
#include "array.h"
#include "list.h"
#include "include.h"

DT_array::DT_array() {}

void DT_array::New(int size, int elem_size, std::string type) {
    this->virtual_size = size;
    this->elem_size = elem_size;
    this->type = type;

    size = ((size + 7) / 8)*8;
    this->size = size;
    
    data = (void*)malloc(size*elem_size);
}

void DT_array::New(int size, std::string type) {
    this->virtual_size = size;
    this->elem_size = 8;
    this->type = type;

    size = ((size + 7) / 8)*8;
    this->size = size;
    
    data = (void*)malloc(size*8);
}


extern "C" DT_array *array_Create(Scope_Struct *scope_struct, Data_Tree dt)
{
  std::string elem_type = dt.Nested_Data[0].Type;
  int elem_size;
  if(data_name_to_size.count(elem_type)>0)
      elem_size = data_name_to_size[elem_type];
  else
      elem_size = 8;


  DT_array *vec = newT<DT_array>(scope_struct, "array");
  vec->New(8, elem_size, elem_type);
  vec->virtual_size = 0;

  return vec;
}

void array_Clean_Up(void *data_ptr) {
    DT_array *array = static_cast<DT_array *>(data_ptr);
    free(array->data);
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

extern "C" void array_print_int(Scope_Struct *scope_struct, DT_array *vec) {
    int *ptr = static_cast<int*>(vec->data);
    int size = vec->virtual_size;

    std::cout << "[";
    for (int i=0; i<size-1; ++i)
        std::cout << ptr[i] << ",";
    std::cout << ptr[size-1] << "]\n";
}


extern "C" DT_array *arange_int(Scope_Struct *scope_struct, int begin, int end) {
    DT_array *vec = newT<DT_array>(scope_struct, "array");
    vec->New(end-begin, 4, "int");

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
    vec->New(N, 4, "int");

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
    vec->New(size,4, "int");

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


extern "C" DT_array *ones_int(Scope_Struct *scope_struct, int N) {
    DT_array *vec = newT<DT_array>(scope_struct, "array");
    vec->New(N, 4, "int");

    int *ptr = static_cast<int*>(vec->data);
    
    int c=0;
    for(int i=0; i<N; ++i)
    {
        ptr[c] = 1;
        c++;
    }

    return vec;
}

extern "C" DT_array *array_int_add(Scope_Struct *scope_struct, DT_array *array, int x) {
    DT_array *new_array = newT<DT_array>(scope_struct, "array");
    new_array->New(array->virtual_size, 4, "int");
    
    int *data = static_cast<int *>(array->data);
    int *new_data = static_cast<int *>(new_array->data);
    for (int i=0; i<array->virtual_size; ++i) {
        new_data[i] = data[i] + x;
    }

    return new_array;
}



extern "C" DT_array *randfloat_array(Scope_Struct *scope_struct, int size, float min_val, float max_val) {
    DT_array *vec = newT<DT_array>(scope_struct, "array");
    vec->New(size,4,"float");

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

extern "C" void array_print_float(Scope_Struct *scope_struct, DT_array *vec) {
    float *ptr = static_cast<float*>(vec->data);
    int size = vec->virtual_size;

    std::cout << "[";
    for (int i=0; i<size-1; ++i)
        printf("%.3f, ",ptr[i]);
    printf("%.3f]\n",ptr[size-1]);
}


extern "C" DT_array *arange_float(Scope_Struct *scope_struct, float begin, float end) {
    DT_array *vec = newT<DT_array>(scope_struct, "float_vec");
    vec->New(end-begin, 4, "float");

    float *ptr = static_cast<float*>(vec->data);
    
    int c=0;
    for(int i=begin; i<end; ++i)
    {
        ptr[c] = i;
        c++;
    }

    return vec; 
} 

extern "C" DT_array *zeros_float(Scope_Struct *scope_struct, int N) {
    DT_array *vec = newT<DT_array>(scope_struct, "array");
    vec->New(N, 4, "float");

    float *ptr = static_cast<float*>(vec->data);
    
    int c=0;
    for(int i=0; i<N; ++i)
    {
        ptr[c] = 0.0f;
        c++;
    }

    return vec; 
}


extern "C" DT_array *ones_float(Scope_Struct *scope_struct, int N) {
    DT_array *vec = newT<DT_array>(scope_struct, "array");
    vec->New(N, 4, "float");

    float *ptr = static_cast<float*>(vec->data);
    
    int c=0;
    for(int i=0; i<N; ++i)
    {
        ptr[c] = 1.0f;
        c++;
    }

    return vec; 
}



extern "C" void array_print_str(Scope_Struct *scope_struct, DT_array *vec) {
    char **ptr = static_cast<char**>(vec->data);
    int size = vec->virtual_size;

    std::cout << "[";
    for (int i=0; i<size-1; ++i)
        std::cout << ptr[i] << ",";
    std::cout << ptr[size-1] << "]\n";
}


extern "C" DT_array *array_Split_Parallel(Scope_Struct *scope_struct, DT_array *vec) {
    int threads_count = scope_struct->asyncs_count;
    int thread_id = scope_struct->thread_id-1;

    int vec_size = vec->virtual_size;
    int elem_size = vec->elem_size;
    int segment_size;

    segment_size = ceilf(vec_size/(float)threads_count);


    int size = segment_size;
    if((thread_id+1)==threads_count)
      size = vec_size - segment_size*thread_id;
      

    int copy_size;
    if(segment_size*(thread_id+1)>vec_size) 
        copy_size = (vec_size - segment_size*thread_id)*elem_size;
    else
        copy_size = segment_size*elem_size;


    DT_array *out_vector = newT<DT_array>(scope_struct, "array");
    out_vector->New(size, elem_size, vec->type);

    
    memcpy(out_vector->data,
           static_cast<char*>(vec->data) + segment_size*thread_id*elem_size,
           copy_size);

    return out_vector;
}
