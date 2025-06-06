
#include <algorithm>
#include <cstdarg>
#include <cassert>
#include <cctype>
#include <cstring>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <memory>
#include <string>
#include <iostream>
#include <numeric>
#include <utility>
#include <vector>
#include <iomanip>
#include <math.h>
#include <fenv.h>
#include <tuple>
#include <glob.h>
#include <chrono>
#include <thread>
#include <random>
#include <float.h>
#include <fstream>
#include <sstream>
#include <filesystem>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#include <cuda_fp16.h>

#include "../common/include.h"
#include "../compiler_frontend/logging.h"
#include "include.h"


std::map<int, std::map<int, std::vector<int8_t *>>> TensorPool_i8;




int8_t *get_i8pool(int thread_id, int dims_prod, std::string from)
{

  if (dims_prod==0)
    return nullptr;

  dims_prod = dims_prod/4;


  float *tensor_ptr;

  if(TensorPool[thread_id].count(dims_prod)>0)
  {
    std::vector<float *> tensors_in_pool = TensorPool[thread_id][dims_prod];
    if (tensors_in_pool.size()>0)
    {
      //std::cout << "GETTING FROM POOL: " << dims_prod << "\n";
      tensor_ptr = tensors_in_pool.back();
      TensorPool[thread_id][dims_prod].pop_back();
      return (int8_t*)tensor_ptr;
    }
  }

  
  std::cout << "Malloc new INT8 from " << from << " of size: " << dims_prod << ", at thread: " << thread_id << "\n";
  cudaCheck(cudaMalloc(&tensor_ptr, dims_prod*sizeof(4)));
  return (int8_t*)tensor_ptr;
}


void move_to_i8pool(int thread_id, int dims_prod, int8_t *_tensor_ptr, std::string from)
{

  if (dims_prod==0)
    return;
  float *tensor_ptr = (float*) _tensor_ptr;

  dims_prod = dims_prod/4;
  

  std::vector<float *> tensors_in_pool = TensorPool[thread_id][dims_prod];
  if (!in_float_ptr_vec(tensor_ptr, tensors_in_pool))
    TensorPool[thread_id][dims_prod].push_back(tensor_ptr);
}

