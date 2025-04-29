
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


std::map<int, std::map<float, std::vector<float *>>> TensorPool;
std::map<int, std::map<float, std::vector<half *>>> TensorHalfPool;



float *get_from_pool(int thread_id, float dims_prod, std::string from, bool is_new)
{

  if (dims_prod==0)
    return nullptr;

  // if (dims_prod==400)
  //   std::cout << "*************Get " << from << ".\n";


  float *tensor_ptr;

  if(TensorPool[thread_id].count(dims_prod)>0 && !is_new)
  {
    std::vector<float *> tensors_in_pool = TensorPool[thread_id][dims_prod];
    if (tensors_in_pool.size()>0)
    {
      //std::cout << "GETTING FROM POOL: " << dims_prod << "\n";
      tensor_ptr = tensors_in_pool.back();
      TensorPool[thread_id][dims_prod].pop_back();
      return tensor_ptr;
    }
  }

  

  std::cout << "Malloc new space from " << from << " of size: " << dims_prod << ", at thread: " << thread_id << "\n";

  cudaCheck(cudaMalloc(&tensor_ptr, dims_prod*sizeof(float)));
  return tensor_ptr;
}


void move_to_pool(int thread_id, float dims_prod, float *tensor_ptr, std::string from)
{
  //if (dims_prod==50*256)
  //  std::cout << "push B*OC of " << from << "\n";

  // if (dims_prod==400)
  //   std::cout << "-------------Move " << from << ".\n";

  if (dims_prod==0)
    return;
  //std::cout << "move_to_pool from: " << from << "\n";
  

  std::vector<float *> tensors_in_pool = TensorPool[thread_id][dims_prod];
  if (!in_float_ptr_vec(tensor_ptr, tensors_in_pool))
  {
    //if(!(tensors_in_pool.size()<30&&dims_prod==1))
    /*
    if(tensors_in_pool.size()<1000)
      TensorPool[dims_prod].push_back(tensor_ptr);
    else
    {
      std::cout << "FREEING TENSOR WITH dims prod: " << dims_prod << " from: " << from <<  "\n";
      cudaCheck(cudaFree(tensor_ptr));
    }
    */
    TensorPool[thread_id][dims_prod].push_back(tensor_ptr);
  }  
}




void move_to_pool(int thread_id, float dims_prod, half *tensor_ptr, std::string from)
{
  if (dims_prod==0)
    return;

  std::vector<half *> tensors_in_pool = TensorHalfPool[thread_id][dims_prod];
  if (!in_half_ptr_vec(tensor_ptr, tensors_in_pool))
  {
    TensorHalfPool[thread_id][dims_prod].push_back(tensor_ptr);
  }  
}



half *get_half_from_pool(int thread_id, float dims_prod, std::string from)
{
  

  if (dims_prod==0)
    return nullptr;


  half *tensor_ptr;

  if(TensorHalfPool[thread_id].count(dims_prod)>0)
  {
    std::vector<half *> tensors_in_pool = TensorHalfPool[thread_id][dims_prod];
    if (tensors_in_pool.size()>0)
    {
      tensor_ptr = tensors_in_pool.back();
      TensorHalfPool[thread_id][dims_prod].pop_back();
      return tensor_ptr;
    }
  }

  

  std::cout << "Malloc HALF new space from " << from << " of size: " << dims_prod << ", at thread: " << thread_id << "\n";

  cudaCheck(cudaMalloc(&tensor_ptr, dims_prod*sizeof(half)));
  return tensor_ptr;
}




void move_to_pool_pow2(int thread_id, float dims_prod, float *tensor_ptr, std::string from)
{
  
  if (dims_prod==0)
    return;

  float nearest_ceil_pow2 = 1;
  while(nearest_ceil_pow2<dims_prod)
    nearest_ceil_pow2*=2;
  dims_prod = nearest_ceil_pow2;

  std::vector<float *> tensors_in_pool = TensorPool[thread_id][dims_prod];

  if (!in_float_ptr_vec(tensor_ptr, tensors_in_pool))
    TensorPool[thread_id][dims_prod].push_back(tensor_ptr);
  
}

float *get_from_pool_pow2(int thread_id, float dims_prod, std::string from)
{
  if (dims_prod==0)
    return nullptr;

  float nearest_ceil_pow2 = 1;
  while(nearest_ceil_pow2<dims_prod)
    nearest_ceil_pow2*=2;
  dims_prod = nearest_ceil_pow2;

  float *tensor_ptr;

  if(TensorPool[thread_id].count(dims_prod)>0)
  {
    std::vector<float *> tensors_in_pool = TensorPool[thread_id][dims_prod];
    if (tensors_in_pool.size()>0)
    {
      tensor_ptr = tensors_in_pool.back();
      TensorPool[thread_id][dims_prod].pop_back();
      return tensor_ptr;
    }
  }


  std::cout << "Malloc new space from " << from << " of size: " << dims_prod << ", at thread: " << thread_id << " with the nearest pow of 2\n";

  cudaCheck(cudaMalloc(&tensor_ptr, dims_prod*sizeof(float)));
  return tensor_ptr;
}

