
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


std::map<int, std::map<int, std::vector<int *>>> TensorPool_i32;




int *get_i32pool(int thread_id, int dims_prod, std::string from)
{
  
  // if (dims_prod==0)
  //   return nullptr;



  // int *tensor_ptr;

  // if(TensorPool_i32[thread_id].count(dims_prod)>0)
  // {
  //   std::vector<int *> tensors_in_pool = TensorPool_i32[thread_id][dims_prod];
  //   if (tensors_in_pool.size()>0)
  //   {
  //     //std::cout << "GETTING FROM POOL: " << dims_prod << "\n";
  //     tensor_ptr = tensors_in_pool.back();
  //     TensorPool_i32[thread_id][dims_prod].pop_back();
  //     return tensor_ptr;
  //   }
  // }

  
  // std::cout << "Malloc new INT8 from " << from << " of size: " << dims_prod << ", at thread: " << thread_id << "\n";
  // cudaCheck(cudaMalloc(&tensor_ptr, dims_prod*sizeof(int)));
  // return tensor_ptr;
}


void move_to_i32pool(int thread_id, int dims_prod, int *tensor_ptr, std::string from)
{
  // //if (dims_prod==50*256)
  // //  std::cout << "push B*OC of " << from << "\n";

  // // if (dims_prod==393216)
  // //   std::cout << "-------------Move " << from << ".\n";

  // if (dims_prod==0)
  //   return;
  // //std::cout << "move_to_pool from: " << from << "\n";
  

  // std::vector<int *> tensors_in_pool = TensorPool_i32[thread_id][dims_prod];
  // if (!in_int8_ptr_vec(tensor_ptr, tensors_in_pool))
  // {
  //   //if(!(tensors_in_pool.size()<30&&dims_prod==1))
  //   /*
  //   if(tensors_in_pool.size()<1000)
  //     TensorPool_i32[dims_prod].push_back(tensor_ptr);
  //   else
  //   {
  //     std::cout << "FREEING TENSOR WITH dims prod: " << dims_prod << " from: " << from <<  "\n";
  //     cudaCheck(cudaFree(tensor_ptr));
  //   }
  //   */
  //   TensorPool_i32[thread_id][dims_prod].push_back(tensor_ptr);
  // }  
}

