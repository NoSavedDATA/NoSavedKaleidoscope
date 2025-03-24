
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



float *get_from_pool(int thread_id, float dims_prod, std::string from)
{

  if (dims_prod==0)
    return nullptr;


  float *tensor_ptr;

  if(TensorPool[thread_id].count(dims_prod)>0)
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
