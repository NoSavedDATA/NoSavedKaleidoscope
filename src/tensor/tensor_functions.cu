
// #include <algorithm>
// #include <cstdarg>
// #include <cassert>
// #include <cctype>
// #include <cstring>
// #include <cstdint>
// #include <cstdio>
// #include <cstdlib>
// #include <map>
// #include <memory>
// #include <string>
// #include <iostream>
// #include <numeric>
// #include <utility>
// #include <vector>
// #include <iomanip>
// #include <math.h>
// #include <fenv.h>
// #include <tuple>
// #include <glob.h>
// #include <chrono>
// #include <thread>
// #include <random>
// #include <float.h>
// #include <fstream>
// #include <sstream>
// #include <filesystem>
// #include <stdio.h>
// #include <stdlib.h>
// #include <omp.h>

// #include <cuda_fp16.h>

// #include "../common/include.h"
// #include "../cu_commons.h"
// #include "include.h"





// extern "C" float CreateTensorOnDemand(char *tensor_name, char *scopeless_name, char *init, int is_weight, int thread_id, char *scope)
// {
//   //std::cout << "CREATING TENSOR " << tensor_name << " AT THREAD: " << thread_id << "\n";

//   Tensor *tensor;

//   std::vector<float> dims = NamedDims[tensor_name];
//   NamedDims[tensor_name].clear(); //TODO: Global vars are bad with threads.

//   int product = DimsProd(dims);

//   float *tensor_ptr;
//   float *tensor_cpu;

//   if(product>0)
//   {
//     if (std::strcmp(init, "randu") == 0)
//       tensor_cpu = make_random_float_uniform(product);
//     if (std::strcmp(init, "zeros") == 0)
//       tensor_cpu = make_zeros_float(product);
//     if (std::strcmp(init, "ones") == 0)
//       tensor_cpu = make_ones_float(product);
//     if (std::strcmp(init, "normal") == 0)
//       tensor_cpu = make_normal(product);
//     if (std::strcmp(init, "xavu") == 0)
//       tensor_cpu = make_xavier_uniform_float(product, dims[dims.size()-1], dims[dims.size()-2]);
//     if (std::strcmp(init, "xavu_relu") == 0)
//       tensor_cpu = make_xavier_uniform_float_relu(product, dims[dims.size()-1], dims[dims.size()-2]);
//     if (std::strcmp(init, "xavu_tanh") == 0)
//       tensor_cpu = make_xavier_uniform_float_tanh(product, dims[dims.size()-1], dims[dims.size()-2]);
//     if (std::strcmp(init, "he_normal_relu") == 0)
//       tensor_cpu = make_he_normal_float_relu(product, dims[dims.size()-1]);
//     if (std::strcmp(init, "init_gpt") == 0)
//       tensor_cpu = make_gpt_init(product);
//     if (std::strcmp(init, "int") == 0)
//       tensor_cpu = make_random_int(product, 10);
//     if (std::strcmp(init, "arange") == 0)
//       tensor_cpu = make_arange(product);
//     if (std::strcmp(init, "binary") == 0)
//       tensor_cpu = make_random_int(product, 1);

//     cudaCheck(cudaGetLastError());
//     std::string _name = "create tensor ";
//     _name = _name + tensor_name;
//     tensor_ptr = get_from_pool(thread_id, product, _name);
//     //std::cout << "cpy of: " << tensor_name << "\n";

//     cudaStream_t stream = ThreadsStream[thread_id];
//     cudaCheck(cudaMemcpyAsync(tensor_ptr, tensor_cpu, product*sizeof(float), cudaMemcpyHostToDevice, stream));
//     //cudaStreamSynchronize(stream);
//     delete[] tensor_cpu;
//   }


  
//   /*
//   if(NamedTensorsT.count(tensor_name)>0)
//   {
//     tensor = NamedTensorsT[tensor_name];
//     if (tensor!=nullptr)
    
//       delete tensor;
//       //cudaCheck(cudaFree(aux_ptr));
//   }
//   */
  
  


//   tensor = createTensor(tensor_ptr, dims, product, true, tensor_name);
//   tensor->scopeless_name = scopeless_name;
//   if((bool)is_weight)
//     tensor->SetIsWeight();
//   tensor->op = create_tensor_op;

  
//   NamedTensorsT[tensor_name] = tensor;

  
//   delete[] tensor_name;
//   delete[] scopeless_name;


//   return 0;
// }



