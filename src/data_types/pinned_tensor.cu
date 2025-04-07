
#include<string>
#include<vector>
#include<map>
#include<cstring>
#include<random>
#include<thread>

#include "../backprop/include.h"
#include "../common/include.h"
#include "../tensor/include.h"
#include "include.h"



extern "C" float pinned_tensor_Create(char *tensor_name, char *scopeless_name, float init_val, AnyVector *notes_vector, int thread_id, char *scope)
{

  // std::cout << "PINNED TENSOR CREATE"  << ".\n";

  Tensor *tensor;

  std::vector<float> dims;
  bool is_weight = false;
  for (int i=0; i<notes_vector->data->size(); i++)
  {
    if(notes_vector->data_types->at(i)=="float")
      dims.push_back(notes_vector->get<float>(i));
    if(notes_vector->data_types->at(i)=="string")
      char *note = notes_vector->get<char *>(i);
  }


  int product = DimsProd(dims);
  float *tensor_ptr, *pool_tensor;
  float *tensor_cpu;


  cudaMallocHost(&tensor_cpu, product*sizeof(float));
  //tensor_cpu = new float[product];

  for (int i = 0; i < product; ++i) {
    tensor_cpu[i] = 0.0f;
  }
  

  cudaMalloc(&tensor_ptr, product*sizeof(float));  
  tensor = createPinned(tensor_ptr, tensor_cpu, dims, product, tensor_name);
  NamedTensorsT[tensor_name] = tensor;
  

  
  // pinned tensors are 1 pool tensor behind.
  std::vector<float> pool_dims = dims;
  pool_dims.erase(pool_dims.begin());
  float pool_product = DimsProd(pool_dims);
  pool_tensor = get_from_pool(0, pool_product, "create pinned");
  move_to_pool(0, pool_product, pool_tensor, "create pinned");
  

  return 0;
}


