
#include<string>
#include<vector>
#include<map>
#include<cstring>
#include<random>
#include<thread>

#include "../backprop/include.h"
#include "../codegen/tensor_dim_functions.h"
#include "../common/include.h"
#include "../tensor/include.h"
#include "include.h"


extern "C" void CreatePinnedTensorOnDemand(char *tensor_name, char *init)
{
  std::vector<float> dims = NamedDims[tensor_name];
  NamedDims[tensor_name].clear();
  Tensor *tensor;

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
  
}