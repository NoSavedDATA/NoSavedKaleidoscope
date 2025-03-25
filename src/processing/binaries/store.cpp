#include <iostream>
#include <string>
#include <fstream>
#include <iostream>


#include <cuda_runtime.h>

#include "../../tensor/tensor_struct.h"


extern "C" float save_as_bin(int thread_id, Tensor *tensor, char *bin_name)
{
  std::ofstream file(bin_name, std::ios::binary);

  if (!file.is_open()) {
    std::cerr << "Failed to open file: " << bin_name << "\n";
    return -1;
  }


  if (tensor->cpu_tensor_ptr==nullptr)
    cudaMallocHost(&tensor->cpu_tensor_ptr, tensor->dims_prod*sizeof(float));
  
  cudaMemcpy(tensor->cpu_tensor_ptr, tensor->tensor_ptr, tensor->dims_prod*sizeof(float), cudaMemcpyDeviceToHost);

  
  file.write(reinterpret_cast<const char*>(tensor->cpu_tensor_ptr), tensor->dims_prod * sizeof(float));
  file.close();


  return 0;
}