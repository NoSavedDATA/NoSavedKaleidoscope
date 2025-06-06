#include <iostream>
#include <string>
#include <fstream>
#include <iostream>


#include "../../tensor/tensor_struct.h"
#include "../../tensor/pool.h"

extern "C" float save_as_int(int thread_id, DT_tensor *tensor, char *bin_name)
{
  // Open the binary file for writing
  std::ofstream file(bin_name, std::ios::binary);

  if (!file.is_open()) {
    std::cerr << "Failed to open file: " << bin_name << "\n";
    return -1;
  }


  if (tensor->cpu_tensor_ptr==nullptr)
    cudaMallocHost(&tensor->cpu_tensor_ptr, round_to_nearest_pow2(tensor->dims_prod)*sizeof(float));
  
  cudaMemcpy(tensor->cpu_tensor_ptr, tensor->tensor_ptr, tensor->dims_prod*sizeof(float), cudaMemcpyDeviceToHost);

  int *int_data = new int[tensor->dims_prod];

  for (int i=0; i < tensor->dims_prod; ++i)
    int_data[i] = (int)tensor->cpu_tensor_ptr[i];

  
  file.write(reinterpret_cast<const char*>(int_data), tensor->dims_prod * sizeof(float));
  file.close();


  return 0;
}