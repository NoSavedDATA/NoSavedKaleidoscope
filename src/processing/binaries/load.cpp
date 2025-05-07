#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <cstdarg>

#include "../../common/include.h"
#include "../../tensor/tensor_dim_functions.h"
#include "../../tensor/tensor_struct.h"
#include "../../compiler_frontend/logging.h"



extern "C" float load_bin(DT_tensor *tensor, char *bin_name)
{

  //std::ifstream file(bin_name, std::ios::binary);
  std::ifstream file(bin_name, std::ios::binary | std::ios::ate);
  if (!file.is_open()) {
    std::string _error = bin_name;
    _error = "Failed to open file: " + _error;
    LogErrorS(_error);
    return 0;
  }

  file.seekg(0, std::ios::end);
  std::size_t file_size = file.tellg();
  file.seekg(0, std::ios::beg);


  float *file_data = new float[file_size / sizeof(float)];

  file.read(reinterpret_cast<char*>(file_data), file_size);
  file.close();

  //std::cout << bin_name << " has a file size of " << file_size << "\n";
  size_t num_elements = file_size / sizeof(float);


  if (num_elements>tensor->dims_prod)
  {
    std::string _error = "Tried to load binary data of size " + std::to_string(num_elements) + " into tensor " + tensor->name + " of size " + std::to_string(tensor->dims_prod);
    LogErrorS(_error);
    return 0;
  }


  float *image_data_float = tensor->cpu_tensor_ptr;


  for (size_t i = 0; i < num_elements; ++i)
  {
    //std::cout << "" << file_data[i] << "\n";
    image_data_float[i] = file_data[i];
  }
  
  
  delete[] file_data;

  return 0;
}



extern "C" float load_bin_idx(DT_tensor *tensor, char *bin_name, float first_idx, ...)
{
  std::vector<float> idxs;

  va_list args;
  va_start(args, first_idx);

  idxs.push_back(first_idx);

  for (int i=0; i<10; i++)
  {
    float dim = va_arg(args, float);
    if (dim==TERMINATE_VARARG)
      break;
    idxs.push_back(dim);
  }
  va_end(args);
  



  std::vector<float> dims, aux_dims;
  dims = tensor->dims;
  std::vector<float> new_dims;

  float offset=0;


  if (dims.size()==1)
    new_dims = {1.0f};
  else
  {
    aux_dims = dims;
    for (int i = 0; i < idxs.size(); i++)
    {
      aux_dims = RemoveFirstDim(aux_dims);
      offset += idxs[i]*DimsProd(aux_dims);
    }
    new_dims = aux_dims;
  }

  int new_dims_prod = DimsProd(new_dims);




  //std::ifstream file(bin_name, std::ios::binary);
  std::ifstream file(bin_name, std::ios::binary | std::ios::ate);
  if (!file.is_open()) {
    std::string _error = bin_name;
    _error = "Failed to open file: " + _error;
    LogErrorS(_error);
    return 0;
  }

  file.seekg(0, std::ios::end);
  std::size_t file_size = file.tellg();
  file.seekg(0, std::ios::beg);


  float *file_data = new float[file_size / sizeof(float)];

  file.read(reinterpret_cast<char*>(file_data), file_size);
  file.close();

  //std::cout << bin_name << " has a file size of " << file_size << "\n";
  size_t num_elements = file_size / sizeof(float);


  if (num_elements>tensor->dims_prod)
  {
    std::string _error = "Tried to load binary data of size " + std::to_string(num_elements) + " into tensor " + tensor->name + " of size " + std::to_string(tensor->dims_prod);
    LogErrorS(_error);
    return 0;
  }


  float *image_data_float = tensor->cpu_tensor_ptr + (int) offset;


  for (size_t i = 0; i < num_elements; ++i)
    image_data_float[i] = file_data[i];
  
  
  
  delete[] file_data;

  return 0;
}




extern "C" float wload_bin(DT_tensor *tensor, char *bin_name, float worker_idx, float batch_idx)
{
  //std::cout << "LOADING BINARY FOR: " << tensor->name <<  "\n";
  //std::cout << "Binary: " << bin_name <<  "\n";

  std::ifstream file(bin_name, std::ios::binary | std::ios::ate);
  if (!file.is_open()) {
    std::string _error = bin_name;
    _error = "Failed to open file: " + _error;
    LogErrorS(_error);
    return 0;
  }

  file.seekg(0, std::ios::end);
  std::size_t file_size = file.tellg();
  file.seekg(0, std::ios::beg);


  float *file_data = new float[file_size / sizeof(float)];

  // Read the binary file into the tensor_data pointer
  file.read(reinterpret_cast<char*>(file_data), file_size);
  file.close();

  size_t num_elements = file_size / sizeof(float);





  std::vector<float> dims = tensor->dims;

  std::vector<float> workerless_dims = BatchLessDims(dims);
  int workerless_dims_prod = DimsProd(workerless_dims);

  std::vector<float> batchless_dims = BatchLessDims(workerless_dims);
  int batchless_dims_prod = DimsProd(batchless_dims);


  float *image_data_float = tensor->cpu_tensor_ptr;
  int idx_offset = (int) (batchless_dims_prod*batch_idx + workerless_dims_prod*worker_idx);


  //todo: add out of bounds error here



  for (size_t i = 0; i < num_elements; ++i)
  {
    //std::cout << ", " << file_data[i];
    image_data_float[idx_offset + i] = file_data[i];
  }
  
  delete[] file_data;

  return 0;
}