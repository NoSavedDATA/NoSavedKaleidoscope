#include <cuda_runtime.h>

#include "../backprop/cleaners.h"
#include "../cuda_kernels/include.h"
#include "../data_types/include.h"
#include "../tensor/tensor_struct.h"





extern "C" float CopyArgTensor(Scope_Struct *scope_struct, DT_tensor *tensor, char *new_tensor_name)
{

  char *scope = scope_struct->scope;
  int thread_id = scope_struct->thread_id;
  std::string tensor_name = tensor->name;

  
  
  std::string arg_tensor_name = scope;
  arg_tensor_name = arg_tensor_name + new_tensor_name;
  

  std::vector<int> dims = tensor->dims;
  int dims_prod = tensor->dims_prod;

  float *tensor_ptr = tensor->tensor_ptr;

  

  DT_tensor *new_tensor = createTensor(tensor_ptr, dims, dims_prod, true, arg_tensor_name, tensor->cuda_stream, tensor->loader);
  new_tensor->scopeless_name = tensor->scopeless_name;

  NamedTensorsT[arg_tensor_name] = new_tensor;

  return 0;
}









// extern "C" float CopyArgTensor(Scope_Struct *scope_struct, DT_tensor *tensor)
// {
//   std::string tensor_name = tensor->name;

  
  
//   std::string arg_tensor_name = scope;
//   arg_tensor_name = arg_tensor_name + new_tensor_name;
  

//   std::vector<int> dims = tensor->dims;
//   int dims_prod = tensor->dims_prod;

//   float *arg_tensor, *tensor_ptr;

//   tensor_ptr = tensor->tensor_ptr;

//   std::string _name = "copy arg tensor of ";
//   _name = _name + tensor_name;
//   arg_tensor = get_from_pool(thread_id, dims_prod, _name);
  
//   //if (dims_prod!=0)//
//   //  cudaMemcpy(arg_tensor, tensor_ptr, dims_prod*sizeof(float), cudaMemcpyDeviceToDevice);//
  
//   cudaStream_t stream = ThreadsStream[thread_id];
//   if (dims_prod!=0)
//   {
//     int grid_size, block_size; 
//     CalculateGridAndBlockSizes(tensor->dims_prod, grid_size, block_size);

//     tensor->Sync();
//     copy_tensor_kernel<<<grid_size,block_size,0,stream>>>(arg_tensor, tensor_ptr, dims_prod);
//   }
  

//   DT_tensor *new_tensor = createTensor(arg_tensor, dims, dims_prod, true, arg_tensor_name, stream, tensor->loader);
//   new_tensor->scopeless_name = tensor->scopeless_name;
//   NamedTensorsT[arg_tensor_name] = new_tensor;


//   return 0;
// }