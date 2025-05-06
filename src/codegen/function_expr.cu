#include <cuda_runtime.h>

#include "../backprop/cleaners.h"
#include "../cuda_kernels/include.h"
#include "../data_types/include.h"
#include "../tensor/tensor_struct.h"


extern "C" void StoreArgOnDemand(char *scope, char *name, float value){
  // std::cout << "StoreArgOnDemand: " << name  << " " << value << "\n";
  
  pthread_mutex_lock(&clean_scope_mutex);
  
  NamedClassValues[name] = value;

  std::string _name = name;
  
  pthread_mutex_unlock(&clean_scope_mutex);

}



extern "C" float CopyArgTensor(data_type_tensor *tensor, char *new_tensor_name, char *previous_scope, char *scope, int thread_id)
{
  std::string tensor_name = tensor->name;
  //std::cout << "\n\n\nCOPY ARG TENSOR OF " << previous_scope << tensor_name << " into " << scope<<new_tensor_name  << " at thread: " << thread_id << "\n";

  
  
  std::string arg_tensor_name = scope;
  arg_tensor_name = arg_tensor_name + new_tensor_name;
  

  std::vector<float> dims = tensor->dims;
  int dims_prod = tensor->dims_prod;

  float *arg_tensor, *tensor_ptr;

  tensor_ptr = tensor->tensor_ptr;

  std::string _name = "arg tensor of ";
  _name = _name + tensor_name;
  arg_tensor = get_from_pool(thread_id, dims_prod, _name);
  
  //if (dims_prod!=0)//
  //  cudaMemcpy(arg_tensor, tensor_ptr, dims_prod*sizeof(float), cudaMemcpyDeviceToDevice);//
  
  if (dims_prod!=0)
  {
    int grid_size, block_size, shared_mem_size; 
    std::vector<int> grid_block_mem_sizes = CalculateGridAndBlockSizes(tensor->dims_prod);
    grid_size = grid_block_mem_sizes[0];
    block_size = grid_block_mem_sizes[1];

    tensor->Sync();
    /*
    if (tensor->thread_id!=thread_id)
    {
      cudaStream_t prev_stream = ThreadsStream[tensor->thread_id];
      cudaStream_t stream = ThreadsStream[thread_id];

      cudaStreamSynchronize(prev_stream);

      copy_tensor_kernel<<<grid_size,block_size,0,stream>>>(arg_tensor, tensor_ptr, dims_prod);

      StreamAwaitStreamB(prev_stream, stream);
    } else {
      cudaStream_t stream = ThreadsStream[thread_id];
      copy_tensor_kernel<<<grid_size,block_size,0,stream>>>(arg_tensor, tensor_ptr, dims_prod);
    }
    */
    cudaStream_t stream = ThreadsStream[thread_id];
    copy_tensor_kernel<<<grid_size,block_size,0,stream>>>(arg_tensor, tensor_ptr, dims_prod);
  }
  

  data_type_tensor *new_tensor = createTensor(arg_tensor, dims, dims_prod, true, arg_tensor_name, tensor->cuda_stream, tensor->loader);
  new_tensor->scopeless_name = tensor->scopeless_name;
  new_tensor->from_grad_or_load = tensor->from_grad_or_load;//
  NamedTensorsT[arg_tensor_name] = new_tensor;

  //if (thread_id!=0)
  //  ThreadedScopeTensorsToClean[thread_id][scope].push_back(arg_tensor_name);
  if (tensor->thread_id!=0)
  {
    //ThreadedScopeTensorsToClean[tensor->thread_id][previous_scope].erase(std::remove(ThreadedScopeTensorsToClean[tensor->thread_id][previous_scope].begin(), ThreadedScopeTensorsToClean[tensor->thread_id][previous_scope].end(), tensor_name), ThreadedScopeTensorsToClean[tensor->thread_id][previous_scope].end());
    threaded_Tensors_to_save[tensor->thread_id][previous_scope].push_back(tensor);
    threaded_tensors_to_save[tensor->thread_id][previous_scope].push_back(tensor->tensor_ptr);
  }

  return 0;
}