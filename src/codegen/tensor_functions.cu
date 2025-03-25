
#include<string>
#include<vector>
#include<map>
#include<cstring>
#include<random>
#include<thread>

#include "../common/include.h"
#include "../tensor/include.h"
#include "include.h"




extern "C" float CreateTensorOnDemand(char *tensor_name, char *scopeless_name, char *init, int is_weight, int thread_id, char *scope)
{
  //std::cout << "CREATING TENSOR " << tensor_name << " AT THREAD: " << thread_id << "\n";

  Tensor *tensor;

  std::vector<float> dims = NamedDims[tensor_name];
  NamedDims[tensor_name].clear(); //TODO: Global vars are bad with threads.

  int product = DimsProd(dims);

  float *tensor_ptr;
  float *tensor_cpu;

  if(product>0)
  {
    if (std::strcmp(init, "randu") == 0)
      tensor_cpu = make_random_float_uniform(product);
    if (std::strcmp(init, "zeros") == 0)
      tensor_cpu = make_zeros_float(product);
    if (std::strcmp(init, "ones") == 0)
      tensor_cpu = make_ones_float(product);
    if (std::strcmp(init, "normal") == 0)
      tensor_cpu = make_normal(product);
    if (std::strcmp(init, "xavu") == 0)
      tensor_cpu = make_xavier_uniform_float(product, dims[dims.size()-1], dims[dims.size()-2]);
    if (std::strcmp(init, "xavu_relu") == 0)
      tensor_cpu = make_xavier_uniform_float_relu(product, dims[dims.size()-1], dims[dims.size()-2]);
    if (std::strcmp(init, "xavu_tanh") == 0)
      tensor_cpu = make_xavier_uniform_float_tanh(product, dims[dims.size()-1], dims[dims.size()-2]);
    if (std::strcmp(init, "he_normal_relu") == 0)
      tensor_cpu = make_he_normal_float_relu(product, dims[dims.size()-1]);
    if (std::strcmp(init, "init_gpt") == 0)
      tensor_cpu = make_gpt_init(product);
    if (std::strcmp(init, "int") == 0)
      tensor_cpu = make_random_int(product, 10);
    if (std::strcmp(init, "arange") == 0)
      tensor_cpu = make_arange(product);
    if (std::strcmp(init, "binary") == 0)
      tensor_cpu = make_random_int(product, 1);

    cudaCheck(cudaGetLastError());
    std::string _name = "create tensor ";
    _name = _name + tensor_name;
    tensor_ptr = get_from_pool(thread_id, product, _name);
    //std::cout << "cpy of: " << tensor_name << "\n";

    cudaStream_t stream = ThreadsStream[thread_id];
    cudaCheck(cudaMemcpyAsync(tensor_ptr, tensor_cpu, product*sizeof(float), cudaMemcpyHostToDevice, stream));
    //cudaStreamSynchronize(stream);
    delete[] tensor_cpu;
  }


  
  /*
  if(NamedTensorsT.count(tensor_name)>0)
  {
    tensor = NamedTensorsT[tensor_name];
    if (tensor!=nullptr)
    
      delete tensor;
      //cudaCheck(cudaFree(aux_ptr));
  }
  */
  
  


  tensor = createTensor(tensor_ptr, dims, product, true, tensor_name);
  tensor->scopeless_name = scopeless_name;
  if((bool)is_weight)
    tensor->SetIsWeight();
  tensor->op = create_tensor_op;

  
  NamedTensorsT[tensor_name] = tensor;

  
  delete[] tensor_name;
  delete[] scopeless_name;


  return 0;
}





extern "C" void *gpu(int thread_id, Tensor *tensor, Tensor *pinned_tensor)
{
  //std::cout << "\nGpu transfer for: " << tensor.name << " on worker " << idx << "\n";
  
  float *tensor_ptr, *tensor_cpu;

  
  tensor_cpu = pinned_tensor->cpu_tensor_ptr;
  std::vector<float> dims = pinned_tensor->dims;
  float dims_prod = pinned_tensor->dims_prod;
  



  
  if (tensor->dims_prod==dims_prod)
    tensor_ptr = tensor->tensor_ptr;
  else
    tensor_ptr = get_from_pool(thread_id, dims_prod, "gpu");
  
  //tensor_ptr = get_from_pool(dims_prod, "gpu");


  
  Loader *loader=nullptr;
  CudaStreams *cuda_stream=nullptr;
  
  
  cuda_stream = AllocateStream(0);
  cudaMemcpyAsync(tensor_ptr, tensor_cpu, dims_prod * sizeof(float), cudaMemcpyHostToDevice, cuda_stream->stream);
  //cudaMemcpy(tensor_ptr, tensor_cpu, dims_prod * sizeof(float), cudaMemcpyHostToDevice);
  pinned_tensor->cuda_stream = cuda_stream;
  



  if (nn_mode==eval_mode)
  {

  } else {
    
    Tensor *attr_tensor;
    attr_tensor = createTensor(tensor_ptr, dims, dims_prod, true, "");
    attr_tensor->op = gpu_op;
    todo_backward_tensors.push_back(attr_tensor); // pass to gc
    
  }

  tensor->AttrTensor(tensor_ptr, dims, dims_prod, cuda_stream, loader);
  tensor->from_grad_or_load = true;

  return 0;
}



extern "C" float gpuw(int thread_id, Tensor *tensor, Tensor *pinned_tensor, float idx)
{
  //std::cout << "\nGpu transfer for: " << tensor->name << " on worker " << idx << "\n";
  
  float *tensor_ptr, *tensor_cpu;

  
  
  std::vector<float> dims, batchless_dims;
  dims = pinned_tensor->dims;
  

  batchless_dims = BatchLessDims(dims);
  float batchless_dims_prod = (float)DimsProd(batchless_dims);


  tensor_cpu = pinned_tensor->cpu_tensor_ptr + static_cast<int>(idx*batchless_dims_prod);

  
  if (tensor->dims_prod==batchless_dims_prod)
    tensor_ptr = tensor->tensor_ptr;
  else
    tensor_ptr = get_from_pool(thread_id, batchless_dims_prod, "gpuw");
  
  //tensor_ptr = get_from_pool(batchless_dims_prod, "gpuw");


  
  Loader *loader=nullptr;
  CudaStreams *cuda_stream=nullptr;
  
  
  if (batchless_dims_prod<2000){
    cudaMemcpy(tensor_ptr, tensor_cpu, batchless_dims_prod * sizeof(float), cudaMemcpyHostToDevice);
  }
  else// if (batchless_dims_prod<1000)
  {
    cuda_stream = AllocateStream(0);
    cudaMemcpyAsync(tensor_ptr, tensor_cpu, batchless_dims_prod * sizeof(float), cudaMemcpyHostToDevice, cuda_stream->stream);
    //cudaMemcpy(tensor_ptr, tensor_cpu, batchless_dims_prod * sizeof(float), cudaMemcpyHostToDevice);
    pinned_tensor->cuda_stream = cuda_stream;
  }
  /*
  else
  {
    //cuda_stream = AllocateStream(0);
    //cudaMemcpyAsync(tensor_ptr, tensor_cpu, batchless_dims_prod * sizeof(float), cudaMemcpyHostToDevice, cuda_stream->stream);
    loader = new Loader();
    loader->Load(tensor_ptr, tensor_cpu, batchless_dims_prod);
  }
  */



  if (nn_mode==eval_mode)
  {

  } else {
    
    Tensor *attr_tensor;
    attr_tensor = createTensor(tensor_ptr, batchless_dims, batchless_dims_prod, true, "");
    attr_tensor->op = gpu_op;
    todo_backward_tensors.push_back(attr_tensor); // pass to gc
    
  }

  tensor->AttrTensor(tensor_ptr, batchless_dims, batchless_dims_prod, cuda_stream, loader);
  tensor->from_grad_or_load = true;

  return 0;
}