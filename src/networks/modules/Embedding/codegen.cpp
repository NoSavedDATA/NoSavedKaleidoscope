
#include <cuda_runtime.h>
#include <cublasLt.h>
#include <cublas_v2.h>
#include <curand_kernel.h>
#include <cuda_fp16.h>
#include <cudnn.h>
#include <mma.h>

#include <string>
#include <vector>
#include <iostream>



#include "../../../compiler_frontend/logging.h"
#include "../../../tensor/include.h"
#include "../globals.h"
#include "class.h"




extern "C" void *EmbeddingForward(char *self, DT_tensor *tensor_x, int thread_id, char *conv_namec, int is_obj_attr_or_self)
{
  //TODO: remove self arg and concatenate it instead during the function call
  
  
  std::string _self = self;
  std::string conv_name = conv_namec;
  if (is_obj_attr_or_self)
    conv_name = _self + conv_name;

  //std::cout << "Embedding forward of " << conv_name << " with input " << tensor_x->name <<  "\n";



  float *tensor_ptr, *output;
  
  std::vector<int> dims = tensor_x->dims;
  int input_dims_prod = DimsProd(dims);


  std::unique_ptr<Embedding> embedding = std::move(NamedEmbedding[conv_name]);

  tensor_x->Sync();

  
  output = embedding->Forward(tensor_x, tensor_x->dims_prod, thread_id);
  

  int is_forward_func = 1;
  

  std::vector<int> new_dims = tensor_x->dims;
  new_dims.push_back(embedding->OC); 

  


  NamedEmbedding[conv_name] = std::move(embedding);
  
  

  return customOpTensor(output, new_dims, DimsProd(new_dims), "embedding_backward", conv_name, tensor_x);
}









extern "C" float CreateEmbeddingOnDemand(char *tensor_name, char *init,
  float C, float OC)
{
    std::cout << "\nCreate embedding on demand:\n   C: " << C << " OC " << OC << "\n";

    auto embedding = std::make_unique<Embedding>((int)C, (int)OC, init, tensor_name);

    std::cout << "Adding " << tensor_name << " to NamedEmbedding dict\n";
    NamedEmbedding[tensor_name] = std::move(embedding);
    return 0;
}

void embedding_backward(float *inp, int size, float *out,
                     float *dinp, float *dout,
                     std::string module_name, DT_tensor *node)
{
  std::unique_ptr<Embedding> embedding = std::move(NamedEmbedding[module_name]);

  embedding->Backward(inp, dout);

  NamedEmbedding[module_name] = std::move(embedding);
}