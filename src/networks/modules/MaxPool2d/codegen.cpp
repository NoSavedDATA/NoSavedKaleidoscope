
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
#include "../../../data_types/codegen_notes.h"
#include "../../../tensor/include.h"
#include "../globals.h"
#include "class.h"



// extern "C" void *MaxPoolForward2d(char *self, DT_tensor *tensor, int thread_id, char *pool_namec, int is_obj_attr_or_self)
extern "C" void *Pool2d(Scope_Struct *scope_struct, DT_tensor *tensor)
{
  //std::cout << "MaxPoolForward2d of " << pool_namec << " and tensor " << tensor.name << "\n";
  

  std::string pool_name = scope_struct->first_arg;
  int thread_id = scope_struct->thread_id;

  //std::cout << "Conv forward for  conv: " << pool_name <<"\n";
  

  float *tensor_ptr, *output, *d_filter;
  tensor_ptr = tensor->tensor_ptr;
  std::vector<int> dims = tensor->dims;
  float input_dims_prod = DimsProd(dims);

  int B = dims[0];
  int C = dims[dims.size()-3];
  int H = dims[dims.size()-2];
  int W = dims[dims.size()-1];
  int OC = C;


  std::unique_ptr<MaxPool2dCPP> conv = std::move(NamedMaxPool2d[pool_name]);

  tensor->Sync();
  output = conv->Forward(tensor, H, W, B, C, thread_id);


  
  
  float resultingDimsProd = B * OC * conv->out_W * conv->out_W;



  std::vector<int> new_dims = {B, OC, conv->out_H, conv->out_W};
  

  NamedMaxPool2d[pool_name] = std::move(conv);


  return customOpTensor(output, new_dims, DimsProd(new_dims), "pool2d_backward", pool_name, tensor);
}





void pool2d_backward(float *inp, int size, float *out,
                     float *dinp, float *dout,
                     std::string module_name, DT_tensor *node)
{
  //std::cout << "maxpool2d_backward of " << pool_name << "\n";
  std::unique_ptr<MaxPool2dCPP> conv = std::move(NamedMaxPool2d[module_name]);

  conv->Backward(inp, out, dinp, dout);

  NamedMaxPool2d[module_name] = std::move(conv);

  
}


extern "C" float Pool2d_Create(Scope_Struct *scope_struct, char *name, char *scopeless_name, void *init_val, DT_list *notes_vector)
{
  std::cout << "Pool2d_Create execution" << ".\n";
  
  int ks = notes_vector->get<int>(0);
  int stride = notes_vector->get<int>(1);
  int padding = notes_vector->get<int>(2);
  
  std::cout << "\nCreate maxpool2d on demand:\n" << "   ks " << ks << " stride " << stride << " padding " << padding << "\n";
  char *type = "max";
  if(notes_vector->data->size()==4) 
  type = notes_vector->get<char *>(3);


  auto maxpool = std::make_unique<MaxPool2dCPP>(ks, stride, padding, type);

  NamedMaxPool2d[name] = std::move(maxpool);
  return 0;
}