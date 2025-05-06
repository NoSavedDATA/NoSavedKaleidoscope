
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



// extern "C" void *MaxPoolForward2d(char *self, data_type_tensor *tensor, int thread_id, char *pool_namec, int is_obj_attr_or_self)
extern "C" void *Pool2d(Scope_Struct *scope_struct, data_type_tensor *tensor)
{
  //std::cout << "MaxPoolForward2d of " << pool_namec << " and tensor " << tensor.name << "\n";
  

  std::string pool_name = scope_struct->first_arg;
  int thread_id = scope_struct->thread_id;

  //std::cout << "Conv forward for  conv: " << pool_name <<"\n";
  

  float *tensor_ptr, *output, *d_filter;
  tensor_ptr = tensor->tensor_ptr;
  std::vector<float> dims = tensor->dims;
  float input_dims_prod = DimsProd(dims);

  float B = dims[0];
  float C = dims[dims.size()-3];
  float H = dims[dims.size()-2];
  float W = dims[dims.size()-1];
  float OC = C;


  std::unique_ptr<MaxPool2dCPP> conv = std::move(NamedMaxPool2d[pool_name]);

  tensor->Sync();
  output = conv->Forward(tensor, H, W, B, C, thread_id);


  
  
  float resultingDimsProd = B * (float)OC * (float)conv->out_W * (float)conv->out_W;



  std::vector<float> new_dims = {(float)B, (float)OC, (float)conv->out_H, (float)conv->out_W};
  

  NamedMaxPool2d[pool_name] = std::move(conv);


  return customOpTensor(output, new_dims, DimsProd(new_dims), "pool2d_backward", pool_name, tensor);
}





void pool2d_backward(float *inp, float size, float *out,
                     float *dinp, float *dout,
                     std::string pool_name)
{
  //std::cout << "maxpool2d_backward of " << pool_name << "\n";
  std::unique_ptr<MaxPool2dCPP> conv = std::move(NamedMaxPool2d[pool_name]);

  conv->Backward(inp, out, dinp, dout);

  NamedMaxPool2d[pool_name] = std::move(conv);

  
}


extern "C" float Pool2d_Create(char *name, char *scopeless_name, void *init_val, data_type_list *notes_vector, Scope_Struct *scope_struct)
{
  std::cout << "Pool2d_Create execution" << ".\n";
  
  float ks = notes_vector->get<float>(0);
  float stride = notes_vector->get<float>(1);
  float padding = notes_vector->get<float>(2);
  
  std::cout << "\nCreate maxpool2d on demand:\n" << "   ks " << ks << " stride " << stride << " padding " << padding << "\n";
  char *type = "max";
  if(notes_vector->data->size()==4) 
  type = notes_vector->get<char *>(3);


  auto maxpool = std::make_unique<MaxPool2dCPP>((int)ks, (int)stride, (int)padding, type);

  NamedMaxPool2d[name] = std::move(maxpool);
  return 0;
}