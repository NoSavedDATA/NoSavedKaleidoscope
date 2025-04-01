
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



extern "C" void *MaxPoolForward2d(char *self, Tensor *tensor, int thread_id, char *conv_namec, int is_obj_attr_or_self)
{
  //std::cout << "MaxPoolForward2d of " << conv_namec << " and tensor " << tensor.name << "\n";
  
  std::string _self = self;
  std::string conv_name = conv_namec;
  if (is_obj_attr_or_self)
    conv_name = _self + conv_name;

  //std::cout << "Conv forward for  conv: " << conv_name <<"\n";
  

  float *tensor_ptr, *output, *d_filter;
  tensor_ptr = tensor->tensor_ptr;
  std::vector<float> dims = tensor->dims;
  float input_dims_prod = DimsProd(dims);

  float B = dims[0];
  float C = dims[dims.size()-3];
  float H = dims[dims.size()-2];
  float W = dims[dims.size()-1];
  float OC = C;


  std::unique_ptr<MaxPool2d> conv = std::move(NamedMaxPool2d[conv_name]);

  tensor->Sync();
  output = conv->Forward(tensor, H, W, B, C, thread_id);


  
  
  float resultingDimsProd = B * (float)OC * (float)conv->out_W * (float)conv->out_W;



  std::vector<float> new_dims = {(float)B, (float)OC, (float)conv->out_H, (float)conv->out_W};
  

  NamedMaxPool2d[conv_name] = std::move(conv);

  Tensor *new_tensor = createTensor(output, new_dims, DimsProd(new_dims), false, conv_name);
  new_tensor->AttrLNode(tensor, maxpool2d);
  new_tensor->from_cudnn = conv_name;
  return new_tensor;
}





void maxpool2d_backward(float *inp,  float *out,
                     float *dinp,
                     float *dout, std::string conv_name)
{
  //std::cout << "maxpool2d_backward of " << conv_name << "\n";
  std::unique_ptr<MaxPool2d> conv = std::move(NamedMaxPool2d[conv_name]);

  conv->Backward(inp, out, dinp, dout);

  NamedMaxPool2d[conv_name] = std::move(conv);

  
}


extern "C" float CreateMaxPool2dOnDemand(char *tensor_name, char *type, float ks, float stride, float padding)
{
  std::cout << "\nCreate maxpool2d on demand:\n" << "   ks " << ks << " stride " << stride << " padding " << padding << "\n";

  auto maxpool = std::make_unique<MaxPool2d>((int)ks, (int)stride, (int)padding, type);

  NamedMaxPool2d[tensor_name] = std::move(maxpool);
  return 0;
}