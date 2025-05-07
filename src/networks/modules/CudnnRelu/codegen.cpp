
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






extern "C" void *ReluForward(char *self, DT_tensor *tensor, char *conv_namec, int is_obj_attr_or_self)
{
  
  //TODO: remove self arg and concatenate it instead during the function call
  //std::cout << "\nReluForward2d " << conv_namec << " and tensor " << tensor.name << "\n";
  
  std::string _self = self;
  std::string conv_name = conv_namec;
  if (is_obj_attr_or_self)
    conv_name = _self + conv_name;


  float *tensor_ptr, *output;
  tensor_ptr = tensor->tensor_ptr;
  std::vector<float> dims = tensor->dims;
  float input_dims_prod = DimsProd(dims);

  float B = dims[0];
  float C = dims[dims.size()-3];
  float H = dims[dims.size()-2];
  float W = dims[dims.size()-1];



  std::unique_ptr<Relu> conv = std::move(NamedRelu[conv_name]);


  tensor->Sync();
  output = conv->Forward(tensor, H, W, B, C);

  float resultingDimsProd = B * (float)C * (float)H * (float)W;

  
  
  std::vector<float> bn_dims = {(float)C};
  std::string bias_name = conv_name+"_bias";

  DT_tensor *scale_bias_tensor, *scale_tensor, *bias_tensor;




  NamedRelu[conv_name] = std::move(conv);

  std::vector<float> new_dims = {(float)B, (float)C, (float)H, (float)W};
  DT_tensor *new_tensor = createTensor(output, new_dims, DimsProd(new_dims), false, conv_name);
  new_tensor->AttrLNode(tensor, cudnn_relu_op);
  new_tensor->from_cudnn = conv_name;
  return new_tensor;
}




void cudnn_relu_backward(float *inp, float *out,
                     float *dinp, 
                     float *dout, std::string conv_name)
{

  //std::cout << "batchnorm2d_backward for " << conv_name << "\n";
  std::unique_ptr<Relu> conv = std::move(NamedRelu[conv_name]);

  conv->Backward(inp, out, dinp, dout);

  NamedRelu[conv_name] = std::move(conv);

}





extern "C" float CreateReluOnDemand(char *tensor_name)
{
  auto conv = std::make_unique<Relu>(tensor_name);

  NamedRelu[tensor_name] = std::move(conv);
  return 0;
}