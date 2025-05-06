
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



extern "C" void *BN2dReluForward(char *self, data_type_tensor *tensor, int thread_id, char *conv_namec, int is_obj_attr_or_self)
{
  //TODO: remove self arg and concatenate it instead during the function call
  //std::cout << "\nBN2dReluForward2d " << conv_namec << " and tensor " << tensor.name << "\n";
  
  std::string _self = self;
  std::string conv_name = conv_namec;
  if (is_obj_attr_or_self)
    conv_name = _self + conv_name;

  std::cout << "Conv forward for conv: " << conv_name <<"\n";
  

  float *tensor_ptr, *output;
  tensor_ptr = tensor->tensor_ptr;
  std::vector<float> dims = tensor->dims;
  float input_dims_prod = DimsProd(dims);

  float B = dims[0];
  float C = dims[dims.size()-3];
  float H = dims[dims.size()-2];
  float W = dims[dims.size()-1];




  std::unique_ptr<BN2dRelu> conv = std::move(NamedBN2dRelu[conv_name]);

  if ((int)C!=(int)conv->C)
  {
    std::string error = "Input tensor channels are: " + std::to_string((int)C) + ", while the expected input channels of the BN2dRelu are: " + std::to_string(conv->C);
    LogError(error);
    
    NamedBN2dRelu[conv_name] = std::move(conv);
    return nullptr;
  }


  tensor->Sync();
  output = conv->Forward(tensor, H, W, B, C, thread_id);

  float resultingDimsProd = B * (float)C * (float)H * (float)W;

  
  
  std::vector<float> bn_dims = {(float)C};
  std::string bias_name = conv_name+"_bias";

  data_type_tensor *scale_bias_tensor, *scale_tensor, *bias_tensor;

  // for the backprop
  scale_bias_tensor = createTensor(conv->scale, bn_dims, C, true, conv_name);
  scale_bias_tensor->SetBias(conv->bias, C);
  scale_bias_tensor->SetIsWeight();


  // for the optimizer only
  scale_tensor = NamedTensorsT[conv_name];
  scale_tensor->NewTensor(conv->scale, bn_dims, C, true, conv_name);
  scale_tensor->SetIsWeight();
  
  bias_tensor = NamedTensorsT[bias_name];
  bias_tensor->NewTensor(conv->bias, bn_dims, C, true, conv_name);
  bias_tensor->SetIsWeight();



  NamedBN2dRelu[conv_name] = std::move(conv);

  std::vector<float> new_dims = {(float)B, (float)C, (float)H, (float)W};
  data_type_tensor *new_tensor = createTensor(output, new_dims, DimsProd(new_dims), false, conv_name);
  new_tensor->AttrNodes(tensor, scale_bias_tensor, bn2drelu);
  new_tensor->from_cudnn = conv_name;
  return new_tensor;
}


void bn2drelu_backward(float *inp, float *intermediate, float *out,
                     float *dinp, float *dw, float *db, float *dintermediate,
                     float *dout, std::string conv_name)
{

  std::unique_ptr<BN2dRelu> conv = std::move(NamedBN2dRelu[conv_name]);

  conv->Backward(inp, intermediate, out, dinp, dw, db, dintermediate, dout);

  NamedBN2dRelu[conv_name] = std::move(conv);

}

extern "C" float CreateBN2dReluOnDemand(char *tensor_name, float C)
{
  std::cout << "\nCreate BatchNorm2d on demand:\n   C: " << C  << "\n";

  auto conv = std::make_unique<BN2dRelu>((int)C, tensor_name);

  NamedBN2dRelu[tensor_name] = std::move(conv);
  return 0;
}