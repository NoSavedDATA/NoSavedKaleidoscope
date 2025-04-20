
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
#include "../../../mangler/scope_struct.h"
#include "../../../tensor/include.h"
#include "../globals.h"
#include "class.h"



void conv2d_backward(float *inp,  float *weight,
                     float *dinp, float *dw,
                     float *dout, std::string conv_name)
{

  //std::cout << "conv2d_backward for " << conv_name << "\n";
  std::unique_ptr<Conv2d> conv = std::move(NamedConv2d[conv_name]);

  conv->Backward(inp, dinp, dw, dout);


  NamedConv2d[conv_name] = std::move(conv);

  
}




extern "C" void *ConvForward2d(Scope_Struct *scope_struct, Tensor *tensor)
{
  //TODO: remove self arg and concatenate it instead during the function call
  
  

  std::string conv_name = scope_struct->first_arg;
  int thread_id = scope_struct->thread_id;


  std::cout << "Conv forward of " << conv_name << " and tensor " << tensor->name << "\n";
  // std::cout << "Conv forward for conv: " << conv_name <<"\n";
  

  float *tensor_ptr, *output, *d_filter;
  tensor_ptr = tensor->tensor_ptr;
  std::vector<float> dims = tensor->dims;
  float input_dims_prod = DimsProd(dims);

  float B = dims[0];
  float C = dims[dims.size()-3];
  float H = dims[dims.size()-2];
  float W = dims[dims.size()-1];



  std::unique_ptr<Conv2d> conv = std::move(NamedConv2d[conv_name]);



  if ((int)C!=(int)conv->C)
  {
    std::string error = "Input tensor channels are: " + std::to_string((int)C) + ", while the expected input channels of the convolution are: " + std::to_string(conv->C);
    LogError(error);
    
    NamedConv2d[conv_name] = std::move(conv);
    return nullptr;
  }



  tensor->Sync();

  output = conv->Forward(tensor, H, W, B, thread_id);

  int ks_H = conv->ks;
  int ks_W = conv->ks;


  
  
  float resultingDimsProd = B * (float)conv->OC * (float)conv->out_H * (float)conv->out_W;

  int is_forward_func = 1;
  


  std::vector<float> new_dims = {(float)conv->B, (float)conv->OC, (float)conv->out_H, (float)conv->out_W};
  

  //for backprop:
  std::vector<float> kernel_dims = {(float)conv->OC, (float)C, (float)conv->ks, (float)conv->ks}; 




  Tensor *conv_tensor = NamedTensorsT[conv_name];
  conv_tensor->NewTensor(conv->d_filter, kernel_dims, DimsProd(kernel_dims), true, conv_name);
  conv_tensor->SetIsWeight();
  

  NamedConv2d[conv_name] = std::move(conv);

  Tensor *new_tensor = createTensor(output, new_dims, DimsProd(new_dims), false, "");
  new_tensor->AttrNodes(tensor, conv_tensor, conv2d);
  new_tensor->from_cudnn = conv_name;
  return new_tensor;
}





extern "C" float CreateConv2dOnDemand(char *tensor_name, char *init,
                                      float C, float OC, float ks, float stride, float padding)
{
  std::cout << "\nCreate conv on demand:\n   C: " << C << " OC " << OC << " ks " << ks << " stride " << stride << " padding " << padding << "\n";

  auto conv = std::make_unique<Conv2d>((int)C, (int)OC, (int)ks, (int)stride, (int)padding, init, tensor_name);

  std::cout << "Adding " << tensor_name << " to NamedConv2d dict\n";
  NamedConv2d[tensor_name] = std::move(conv);
  return 0;
}