
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



extern "C" void *BatchNorm2d(Scope_Struct *scope_struct, DT_tensor *tensor)
{
  std::string bn_name = scope_struct->first_arg;
  int thread_id = scope_struct->thread_id;

  // std::cout << "BatchNorm2d forward for " << bn_name <<"\n";
  

  float *tensor_ptr, *output;
  tensor_ptr = tensor->tensor_ptr;
  std::vector<float> dims = tensor->dims;
  float input_dims_prod = DimsProd(dims);

  float B = dims[0];
  float C = dims[dims.size()-3];
  float H = dims[dims.size()-2];
  float W = dims[dims.size()-1];



  std::unique_ptr<BatchNorm2dCPP> conv = std::move(NamedBatchNorm2d[bn_name]);

  if ((int)C!=(int)conv->C)
  {
    std::string error = "Input tensor channels are: " + std::to_string((int)C) + ", while the expected input channels of the BatchNorm2d are: " + std::to_string(conv->C);
    LogError(error);
    
    NamedBatchNorm2d[bn_name] = std::move(conv);
    return nullptr;
  }


  tensor->Sync();
  output = conv->Forward(tensor, H, W, B, C, thread_id);

  

  NamedBatchNorm2d[bn_name] = std::move(conv);

  std::vector<float> new_dims = {(float)B, (float)C, (float)H, (float)W};

  return customOpTensor(output, new_dims, DimsProd(new_dims), "batchnorm2d_backward", bn_name, tensor);
}


// void batchnormd2d_backward(float *inp, 
//   float *dinp, float *dw, float *db,
  // float *dout, std::string bn_name)
void batchnorm2d_backward(float *inp, float size, float *out,
  float *dinp, float *dout,
  std::string bn_name)
{

  // std::cout << "batchnorm2d_backward for " << bn_name << "\n";
  std::unique_ptr<BatchNorm2dCPP> bn = std::move(NamedBatchNorm2d[bn_name]);

  bn->Backward(inp, dinp, dout);

  NamedBatchNorm2d[bn_name] = std::move(bn);
}


// extern "C" float CreateBatchNorm2dOnDemand(char *tensor_name, float C)
extern "C" float BatchNorm2d_Create(char *name, char *scopeless_name, void *init_val, DT_list *notes_vector, Scope_Struct *scope_struct)
{
  if (notes_vector->data->size()<1)
    LogErrorS("BatchNorm2d requires input channels information.");

  float C = notes_vector->get<float>(0);
  // std::cout << "\nCreate BatchNorm2d " << name << " on demand:\n   C: " << C  << "\n";
  auto conv = std::make_unique<BatchNorm2dCPP>((int)C, name);


  NamedBatchNorm2d[name] = std::move(conv);
  
  return 0;
}