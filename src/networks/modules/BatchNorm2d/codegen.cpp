
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



extern "C" DT_tensor *BatchNorm2d(Scope_Struct *scope_struct, DT_tensor *tensor)
{
  std::string bn_name = scope_struct->first_arg;
  int thread_id = scope_struct->thread_id;

  // std::cout << "BatchNorm2d forward for " << bn_name <<"\n";
  

  float *tensor_ptr, *output;
  tensor_ptr = tensor->tensor_ptr;
  std::vector<int> dims = tensor->dims;
  int input_dims_prod = DimsProd(dims);

  int B = dims[0];
  int C = dims[dims.size()-3];
  int H = dims[dims.size()-2];
  int W = dims[dims.size()-1];



  std::unique_ptr<BatchNorm2dCPP> conv = std::move(NamedBatchNorm2d[bn_name]);

  if ((int)C!=(int)conv->C)
  {
    std::string error = "Input tensor channels are: " + std::to_string(C) + ", while the expected input channels of the BatchNorm2d are: " + std::to_string(conv->C);
    LogError(error);
    
    NamedBatchNorm2d[bn_name] = std::move(conv);
    return nullptr;
  }


  tensor->Sync();
  output = conv->Forward(tensor, H, W, B, C, thread_id);

  

  NamedBatchNorm2d[bn_name] = std::move(conv);

  std::vector<int> new_dims = {B, C, H, W};

  return customOpTensor(output, new_dims, DimsProd(new_dims), "batchnorm2d_backward", nullptr, tensor);
}


// void batchnormd2d_backward(float *inp, 
//   float *dinp, float *dw, float *db,
  // float *dout, std::string bn_name)
void batchnorm2d_backward(float *inp, int size, float *out,
                     float *dinp, float *dout,
                     void *network_module, DT_tensor *node)
{

  // std::cout << "batchnorm2d_backward for " << bn_name << "\n";
  BatchNorm2dCPP *bn = (BatchNorm2dCPP*) network_module;
  bn->Backward(inp, dinp, dout);
}


// extern "C" float CreateBatchNorm2dOnDemand(char *tensor_name, float C)
extern "C" float BatchNorm2d_Create(Scope_Struct *scope_struct, char *name, char *scopeless_name, void *init_val, DT_list *notes_vector)
{

  if (notes_vector->size<1)
    LogErrorS("BatchNorm2d requires input channels information.");

  int C = notes_vector->get<int>(0);
  // std::cout << "\nCreate BatchNorm2d " << name << " on demand:\n   C: " << C  << "\n";
  auto conv = std::make_unique<BatchNorm2dCPP>(C, name);


  NamedBatchNorm2d[name] = std::move(conv);
  
  return 0;
}