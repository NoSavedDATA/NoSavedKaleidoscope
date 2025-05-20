
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




void mhsa_backward(float *inp, float size, float *out,
                     float *dinp, float *dout,
                     std::string module_name, DT_tensor *node)
{
  std::unique_ptr<MHSA> mhsa = std::move(NamedMHSA[module_name]);

  mhsa->Backward(inp, dinp, dout);

  NamedMHSA[module_name] = std::move(mhsa);
}


extern "C" void *MHSAForward(char *self, DT_tensor *tensor, int thread_id, char *conv_namec, int is_obj_attr_or_self)
{
  //TODO: remove self arg and concatenate it instead during the function call
  
  
  std::string _self = self;
  std::string conv_name = conv_namec;
  if (is_obj_attr_or_self)
    conv_name = _self + conv_name;

  //std::cout << "\nMHSA forward of " << conv_name << " with input " << tensor_q->name  << "\n";
  //std::cout << "thread id: " << thread_id << "\n\n";

  

  float *output;
  
  std::vector<int> dims = tensor->dims;
  

  int B = dims[0];
  int T = dims[dims.size()-2];
  int C = dims[dims.size()-1];

  //std::vector<int> new_dims = dims;
  std::vector<int> new_dims = {B, T, C};




  std::unique_ptr<MHSA> mhsa = std::move(NamedMHSA[conv_name]);

  if (C!=mhsa->C)
  {
    std::string error = "Input tensor channels are: " + std::to_string(C) + ", while the expected input channels of the MHSA are: " + std::to_string(mhsa->C);
    LogError(error);
    
    NamedMHSA[conv_name] = std::move(mhsa);
    return nullptr;
  }



  tensor->Sync();
  output = mhsa->Forward(tensor, (int) B, (int)T, thread_id);





  NamedMHSA[conv_name] = std::move(mhsa);

  //std::cout << "Return with dims:" << "\n";
  //PrintDims(new_dims);
  


  return customOpTensor(output, new_dims, DimsProd(new_dims), "mhsa_backward", conv_name, tensor);
}



extern "C" float CreateMHSAOnDemand(char *tensor_name, char *init,
                                      float nh, float C, float T, int_vec *notators)
{
  std::cout << "\nCreate mhsa on demand:\n   nh: " << nh << " C " << C << " T " << T << "\n";
  std::cout << "" << tensor_name << " " << init << "\n";

  std::unique_ptr<MHSA> mhsa = std::make_unique<MHSA>(nh, T, C, init, notators, tensor_name);

  std::cout << "Adding " << tensor_name << " to NamedMHSA dict\n";
  NamedMHSA[tensor_name] = std::move(mhsa);
  return 0;
}