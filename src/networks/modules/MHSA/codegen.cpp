
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




void mhsa_backward(float *x, float *dx, float *dy, std::string name)
{
  std::unique_ptr<MHSA> mhsa = std::move(NamedMHSA[name]);

  mhsa->Backward(x, dx, dy);

  NamedMHSA[name] = std::move(mhsa);
}


extern "C" void *MHSAForward(char *self, data_type_tensor *tensor, int thread_id, char *conv_namec, int is_obj_attr_or_self)
{
  //TODO: remove self arg and concatenate it instead during the function call
  
  
  std::string _self = self;
  std::string conv_name = conv_namec;
  if (is_obj_attr_or_self)
    conv_name = _self + conv_name;

  //std::cout << "\nMHSA forward of " << conv_name << " with input " << tensor_q->name  << "\n";
  //std::cout << "thread id: " << thread_id << "\n\n";

  

  float *output;
  
  std::vector<float> dims = tensor->dims;
  

  float B = dims[0];
  float T = dims[dims.size()-2];
  float C = dims[dims.size()-1];

  //std::vector<float> new_dims = dims;
  std::vector<float> new_dims = {B, T, C};




  std::unique_ptr<MHSA> mhsa = std::move(NamedMHSA[conv_name]);

  if ((int)C!=(int)mhsa->C)
  {
    std::string error = "Input tensor channels are: " + std::to_string((int)C) + ", while the expected input channels of the MHSA are: " + std::to_string(mhsa->C);
    LogError(error);
    
    NamedMHSA[conv_name] = std::move(mhsa);
    return nullptr;
  }



  tensor->Sync();
  output = mhsa->Forward(tensor, (int) B, (int)T, thread_id);





  NamedMHSA[conv_name] = std::move(mhsa);

  //std::cout << "Return with dims:" << "\n";
  //PrintDims(new_dims);
  

  data_type_tensor *new_tensor = createTensor(output, new_dims, DimsProd(new_dims), false, "");
  new_tensor->AttrLNode(tensor, mhsa_op);
  new_tensor->scopeless_name = conv_name;
  return new_tensor;
}



extern "C" float CreateMHSAOnDemand(char *tensor_name, char *init,
                                      float nh, float C, float T, int_vec *notators)
{
  std::cout << "\nCreate mhsa on demand:\n   nh: " << nh << " C " << C << " T " << T << "\n";
  std::cout << "" << tensor_name << " " << init << "\n";

  std::unique_ptr<MHSA> mhsa = std::make_unique<MHSA>((int)nh, (int)T, (int) C, init, notators, tensor_name);

  std::cout << "Adding " << tensor_name << " to NamedMHSA dict\n";
  NamedMHSA[tensor_name] = std::move(mhsa);
  return 0;
}