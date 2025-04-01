
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
#include <string>



#include "../../../compiler_frontend/logging.h"
#include "../../../notators/notators.h"
#include "../../../tensor/include.h"
#include "../globals.h"
#include "class.h"




void linear_backward(float *x, float *dx, float *dy, std::string name)
{
  std::unique_ptr<Linear> linear = std::move(NamedLinear[name]);

  linear->Backward(x, dx, dy);

  NamedLinear[name] = std::move(linear);
}





extern "C" void *LinearForward(char *self, Tensor *tensor, int thread_id, char *conv_namec, int is_obj_attr_or_self)
{
  //TODO: remove self arg and concatenate it instead during the function call
  
  
  std::string _self = self;
  std::string conv_name = conv_namec;
  if (is_obj_attr_or_self)
    conv_name = _self + conv_name;

  //std::cout << "\nLinear of " << conv_name << " with input " << tensor->name  << "\n";
  //std::cout << "thread id: " << thread_id << "\n\n";

  

  float *output;
  
  std::vector<float> dims = tensor->dims;
  
  int C = dims[dims.size()-1];



  std::unique_ptr<Linear> linear = std::move(NamedLinear[conv_name]);

  if ((int)C!=(int)linear->C)
  {
    std::string error = "Input tensor channels are: " + std::to_string((int)C) + ", while the expected input channels of the Linear are: " + std::to_string(linear->C);
    LogError(error);
    
    NamedLinear[conv_name] = std::move(linear);
    return nullptr;
  }
    
  std::vector<float> new_dims = RemoveLastDim(dims);
  new_dims.push_back(linear->OC);
  



  tensor->Sync();
  output = linear->Forward(tensor, thread_id);


  NamedLinear[conv_name] = std::move(linear);

  

  Tensor *new_tensor = createTensor(output, new_dims, DimsProd(new_dims), false, "");
  new_tensor->AttrLNode(tensor, linear_op);
  new_tensor->scopeless_name = conv_name;
  return new_tensor;
}







extern "C" float CreateLinearOnDemand(char *tensor_name, char *init,
                                      float C, float OC, int_vec *Notators)
{
  std::cout << "\nCreate linear on demand:\n  C " << C << " OC " << OC << "\n";
  std::cout << "" << tensor_name << " " << init << "\n";


  std::unique_ptr<Linear> linear = std::make_unique<Linear>((int) C, int (OC), init, Notators, tensor_name);

  std::cout << "Adding " << tensor_name << " to NamedMHSA dict\n";
  NamedLinear[tensor_name] = std::move(linear);
  return 0;
}