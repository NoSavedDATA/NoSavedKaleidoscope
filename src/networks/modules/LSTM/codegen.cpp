
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



void lstm_backward(float *x, float *dx, float *dy, std::string name)
{
  std::unique_ptr<LSTM> lstm = std::move(NamedLSTM[name]);

  lstm->Backward(x, dx, dy);

  NamedLSTM[name] = std::move(lstm);
}




extern "C" void *LSTMForward(char *self, data_type_tensor *tensor_x, data_type_tensor *tensor_ht, data_type_tensor *tensor_ct, int thread_id, char *conv_namec, int is_obj_attr_or_self)
{
  //TODO: remove self arg and concatenate it instead during the function call
  
  
  std::string _self = self;
  std::string conv_name = conv_namec;
  if (is_obj_attr_or_self)
    conv_name = _self + conv_name;

  //std::cout << "LSTM forward of " << conv_name << " with input " << tensor_x->name << ", ht: " << tensor_ht->name << "\n";


  

  float *tensor_ptr, *output, *d_filter;
  
  std::vector<float> dims = tensor_x->dims;
  float input_dims_prod = DimsProd(dims);

  float B = dims[0];
  float T = dims[dims.size()-2];
  float C = dims[dims.size()-1];



  std::unique_ptr<LSTM> lstm = std::move(NamedLSTM[conv_name]);

  if ((int)C!=(int)lstm->C)
  {
    std::string error = "Input tensor channels are: " + std::to_string((int)C) + ", while the expected input channels of the LSTM are: " + std::to_string(lstm->C);
    LogError(error);
    
    NamedLSTM[conv_name] = std::move(lstm);
    return nullptr;
  }



  tensor_x->Sync();
  tensor_ht->Sync();
  tensor_ct->Sync();

  output = lstm->Forward(tensor_x, tensor_ht, tensor_ct, (int) B, (int)T, thread_id);


  

  int is_forward_func = 1;
  


  std::vector<float> new_dims = {(float)B, (float)lstm->OC}; 



  /*
  data_type_tensor *conv_tensor = NamedTensorsT[conv_name];
  conv_tensor->NewTensor(conv->d_filter, kernel_dims, DimsProd(kernel_dims), true, conv_name);
  conv_tensor->SetIsWeight();
  */

  NamedLSTM[conv_name] = std::move(lstm);
  
  
  //std::cout << "Returning from lstm forward."  << "\n";

  //data_type_tensor *aux = createTensor(nullptr, {}, 0, false, conv_name);

  data_type_tensor *new_tensor = createTensor(output, new_dims, DimsProd(new_dims), false, "");
  new_tensor->AttrLNode(tensor_x, lstm_op);
  new_tensor->scopeless_name = conv_name;
  return new_tensor;
}




extern "C" float CreateLSTMOnDemand(char *tensor_name, char *init,
                                      float C, float OC)
{
  std::cout << "\nCreate lstm on demand:\n   C: " << C << " OC " << OC << "\n";

  auto lstm = std::make_unique<LSTM>((int)C, (int)OC, init, tensor_name);

  std::cout << "Adding " << tensor_name << " to NamedLSTM dict\n";
  NamedLSTM[tensor_name] = std::move(lstm);
  return 0;
}