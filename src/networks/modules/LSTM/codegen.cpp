
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



#include "../../../compiler_frontend/include.h"
#include "../../../tensor/include.h"
#include "../globals.h"
#include "class.h"



void lstm_backward(float *inp, int size, float *out,
                     float *dinp, float *dout,
                     void *network_module, DT_tensor *node)
{
  DT_LSTM *lstm = (DT_LSTM*) network_module;

  lstm->Backward(inp, dinp, dout);
}




extern "C" DT_tensor *LSTM(Scope_Struct *scope_struct, DT_tensor *tensor, DT_tensor *tensor_ht, DT_tensor *tensor_ct)
{
  //TODO: remove self arg and concatenate it instead during the function call
  
  std::string conv_name = scope_struct->first_arg;
  int thread_id = scope_struct->thread_id;
  

  //std::cout << "LSTM forward of " << conv_name << " with input " << tensor->name << ", ht: " << tensor_ht->name << "\n";


  

  float *tensor_ptr, *output, *d_filter;
  
  std::vector<int> dims = tensor->dims;
  int input_dims_prod = DimsProd(dims);

  int B = dims[0];
  int T = dims[dims.size()-2];
  int C = dims[dims.size()-1];



  std::unique_ptr<DT_LSTM> lstm = std::move(NamedLSTM[conv_name]);

  if ((int)C!=(int)lstm->C)
  {
    std::string error = "Input tensor channels are: " + std::to_string((int)C) + ", while the expected input channels of the LSTM are: " + std::to_string(lstm->C);
    LogError(error);
    
    NamedLSTM[conv_name] = std::move(lstm);
    return nullptr;
  }



  tensor->Sync();
  tensor_ht->Sync();
  tensor_ct->Sync();

  output = lstm->Forward(tensor, tensor_ht, tensor_ct, (int) B, (int)T, thread_id);


  

  int is_forward_func = 1;
  


  std::vector<int> new_dims = {B, lstm->OC}; 



  /*
  DT_tensor *conv_tensor = NamedTensorsT[conv_name];
  conv_tensor->NewTensor(conv->d_filter, kernel_dims, DimsProd(kernel_dims), true, conv_name);
  conv_tensor->SetIsWeight();
  */

  NamedLSTM[conv_name] = std::move(lstm);
  
  


 
  return customOpTensor(output, new_dims, DimsProd(new_dims), "lstm_backward", nullptr, tensor);
}




extern "C" float LSTM_Create(Scope_Struct *scope_struct, char *name, char *scopeless_name, void *init_val, DT_list *notes_vector)
{
  
  std::string init = "xavu";

  int C = notes_vector->get<int>(0);
  int OC = notes_vector->get<int>(1);

  std::vector<std::string> notes; 

  for (int i=2; i<notes_vector->size; i++)
  {
    if(notes_vector->data_types->at(i)=="str")
    {
      char *note = notes_vector->get<char *>(i);
      std::string note_str = note;
      if (in_str(note, tensor_inits))
        init = note;
      else
        notes.push_back(note);
    }
  }


  std::cout << "\nCreate lstm on demand:\n   C: " << C << " OC " << OC << "\n";

  auto lstm = std::make_unique<DT_LSTM>(C, OC, init, name);

  std::cout << "Adding " << name << " to NamedLSTM dict\n";
  NamedLSTM[name] = std::move(lstm);
  return 0;
}