
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



#include "../../../compiler_frontend/include.h"
#include "../../../data_types/codegen_notes.h"
#include "../../../mangler/scope_struct.h"
#include "../../../notators/notators.h"
#include "../../../tensor/include.h"
#include "../globals.h"
#include "class.h"




void linear_backward(float *inp, int size, float *out,
                     float *dinp, float *dout,
                     std::string module_name, DT_tensor *node)
{
  std::unique_ptr<LinearCPP> linear = std::move(NamedLinear[module_name]);

  linear->Backward(inp, dinp, dout);

  NamedLinear[module_name] = std::move(linear);
}










extern "C" DT_tensor *Linear(Scope_Struct *scope_struct, DT_tensor *tensor)
{
  // std::cout << "-------------------------------------CALLING LINEAR " << scope_struct->first_arg << ".\n";
  int thread_id = scope_struct->thread_id;
  
  
  std::string conv_name = scope_struct->first_arg;

  // std::cout << "\nLinear of " << conv_name << " with input " << tensor->name  << "\n";
  // std::cout << "thread id: " << thread_id << "\n\n";

  

  float *output;
  
  std::vector<int> dims = tensor->dims;
  
  int C = dims[dims.size()-1];



  std::unique_ptr<LinearCPP> linear = std::move(NamedLinear[conv_name]);

  if ((int)C!=(int)linear->C)
  {
    std::string error = "Input tensor last dim is: " + std::to_string((int)C) + ", while the expected input dim of the Linear layer is: " + std::to_string(linear->C);
    LogError(error);
    
    NamedLinear[conv_name] = std::move(linear);
    return nullptr;
  }
    
  std::vector<int> new_dims = RemoveLastDim(dims);
  new_dims.push_back(linear->OC);
  

  tensor->Sync();
  output = linear->Forward(tensor, thread_id);


  NamedLinear[conv_name] = std::move(linear);  


  DT_tensor *new_tensor = customOpTensor(output, new_dims, DimsProd(new_dims), "linear_backward", conv_name, tensor);
  return new_tensor;
}



extern "C" char *Linear_Load(Scope_Struct *scope_struct, char *name) {
  return name;
}


extern "C" float Linear_weight(Scope_Struct *scope_struct, char *name) {
  std::string conv_name = scope_struct->first_arg;
  std::cout << "Linear name is " << name << ".\n";


  std::unique_ptr<LinearCPP> linear = std::move(NamedLinear[conv_name]);

  std::string _a = linear->Name+"W";
  char *a = _a.data();

  PrintTensor(scope_struct, a);

  NamedLinear[conv_name] = std::move(linear);
  
  return 0;
}



extern "C" float Linear_Create(Scope_Struct *scope_struct, char *name, char *scopeless_name, void *init_val, DT_list *notes_vector)
{

  // std::cout << "\n\n\n----------------------EXECUTION: CREATING LINEAR: " << name << ".\n\n\n\n";


  std::string init = "xavu";

  int C = notes_vector->get<int>(0);
  int OC = notes_vector->get<int>(1);


  std::vector<std::string> notes;
  

  for (int i=2; i<notes_vector->data->size(); i++)
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


  std::unique_ptr<LinearCPP> linear = std::make_unique<LinearCPP>(C, OC, init, notes, name);


  NamedLinear[name] = std::move(linear);

  std::cout << "***Created Linear: " << name << ".\n";

  return 0;
}
