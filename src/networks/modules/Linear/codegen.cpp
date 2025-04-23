
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




void linear_backward(float *x, float *dx, float *dy, std::string name)
{
  std::unique_ptr<LinearCPP> linear = std::move(NamedLinear[name]);

  linear->Backward(x, dx, dy);

  NamedLinear[name] = std::move(linear);
}





extern "C" void *LinearForward(char *self, Tensor *tensor, int thread_id, char *conv_namec, int is_obj_attr_or_self)
{
  // //TODO: remove self arg and concatenate it instead during the function call
  
  
  // std::string _self = self;
  // std::string conv_name = conv_namec;
  // if (is_obj_attr_or_self)
  //   conv_name = _self + conv_name;

  // //std::cout << "\nLinear of " << conv_name << " with input " << tensor->name  << "\n";
  // //std::cout << "thread id: " << thread_id << "\n\n";

  

  // float *output;
  
  // std::vector<float> dims = tensor->dims;
  
  // int C = dims[dims.size()-1];



  // std::unique_ptr<LinearCPP> linear = std::move(NamedLinear[conv_name]);

  // if ((int)C!=(int)linear->C)
  // {
  //   std::string error = "Input tensor  are: " + std::to_string((int)C) + ", while the expected input channels of the Linear are: " + std::to_string(linear->C);
  //   LogError(error);
    
  //   NamedLinear[conv_name] = std::move(linear);
  //   return nullptr;
  // }
    
  // std::vector<float> new_dims = RemoveLastDim(dims);
  // new_dims.push_back(linear->OC);
  



  // tensor->Sync();
  // output = linear->Forward(tensor, thread_id);


  // NamedLinear[conv_name] = std::move(linear);

  

  // Tensor *new_tensor = createTensor(output, new_dims, DimsProd(new_dims), false, "");
  // new_tensor->AttrLNode(tensor, linear_op);
  // new_tensor->scopeless_name = conv_name;
  // return new_tensor;
}




extern "C" void *Linear(Scope_Struct *scope_struct, Tensor *tensor)
{
  //TODO: remove self arg and concatenate it instead during the function call

  std::cout << "-------------------------------------CALLING LINEAR " << scope_struct->first_arg << ".\n";

  int thread_id = scope_struct->thread_id;
  

  scope_struct->Print();
  
  std::string conv_name = scope_struct->first_arg;

  std::cout << "\nLinear of " << conv_name << " with input " << tensor->name  << "\n";
  std::cout << "thread id: " << thread_id << "\n\n";

  

  float *output;
  
  std::vector<float> dims = tensor->dims;
  
  int C = dims[dims.size()-1];



  std::unique_ptr<LinearCPP> linear = std::move(NamedLinear[conv_name]);

  if ((int)C!=(int)linear->C)
  {
    std::string error = "Input tensor last dim is: " + std::to_string((int)C) + ", while the expected input dim of the Linear layer is: " + std::to_string(linear->C);
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





extern "C" float Linear_Create(char *name, char *scopeless_name, void *init_val, AnyVector *notes_vector, Scope_Struct *scope_struct)
{

  // std::cout << "\n\n\n----------------------EXECUTION: CREATING LINEAR: " << name << ".\n\n\n\n";


  std::string init = "xavu";

  float C = notes_vector->get<float>(0);
  float OC = notes_vector->get<float>(1);


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


  std::unique_ptr<LinearCPP> linear = std::make_unique<LinearCPP>((int) C, (int) OC, init, notes, name);


  NamedLinear[name] = std::move(linear);

  std::cout << "***Created Linear: " << name << ".\n";
  delete[] name;
  delete[] scopeless_name;

  return 0;
}

extern "C" float CreateLinearOnDemand(char *tensor_name, char *init,
                                      float C, float OC, int_vec *Notators)
{
  std::cout << "\nCreate linear on demand:\n  C " << C << " OC " << OC << "\n";
  std::cout << "" << tensor_name << " " << init << "\n";


  // std::unique_ptr<Linear> linear = std::make_unique<Linear>((int) C, int (OC), init, Notators, tensor_name);

  std::cout << "Adding " << tensor_name << " to NamedMHSA dict\n";
  // NamedLinear[tensor_name] = std::move(linear);
  return 0;
}