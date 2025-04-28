
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



#include "../../../backprop/backprop.h"
#include "../../../compiler_frontend/include.h"
#include "../../../data_types/codegen_notes.h"
#include "../../../mangler/scope_struct.h"
#include "../../../notators/notators.h"
#include "../../../tensor/include.h"
#include "../globals.h"
#include "class.h"



void conv2d_backward(float *inp, float size, float *out,
                     float *dinp, float *dout,
                     std::string conv_name)
{
  std::unique_ptr<Conv2dCPP> conv = std::move(NamedConv2d[conv_name]);
  conv->Backward(inp, dinp, dout);
  NamedConv2d[conv_name] = std::move(conv);  
}







extern "C" void *Conv2d(Scope_Struct *scope_struct, Tensor *tensor)
{
  //TODO: remove self arg and concatenate it instead during the function call
  
  

  std::string conv_name = scope_struct->first_arg;
  int thread_id = scope_struct->thread_id;


  // std::cout << "Conv forward of " << conv_name << " and tensor " << tensor->name << "\n";
  // std::cout << "Conv forward for conv: " << conv_name <<"\n";
  

  float *tensor_ptr, *output, *d_filter;
  tensor_ptr = tensor->tensor_ptr;
  std::vector<float> dims = tensor->dims;
  float input_dims_prod = DimsProd(dims);

  float B = dims[0];
  float C = dims[dims.size()-3];
  float H = dims[dims.size()-2];
  float W = dims[dims.size()-1];



  std::unique_ptr<Conv2dCPP> conv = std::move(NamedConv2d[conv_name]);



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




  // Tensor *conv_tensor = NamedTensorsT[conv_name];
  // conv_tensor->NewTensor(conv->d_filter, kernel_dims, DimsProd(kernel_dims), true, conv_name);
  // conv_tensor->SetIsWeight();
  

  NamedConv2d[conv_name] = std::move(conv);


  return customOpTensor(output, new_dims, DimsProd(new_dims), "conv2d_backward", conv_name, tensor);
}








extern "C" float Conv2d_Create(char *name, char *scopeless_name, void *init_val, AnyVector *notes_vector, Scope_Struct *scope_struct)
{

  std::cout << "\n\n\n----------------------EXECUTION: CREATING CONV2D: " << name << ".\n\n\n\n";


  std::string init = "xavu";

  float C = notes_vector->get<float>(0);
  float OC = notes_vector->get<float>(1);
  float ks = notes_vector->get<float>(2);
  float stride = notes_vector->get<float>(3);
  float padding = notes_vector->get<float>(4);


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


  std::unique_ptr<Conv2dCPP> conv2d = std::make_unique<Conv2dCPP>((int) C, (int) OC, (int) ks, (int) stride, (int) padding, init, notes, name);


  NamedConv2d[name] = std::move(conv2d);

  std::cout << "***Created Conv2d: " << name << ".\n";
  delete[] name;
  delete[] scopeless_name;

  return 0;
}