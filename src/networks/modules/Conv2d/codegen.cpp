
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



void conv2d_backward(float *inp, int size, float *out,
                     float *dinp, float *dout,
                     void *network_module, DT_tensor *node)
{
  
  Conv2dCPP *conv = (Conv2dCPP*)network_module;
  // std::cout << "conv2d backward of " << conv->Name << ".\n";

  conv->Backward(inp, dinp, dout);
}







extern "C" DT_tensor *Conv2d(Scope_Struct *scope_struct, DT_tensor *tensor)
{

  // std::cout << "Conv2d pointer is: " << scope_struct->object_ptr << ".\n";

  

  std::string conv_name = scope_struct->first_arg;
  int thread_id = scope_struct->thread_id;

  // std::cout << "Conv forward of " << conv_name << " and tensor " << tensor->name << "\n";
  // std::cout << "Conv forward for conv: " << conv_name <<"\n";  



  float *output;
  std::vector<int> dims = tensor->dims;

  int B = dims[0];
  int C = dims[dims.size()-3];
  int H = dims[dims.size()-2];
  int W = dims[dims.size()-1];

  Conv2dCPP *conv = (Conv2dCPP*) scope_struct->object_ptr;


  if ((int)C!=(int)conv->C)
  {
    std::string error = "Input tensor channels are: " + std::to_string((int)C) + ", while the expected input channels of the convolution are: " + std::to_string(conv->C);
    LogError(error);
    
    return nullptr;
  }

  tensor->Sync();
  output = conv->Forward(tensor, H, W, B, thread_id);
 

  std::vector<int> new_dims = {conv->B, conv->OC, conv->out_H, conv->out_W};  
  

  return customOpTensor(output, new_dims, DimsProd(new_dims), "conv2d_backward", conv, tensor);
}








extern "C" void *Conv2d_Create(Scope_Struct *scope_struct, char *name, char *scopeless_name, void *init_val, DT_list *notes_vector)
{

  // std::cout << "\n\n\n----------------------EXECUTION: CREATING CONV2D: " << name << ".\n\n\n\n";

  std::string init = "xavu";

  int C = notes_vector->get<int>(0);
  int OC = notes_vector->get<int>(1);
  int ks = notes_vector->get<int>(2);
  int stride = notes_vector->get<int>(3);
  int padding = notes_vector->get<int>(4);

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


  Conv2dCPP *conv = new Conv2dCPP(C, OC, ks, stride, padding, init, notes, name);

  // std::unique_ptr<Conv2dCPP> conv2d = std::make_unique<Conv2dCPP>(C, OC, ks, stride, padding, init, notes, name);

  // NamedConv2d[name] = std::move(conv2d);


  return conv;
}