
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
#include "../../../notators/notators.h"
#include "../../../tensor/include.h"
#include "../globals.h"
#include "class.h"




extern "C" DT_tensor *EmbeddingLn(Scope_Struct *scope_struct, DT_tensor *tensor)
{
  //TODO: remove self arg and concatenate it instead during the function call
  
  int thread_id = scope_struct->thread_id;
  std::string conv_name = scope_struct->first_arg;



  float *tensor_ptr, *output;
  
  std::vector<int> dims = tensor->dims;
  int input_dims_prod = DimsProd(dims);


  std::unique_ptr<DT_EmbeddingLn> embedding = std::move(NamedEmbeddingLn[conv_name]);

  tensor->Sync();

  
  output = embedding->Forward(tensor, tensor->dims_prod, thread_id);
  

  int is_forward_func = 1;
  

  std::vector<int> new_dims = tensor->dims;
  // new_dims.pop_back();
  new_dims.push_back(embedding->OC);

  


  NamedEmbeddingLn[conv_name] = std::move(embedding);
  

  std::cout << "Out dims: " << ".\n";
  PrintDims(new_dims);
  std::exit(0);
  

  return customOpTensor(output, new_dims, DimsProd(new_dims), "embeddingln_backward", conv_name, tensor);
}









extern "C" float EmbeddingLn_Create(Scope_Struct *scope_struct, char *name, char *scopeless_name, void *init_val, DT_list *notes_vector)
{
  std::string init = "xavu";

  int V = notes_vector->get<int>(0);
  int C = notes_vector->get<int>(1);
  int OC = notes_vector->get<int>(2);

  std::vector<std::string> notes;
  
  for (int i=3; i<notes_vector->data->size(); i++)
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

  std::cout << "\nCreate embedding ln on demand:\n   V: " << V << " C: " << C << " OC " << OC << "\n";

  auto embedding = std::make_unique<DT_EmbeddingLn>(V, C, OC, init, name);

  std::cout << "Adding " << name << " to NamedEmbeddingLn dict\n";
  NamedEmbeddingLn[name] = std::move(embedding);
  return 0;
}

void embeddingln_backward(float *inp, int size, float *out,
                     float *dinp, float *dout,
                     std::string module_name, DT_tensor *node)
{
  std::unique_ptr<DT_EmbeddingLn> embedding = std::move(NamedEmbeddingLn[module_name]);

  embedding->Backward(inp, dout);

  NamedEmbeddingLn[module_name] = std::move(embedding);
}