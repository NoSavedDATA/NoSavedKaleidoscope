
#include<cstdarg>
#include<cstring>
#include<map>
#include<random>
#include<string>
#include<thread>
#include<vector>


#include "../backprop/include.h"
#include "../common/include.h"
#include "../compiler_frontend/logging.h"
#include "../mangler/scope_struct.h"
#include "../tensor/include.h"
#include "include.h"



extern "C" float pinned_tensor_Create(char *tensor_name, char *scopeless_name, float init_val, AnyVector *notes_vector, Scope_Struct *scope_struct)
{

  // std::cout << "PINNED TENSOR CREATE"  << ".\n";

  Tensor *tensor;

  std::vector<float> dims;
  bool is_weight = false;
  for (int i=0; i<notes_vector->data->size(); i++)
  {
    if(notes_vector->data_types->at(i)=="float")
      dims.push_back(notes_vector->get<float>(i));
    if(notes_vector->data_types->at(i)=="string")
      char *note = notes_vector->get<char *>(i);
  }


  int product = DimsProd(dims);
  float *tensor_ptr, *pool_tensor;
  float *tensor_cpu;


  cudaMallocHost(&tensor_cpu, product*sizeof(float));
  //tensor_cpu = new float[product];

  for (int i = 0; i < product; ++i) {
    tensor_cpu[i] = 0.0f;
  }
  

  cudaMalloc(&tensor_ptr, product*sizeof(float));  
  tensor = createPinned(tensor_ptr, tensor_cpu, dims, product, tensor_name);
  NamedTensorsT[tensor_name] = tensor;
  

  
  // pinned tensors are 1 pool tensor behind.
  std::vector<float> pool_dims = dims;
  pool_dims.erase(pool_dims.begin());
  float pool_product = DimsProd(pool_dims);
  pool_tensor = get_from_pool(0, pool_product, "create pinned");
  move_to_pool(0, pool_product, pool_tensor, "create pinned");
  

  return 0;
}





// extern "C" float float_vec_Store_Idx(char *name, float idx, float value, Scope_Struct *scope_struct){
extern "C" void pinned_tensor_Store_Idx(char *tensor_name, float idx_at, float val, Scope_Struct *scope_struct) { 

  // std::cout << "pinned_tensor_Store_Idx on idx " << idx_at << ".\n";

  Tensor *tensor = NamedTensorsT[tensor_name];
  // PrintDims(tensor->dims);

  std::vector<float> dims = tensor->dims;
  int dims_prod = DimsProd(dims);
  if (idx_at>(dims_prod-1))
  {
    std::string _error = "\n\t- Idexating at pos: \033[32m"+std::to_string((int)idx_at);
    _error = _error + "\033[0m on pinned_tensor \033[95m"+std::string(tensor_name);
    _error = _error + "\033[0m;\n\t- Max idx allowed:  \033[32m"+std::to_string(dims_prod)+"\033[0m.";

    LogErrorS(_error);
    std::cout << "Dimensions:" << "\n";
    PrintDims(dims);
    std::cout << "\n";
  }

  float *base_address = tensor->cpu_tensor_ptr;
  
  
  //std::cout << "idx " << idx_at << ", val " << val << "\n";

  float *device_x = base_address + static_cast<int>(idx_at);

  *device_x = val;
  move_to_char_pool(strlen(tensor_name)+1, tensor_name, "free");
}





extern "C" float pinned_tensor_CalculateIdx(char *tensor_name, float first_idx, ...) {
  
  // std::cout << "pinned_tensor_CalculateIdx of " << tensor_name << "\n";

  Tensor *tensor = NamedTensorsT[tensor_name];

  std::vector<float> idxs, new_dims_no_minus, dims;
  int current_dims_prod;
  bool has_minus = false;
  dims = tensor->dims;

  int idx_at = 0;

  // PrintDims(dims);

  va_list args;
  va_start(args, first_idx);

  if (first_idx!=-1)
    new_dims_no_minus.push_back(first_idx);
  else
    has_minus=true;
  
    
  idxs.push_back(first_idx);

  dims = RemoveFirstDim(dims);
  
  current_dims_prod = DimsProd(dims);

  idx_at += (int)(current_dims_prod*first_idx);

  
  // std::cout << "---idx: " << first_idx << "|cur_dims_prod: " << std::to_string(current_dims_prod) << "|adding: " << std::to_string(current_dims_prod*first_idx) << ".\n";



  for (int i=0; i<10; i++)
  {
    if (i==9)
    {
      LogErrorS("A tensor with 10 dimensions??? (calc idx)");
      return 0;
    }

    float idx = va_arg(args, float);
    if (idx==TERMINATE_VARARG)
      break;

    idxs.push_back(idx);
    
    dims = RemoveFirstDim(dims);
    
    current_dims_prod = DimsProd(dims);
    
    // std::cout << "---idx: " << idx << "|cur_dims_prod: " << std::to_string(current_dims_prod) << "|adding: " << std::to_string(current_dims_prod*idx) << ".\n";

    idx_at += (int)(current_dims_prod*idx);

    

    if (idx!=-1)
      new_dims_no_minus.push_back(idx);
    else
      has_minus=true;
  }
  va_end(args);

  // std::cout << "***returing idx: " << idx_at << ".\n";

  return idx_at;
}