
#include<cstdarg>
#include<cstring>
#include<map>
#include<random>
#include<string>
#include<thread>
#include<vector>


#include "../backprop/include.h"
#include "../common/include.h"
#include "../compiler_frontend/global_vars.h"
#include "../compiler_frontend/logging.h"
#include "../mangler/scope_struct.h"
#include "../nsk_cuda/pool/include.h"
#include "../tensor/include.h"
#include "include.h"



extern "C" void *pinned_tensor_Create(Scope_Struct *scope_struct, char *tensor_name, char *scopeless_name, DT_tensor *init_val, DT_list *notes_vector)
{

  std::cout << "PINNED TENSOR CREATE"  << ".\n";

  DT_tensor *tensor;

  std::vector<int> dims;
  bool is_weight = false;
  for (int i=0; i<notes_vector->Size(); i++)
  {
    if(notes_vector->data_types->at(i)=="int")
      dims.push_back(notes_vector->get<int>(i));
    if(notes_vector->data_types->at(i)=="str")
      char *note = notes_vector->get<char *>(i);
  }



  int product = DimsProd(dims);
  float *tensor_ptr, *pool_tensor;
  float *tensor_cpu;


  cudaMallocHost(&tensor_cpu, round_to_nearest_pow2(product)*sizeof(float));

  for (int i = 0; i < product; ++i) {
    tensor_cpu[i] = 0.0f;
  }
  

  cudaMalloc(&tensor_ptr, round_to_nearest_pow2(product)*sizeof(float));  
  tensor = createPinned(tensor_ptr, tensor_cpu, dims, product, tensor_name);
  // NamedTensorsT[tensor_name] = tensor;
  

  
  // pinned tensors are 1 pool tensor behind.
  std::vector<int> pool_dims = dims;
  pool_dims.erase(pool_dims.begin());
  int pool_product = DimsProd(pool_dims);

  pool_tensor = get_from_pool(0, pool_product, "create pinned");
  move_to_pool(0, pool_product, pool_tensor, "create pinned");
  

  return tensor;
}



extern "C" DT_tensor *pinned_tensor_Load(Scope_Struct *scope_struct, char *tensor_name) {
  // std::cout << "\n\nPINNED LOAD TENSOR: " << tensor_name <<  "\n";

  DT_tensor *ret = NamedTensorsT[tensor_name];
  if(scope_struct->is_at_return && (nn_mode==eval_mode||scope_struct->thread_id!=0))
    ret->leaf = false; // Marks to clean

  return ret;
}





// extern "C" float float_vec_Store_Idx(char *name, float idx, float value, Scope_Struct *scope_struct){
extern "C" void pinned_tensor_Store_Idx(DT_tensor *tensor, int idx_at, float val, Scope_Struct *scope_struct) { 

  // std::cout << "pinned_tensor_Store_Idx on idx " << idx_at << ".\n";


  std::vector<int> dims = tensor->dims;
  int dims_prod = DimsProd(dims);
  if (idx_at>(dims_prod-1))
  {
    std::string _error = "\n\t- Idexating at pos: \033[32m"+std::to_string((int)idx_at);
    _error = _error + "\033[0m on pinned_tensor \033[95m"+std::string(tensor->name);
    _error = _error + "\033[0m;\n\t- Max idx allowed:  \033[32m"+std::to_string(dims_prod)+"\033[0m.";

    LogErrorS(scope_struct->code_line, _error);
    std::cout << "Dimensions:" << "\n";
    PrintDims(dims);
    std::cout << "\n";
  }

  float *base_address = tensor->cpu_tensor_ptr;
  
  
  //std::cout << "idx " << idx_at << ", val " << val << "\n";

  float *device_x = base_address + static_cast<int>(idx_at);

  *device_x = val;
}



extern "C" int pinned_tensor_CalculateIdx(DT_tensor *tensor, int first_idx, ...) {
  
  // std::cout << "pinned_tensor_CalculateIdx of " << tensor->name << "\n";


  std::vector<int> idxs, new_dims_no_minus, dims;
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

  idx_at += current_dims_prod*first_idx;

  
  // std::cout << "---idx: " << first_idx << "|cur_dims_prod: " << std::to_string(current_dims_prod) << "|adding: " << std::to_string(current_dims_prod*first_idx) << ".\n";



  for (int i=0; i<10; i++)
  {
    if (i==9)
    {
      LogErrorS(-1, "A tensor with 10 dimensions??? (calc idx)");
      return 0;
    }

    int idx = va_arg(args, int);
    if (idx==TERMINATE_VARARG)
      break;

    idxs.push_back(idx);
    
    dims = RemoveFirstDim(dims);
    
    current_dims_prod = DimsProd(dims);
    
    // std::cout << "---idx: " << idx << "|cur_dims_prod: " << std::to_string(current_dims_prod) << "|adding: " << std::to_string(current_dims_prod*idx) << ".\n";

    idx_at += current_dims_prod*idx;

    

    if (idx!=-1)
      new_dims_no_minus.push_back(idx);
    else
      has_minus=true;
  }
  va_end(args);

  // std::cout << "***returing idx: " << idx_at << ".\n";

  return idx_at;
}

