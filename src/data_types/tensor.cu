
#include<string>
#include<vector>
#include<map>
#include<cstring>
#include<random>
#include<thread>
#include<cstdarg>

#include "../backprop/include.h"
#include "../common/include.h"
#include "../compiler_frontend/logging.h"
#include "../cuda_kernels/calculate_grids.h"
#include "../cuda_kernels/elementwise_kernels_inline.cu"
#include "../mangler/scope_struct.h"
#include "../tensor/include.h"
#include "include.h"




extern "C" DT_tensor *tensor_Create(Scope_Struct *scope_struct, char *tensor_name, char *scopeless_name, DT_tensor *init_val, DT_list *notes_vector)
{
  
  // if (notes_vector->data->size()>0)
  // {

  
    int thread_id = scope_struct->thread_id;
    // std::cout << "CREATING TENSOR " << tensor_name << " AT THREAD: " << thread_id << "\n";

    DT_tensor *tensor;


    std::vector<float> dims;
    char *init = "xavu";
    bool is_weight = false;
    for (int i=0; i<notes_vector->data->size(); i++)
    {
      if(notes_vector->data_types->at(i)=="float")
        dims.push_back(notes_vector->get<float>(i));
      if(notes_vector->data_types->at(i)=="str")
      {
        std::cout << "get char" << ".\n";
        char *note = notes_vector->get<char *>(i);
        if (std::strcmp(note,"param") == 0)
          is_weight = true;
        else
          init = note; 
        std::cout << "got char" << ".\n";
      }
    }

    
    int product = DimsProd(dims);

    float *tensor_ptr;
    float *tensor_cpu;

    if (init_val==nullptr)
    {

      if(product>0)
      {
        if (std::strcmp(init, "randu") == 0)
          tensor_cpu = make_random_float_uniform(product);
        if (std::strcmp(init, "zeros") == 0)
          tensor_cpu = make_zeros_float(product);
        if (std::strcmp(init, "ones") == 0)
          tensor_cpu = make_ones_float(product);
        if (std::strcmp(init, "normal") == 0)
          tensor_cpu = make_normal(product);
        if (std::strcmp(init, "xavu") == 0)
          tensor_cpu = make_xavier_uniform_float(product, dims[dims.size()-1], dims[dims.size()-2]);
        if (std::strcmp(init, "xavu_relu") == 0)
          tensor_cpu = make_xavier_uniform_float_relu(product, dims[dims.size()-1], dims[dims.size()-2]);
        if (std::strcmp(init, "xavu_tanh") == 0)
          tensor_cpu = make_xavier_uniform_float_tanh(product, dims[dims.size()-1], dims[dims.size()-2]);
        if (std::strcmp(init, "he_normal_relu") == 0)
          tensor_cpu = make_he_normal_float_relu(product, dims[dims.size()-1]);
        if (std::strcmp(init, "init_gpt") == 0)
          tensor_cpu = make_gpt_init(product);
        if (std::strcmp(init, "int") == 0)
          tensor_cpu = make_random_int(product, 10);
        if (std::strcmp(init, "arange") == 0)
          tensor_cpu = make_arange(product);
        if (std::strcmp(init, "binary") == 0)
          tensor_cpu = make_random_int(product, 1);

        cudaCheck(cudaGetLastError());
        std::string _name = "create tensor ";
        _name = _name + tensor_name;
        tensor_ptr = get_from_pool(thread_id, product, _name);
        //std::cout << "cpy of: " << tensor_name << "\n";

        cudaStream_t stream = ThreadsStream[thread_id];
        cudaCheck(cudaMemcpyAsync(tensor_ptr, tensor_cpu, product*sizeof(float), cudaMemcpyHostToDevice, stream));
        //cudaStreamSynchronize(stream);
        delete[] tensor_cpu;
      }
    } else {
      int grid_size, block_size;
      CalculateGridAndBlockSizes(product, grid_size, block_size);
      cudaStream_t stream = ThreadsStream[thread_id];
      copy_tensor_kernel<<<grid_size, block_size, 0, stream>>>(tensor_ptr, init_val->tensor_ptr, product);
    }
    
    

    tensor = createTensor(tensor_ptr, dims, product, true, tensor_name);
    tensor->scopeless_name = scopeless_name;
    if(is_weight)
      tensor->SetIsWeight();
    tensor->op = create_tensor_op;

   

    // }

  if(NamedTensorsT.count(tensor_name)>0)
  {
    DT_tensor *tensor_to_clean = NamedTensorsT[tensor_name];

    // if (tensor_to_clean->name=="batch_acc")
    // if (tensor_to_clean->name=="batch_acc"||tensor_to_clean->name=="y")
    // {
      // std::cout << "0000000000000000000000000000000000000CLEANING " << tensor_name << ".\n";
      move_to_pool(thread_id, tensor_to_clean->dims_prod, tensor_to_clean->tensor_ptr, "tensor_Create tensor substitution of " + tensor_to_clean->name + ".");
    // }
    // delete tensor_to_clean;
  }
    
  NamedTensorsT[tensor_name] = tensor;
  


  return tensor;
}








extern "C" DT_tensor *tensor_Load(Scope_Struct *scope_struct, char *tensor_name) {

  
  // std::cout << "\n\nLOAD TENSOR: " << tensor_name <<  "\n";
  DT_tensor *ret = NamedTensorsT[tensor_name];

  if(scope_struct->is_at_return && (nn_mode==eval_mode||scope_struct->thread_id!=0))
    ret->leaf = false;
  //std::cout << "return load." << "\n";

  return ret;
}


//todo: copy tensor
extern "C" DT_tensor *tensor_Copy(Scope_Struct *scope_struct, DT_tensor *tensor){

  int thread_id = scope_struct->thread_id;

  std::string tensor_name = tensor->name;
  
  std::string arg_tensor_name = "list_" + tensor_name;
  

  std::vector<float> dims = tensor->dims;
  int dims_prod = tensor->dims_prod;

  float *arg_tensor, *tensor_ptr;

  tensor_ptr = tensor->tensor_ptr;

  std::string where_from = "arg tensor of ";
  where_from = where_from + tensor_name;
  arg_tensor = get_from_pool(thread_id, dims_prod, where_from);
  
  
  if (dims_prod!=0)
  {
    int grid_size, block_size, shared_mem_size; 
    std::vector<int> grid_block_mem_sizes = CalculateGridAndBlockSizes(tensor->dims_prod);
    grid_size = grid_block_mem_sizes[0];
    block_size = grid_block_mem_sizes[1];

    tensor->Sync();

    cudaStream_t stream = ThreadsStream[thread_id];
    copy_tensor_kernel<<<grid_size,block_size,0,stream>>>(arg_tensor, tensor_ptr, dims_prod);
  }
  

  DT_tensor *new_tensor = createTensor(arg_tensor, dims, dims_prod, true, arg_tensor_name, tensor->cuda_stream, tensor->loader);
  new_tensor->scopeless_name = tensor->scopeless_name;
  new_tensor->from_grad_or_load = tensor->from_grad_or_load;

  
  if(nn_mode==eval_mode)//
    to_free_tensor_forward(tensor, scope_struct->scope);//
  else
    to_free_tensor(tensor);
  // std::cout << "DT_tensor copied" << ".\n";

  return new_tensor;
}





inline DT_tensor *store_intermediate_result_tensor(DT_tensor *stored_tensor, DT_tensor *tensor, char *tensor_name, int thread_id, int has_grad, char *scope) {
  // RHS does not need to be saved. So we just move the pointer to LHS
  // if(nn_mode==eval_mode||thread_id!=0)
  // {
  //   if(tensor->from_grad_or_load) //if(DoesTreeContainWeight(tensor)>0)
  //     ForwardCleanupToPool(tensor, scope);
  //   ForwardCleanupToPool(stored_tensor, scope);
  // }
  // else {
  DT_tensor *attr_tensor;
  if (has_grad==0)
      tensor->op = detach_op;
  attr_tensor = createBackward(stored_tensor->scopeless_name, tensor);
  todo_backward_tensors.push_back(attr_tensor);
  // } 

  std::string scopeless_name = stored_tensor->scopeless_name;
  stored_tensor = createTensor(tensor->tensor_ptr, tensor->dims, tensor->dims_prod, true, tensor_name, tensor->cuda_stream, tensor->loader);
  stored_tensor->from_grad_or_load = tensor->from_grad_or_load;
  stored_tensor->scopeless_name = scopeless_name;

  return stored_tensor;
}


inline void clean_tensor(DT_tensor *stored_tensor, DT_tensor *tensor, char *tensor_name, int thread_id, int has_grad, char *scope) {
  if (nn_mode==eval_mode||stored_tensor->thread_id!=0)
    CleanTreeNow(stored_tensor->thread_id, stored_tensor, stored_tensor->name);
  // Else, save the tensor for the backrpop.
}

inline DT_tensor *change_tensor_dims(DT_tensor *stored_tensor, DT_tensor *tensor, char *tensor_name, int thread_id, int has_grad, char *scope) {
  stored_tensor->tensor_ptr = get_from_pool(thread_id, tensor->dims_prod, "z=x");
  stored_tensor->dims = tensor->dims;
  stored_tensor->dims_prod = tensor->dims_prod;
  
  return stored_tensor;
}



inline DT_tensor *sync_and_copy_tensors(DT_tensor *stored_tensor, DT_tensor *tensor, char *tensor_name, int thread_id, int has_grad, char *scope) {
  int grid_size, block_size, shared_mem_size; 
  std::vector<int> grid_block_mem_sizes = CalculateGridAndBlockSizes(tensor->dims_prod);
  grid_size = grid_block_mem_sizes[0];
  block_size = grid_block_mem_sizes[1];

  stored_tensor->Sync();
  tensor->Sync();
  
  cudaStream_t stream = ThreadsStream[thread_id];
  copy_tensor_kernel<<<grid_size,block_size,0,stream>>>(stored_tensor->tensor_ptr, tensor->tensor_ptr, tensor->dims_prod);
  return stored_tensor;
}

inline DT_tensor *store_leaf_backward(DT_tensor *stored_tensor, DT_tensor *tensor, char *tensor_name, int thread_id, int has_grad, char *scope) {
  if(nn_mode==training_mode&&thread_id==0)
  {
    DT_tensor *attribution_tensor;
  
    if (has_grad==0)
      tensor->op = detach_op;  
    attribution_tensor = createBackward(stored_tensor->scopeless_name, tensor);
    todo_backward_tensors.push_back(attribution_tensor);

    std::string scopeless_name = stored_tensor->scopeless_name;
    stored_tensor = createTensor(stored_tensor->tensor_ptr, tensor->dims, tensor->dims_prod, true, tensor_name, stored_tensor->cuda_stream, stored_tensor->loader);
    
    stored_tensor->from_grad_or_load = tensor->from_grad_or_load;
    stored_tensor->scopeless_name = scopeless_name;
  }

  return stored_tensor;
}

extern "C" float tensor_Store(char *tensor_name, DT_tensor *tensor, Scope_Struct *scope_struct)
{
  // std::cout << "tensor_Store execution" << ".\n";


  char *scope = scope_struct->scope;
  int thread_id= scope_struct->thread_id;
  int has_grad = scope_struct->has_grad;


  DT_tensor *stored_tensor = NamedTensorsT[tensor_name];
  stored_tensor->is_last_version = false;
  
  // View op
  if (tensor->view_of == tensor_name)
  {
    stored_tensor->dims = tensor->dims;
    delete tensor;
  }
  // Non-leaf RHS.
  // Free current and point to operation result
  else if (tensor->name==""||!tensor->leaf) 
  {
    clean_tensor(stored_tensor, tensor, tensor_name, thread_id, has_grad, scope); // Remove current if evaluating
    stored_tensor = store_intermediate_result_tensor(stored_tensor, tensor, tensor_name, thread_id, has_grad, scope);
  }
  else {
   
    // Is Leaf
    if(tensor->op==tensor_leaf||tensor->op==create_tensor_op||nn_mode==eval_mode||thread_id!=0)
    {
      clean_tensor(stored_tensor, tensor, tensor_name, thread_id, has_grad, scope); // Remove current if evaluating
      stored_tensor = change_tensor_dims(stored_tensor, tensor, tensor_name, thread_id, has_grad, scope);
      stored_tensor = sync_and_copy_tensors(stored_tensor, tensor, tensor_name, thread_id, has_grad, scope);
      stored_tensor = store_leaf_backward(stored_tensor, tensor, tensor_name, thread_id, has_grad, scope);       
    } 
  }

  stored_tensor->thread_id = thread_id;
  stored_tensor->is_last_version = true;
  NamedTensorsT[tensor_name] = stored_tensor;
  cudaCheck(cudaGetLastError());
  return 0;
}



void tensor_Clean_Up(void *data_ptr) {
  // std::cout << "tensor_Clean_Up" << ".\n";
}

extern "C" float tensor_opa(Scope_Struct *scope_struct, void *data_ptr) {
  std::cout << "opa tensor"  << ".\n";
  PrintDims(static_cast<DT_tensor *>(data_ptr)->dims);
  return 25;
}



extern "C" DT_tensor *gpu(Scope_Struct *scope_struct, DT_tensor *tensor, DT_tensor *pinned_tensor)
{
  //std::cout << "\nGpu transfer for: " << tensor.name << " on worker " << idx << "\n";
  int thread_id = scope_struct->thread_id; 
  float *tensor_ptr, *tensor_cpu;

  
  tensor_cpu = pinned_tensor->cpu_tensor_ptr;
  std::vector<float> dims = pinned_tensor->dims;
  float dims_prod = pinned_tensor->dims_prod;
  



  
  if (tensor->dims_prod==dims_prod)
    tensor_ptr = tensor->tensor_ptr;
  else
    tensor_ptr = get_from_pool(thread_id, dims_prod, "gpu");
  
  //tensor_ptr = get_from_pool(dims_prod, "gpu");


  
  Loader *loader=nullptr;
  cudaStream_t cuda_stream = createCudaStream();
  cudaMemcpyAsync(tensor_ptr, tensor_cpu, dims_prod * sizeof(float), cudaMemcpyHostToDevice, cuda_stream);
  //cudaMemcpy(tensor_ptr, tensor_cpu, dims_prod * sizeof(float), cudaMemcpyHostToDevice);
  pinned_tensor->cuda_stream = cuda_stream;
  



  if (nn_mode==eval_mode)
  {

  } else {
    
    DT_tensor *attr_tensor;
    attr_tensor = createTensor(tensor_ptr, dims, dims_prod, true, "");
    attr_tensor->op = gpu_op;
    todo_backward_tensors.push_back(attr_tensor); // pass to gc
    
  }

  tensor->AttrTensor(tensor_ptr, dims, dims_prod, cuda_stream, loader);
  tensor->from_grad_or_load = true;

  return 0;
}



extern "C" float tensor_gpuw(Scope_Struct *scope_struct, DT_tensor *tensor, DT_tensor *pinned_tensor, float idx)
{
  int thread_id = scope_struct->thread_id;

  // std::cout << "\nGpu transfer for: " << tensor->name << " on worker " << idx << " and thread id: " << thread_id << "\n";

  float *tensor_ptr, *tensor_cpu;

  
  
  std::vector<float> dims, batchless_dims;
  dims = pinned_tensor->dims;
  

  batchless_dims = BatchLessDims(dims);
  float batchless_dims_prod = (float)DimsProd(batchless_dims);


  tensor_cpu = pinned_tensor->cpu_tensor_ptr + static_cast<int>(idx*batchless_dims_prod);

  
  if (tensor->dims_prod==batchless_dims_prod)
    tensor_ptr = tensor->tensor_ptr;
  else
    tensor_ptr = get_from_pool(thread_id, batchless_dims_prod, "gpuw");
  
  //tensor_ptr = get_from_pool(batchless_dims_prod, "gpuw");


  
  Loader *loader=nullptr;
  cudaStream_t cuda_stream = nullptr;
  
  if (batchless_dims_prod<2000){
    cudaMemcpy(tensor_ptr, tensor_cpu, batchless_dims_prod * sizeof(float), cudaMemcpyHostToDevice);
  }
  else// if (batchless_dims_prod<1000)
  {
    cuda_stream = createCudaStream();
    cudaMemcpyAsync(tensor_ptr, tensor_cpu, batchless_dims_prod * sizeof(float), cudaMemcpyHostToDevice, cuda_stream);
    //cudaMemcpy(tensor_ptr, tensor_cpu, batchless_dims_prod * sizeof(float), cudaMemcpyHostToDevice);
    pinned_tensor->cuda_stream = cuda_stream;
  }
  /*
  else
  {
    //cuda_stream = AllocateStream(0);
    //cudaMemcpyAsync(tensor_ptr, tensor_cpu, batchless_dims_prod * sizeof(float), cudaMemcpyHostToDevice, cuda_stream);
    loader = new Loader();
    loader->Load(tensor_ptr, tensor_cpu, batchless_dims_prod);
  }
  */



  if (nn_mode==eval_mode||thread_id!=0)
  {

  } else {
    
    DT_tensor *attr_tensor;
    attr_tensor = createTensor(tensor_ptr, batchless_dims, batchless_dims_prod, true, "");
    attr_tensor->op = gpu_op;
    todo_backward_tensors.push_back(attr_tensor); // pass to gc
    
  }

  tensor->AttrTensor(tensor_ptr, batchless_dims, batchless_dims_prod, cuda_stream, loader);
  tensor->from_grad_or_load = true;
  tensor->leaf=true;


  return 0;
}


extern "C" float cpu(Scope_Struct *scope_struct, DT_tensor *tensor)
{

  int thread_id = scope_struct->thread_id; 

  float *tensor_ptr, *tensor_cpu;
  tensor_ptr = tensor->tensor_ptr;
  tensor_cpu = tensor->cpu_tensor_ptr;

  cudaStream_t stream = ThreadsStream[thread_id];
  cudaStreamSynchronize(stream);

  if (tensor_ptr==nullptr)
    LogErrorS("Cannot load tensor to cpu from an null tensor.");

  if (tensor_cpu!=nullptr)
    cudaCheck(cudaFree(tensor_cpu));

  float dims_prod = tensor->dims_prod;



  cudaMallocHost(&tensor_cpu, dims_prod*sizeof(float));
  cudaMemcpy(tensor_cpu, tensor_ptr, dims_prod*sizeof(float), cudaMemcpyDeviceToHost);

  tensor->cpu_tensor_ptr = tensor_cpu;


  return 0;
}

extern "C" float cpu_idx(Scope_Struct *scope_struct, DT_tensor *tensor, float idx)
{

  float *tensor_cpu;
  tensor_cpu = tensor->cpu_tensor_ptr;


  if (tensor_cpu==nullptr)
    LogErrorS("Cannot idx a null cpu tensor.");

  float dims_prod = tensor->dims_prod;
  if (idx>dims_prod)
    LogErrorS("Idx higher than dims prod at cpu_idx().");

  

  return tensor_cpu[(int)idx];
}


extern "C" DT_tensor *randu_like(Scope_Struct *scope_struct, DT_tensor tensor)
{
  int thread_id = scope_struct->thread_id;

  float dims_prod = tensor.dims_prod;

  float *tensor_ptr, *tensor_cpu;

  tensor_cpu = make_random_float_uniform(dims_prod);

  cudaStream_t stream = ThreadsStream[thread_id];
  cudaMalloc(&tensor_ptr, dims_prod*sizeof(float));
  cudaMemcpyAsync(tensor_ptr, tensor_cpu, dims_prod*sizeof(float), cudaMemcpyHostToDevice, stream);
  delete[] tensor_cpu;

  DT_tensor *new_tensor = createTensor(tensor_ptr, tensor.dims, dims_prod, false, "");
  new_tensor->op = randu_like_op;
  return new_tensor;
}



void copyChunk(float* d_data, const float* h_data, int offset, float size, cudaStream_t stream) {
  cudaMemcpyAsync(d_data + offset, h_data + offset, size*sizeof(float), cudaMemcpyHostToDevice, stream);
}


extern "C" float write_zerosw(DT_tensor *tensor, float worker_idx)
{
  std::vector<float> dims = tensor->dims;

  std::vector<float> workerless_dims = BatchLessDims(dims);
  int workerless_dims_prod = DimsProd(workerless_dims);

  int idx_offset = (int) (workerless_dims_prod*worker_idx);

  for(int i=0; i<workerless_dims_prod; i++)
    tensor->cpu_tensor_ptr[i+idx_offset] = 0.0f;
  
  return 0;
}


extern "C" DT_tensor *tensor_view(Scope_Struct *scope_struct, DT_tensor *tensor, float first_dim, ...)
{

  //std::cout << "Executing: " << tensor.name << "." << "view" << "\n";
   
  std::vector<float> new_dims, new_dims_no_minus, current_dims;
  bool has_minus = false;
  current_dims = tensor->dims;

  
  va_list args;
  va_start(args, first_dim);

  if (first_dim!=-1)
    new_dims_no_minus.push_back(first_dim);
  else
    has_minus=true;
  
  
  new_dims.push_back(first_dim);

  for (int i=0; i<10; i++)
  {
    if (i==9)
    {
      LogErrorS("A tensor with 10 dimensions??? (view)");
      return 0;
    }

    float dim = va_arg(args, float);
    if (dim==TERMINATE_VARARG)
      break;
    new_dims.push_back(dim);

    if (dim!=-1)
      new_dims_no_minus.push_back(dim);
    else
      has_minus=true;
  }
  va_end(args);


  


  int current_dims_prod = DimsProd(current_dims);
  int new_dims_prod = DimsProd(new_dims);


  if (has_minus)
  {
    float hidden_dim = (float)current_dims_prod / (float)DimsProd(new_dims_no_minus);

    if ((float)((int)hidden_dim) != hidden_dim)
    {
      LogErrorS("Automatic view dimension calculus resulted on a non-integer dimension.");
      PrintDims(current_dims);
      std::cout << "Current dims product: " << current_dims_prod  << ".\n";
      PrintDims(new_dims);
      std::cout << "New dims product: " << std::to_string(DimsProd(new_dims_no_minus))  << ".\n";
      return 0;
    }
    
    for (int i=0; i<new_dims.size(); i++)
      if (new_dims[i]==-1)
        new_dims[i] = hidden_dim;
    
  } else {
    if (current_dims_prod != new_dims_prod)
    {
      LogErrorS("Incompatible view dimensions.");
      PrintDims(current_dims);
      std::cout << "Current dims product: " << current_dims_prod  << ".\n";
      PrintDims(new_dims);
      std::cout << "New dims product: " << new_dims_prod  << ".\n";
      return 0;
    }
  }

  

  DT_tensor *new_tensor = createTensor(tensor->tensor_ptr, new_dims, DimsProd(new_dims), false, "");
  new_tensor->view_of = tensor->name;
  new_tensor->op=view_op;
  return new_tensor;
}



extern "C" DT_tensor *NewVecToTensor(Scope_Struct *scope_struct, float first_dim, ...)
{
  std::vector<float> values;

  int thread_id = scope_struct->thread_id;
  
  va_list args;
  va_start(args, first_dim);


  values.push_back(first_dim);

  for (int i=0; i<10; i++)
  {
    if (i==9)
    {
      LogErrorS("Tried to create a tensor from brackets with more than 10 positions. This is not yet supported");
      return nullptr;
    }

    float dim = va_arg(args, float);
    if (dim==TERMINATE_VARARG)
      break;
    values.push_back(dim);


  }
  va_end(args);


  float dims_prod = values.size();

  float *tensor_ptr, *tensor_cpu;
  tensor_cpu = values.data();

  tensor_ptr = get_from_pool(thread_id, dims_prod, "tensor from brackets");
  cudaMemcpy(tensor_ptr, tensor_cpu, dims_prod*sizeof(float), cudaMemcpyHostToDevice);
  

  DT_tensor *new_tensor = createTensor(tensor_ptr, {dims_prod}, dims_prod, true, "");
  new_tensor->op=create_tensor_from_brackets_op;
  return new_tensor;
}




extern "C" float tensor_CalculateIdx(char *tensor_name, float first_idx, ...) {
  
  // std::cout << "pinned_tensor_CalculateIdx of " << tensor_name << "\n";

  DT_tensor *tensor = NamedTensorsT[tensor_name];

  std::vector<float> idxs, new_dims_no_minus, dims;
  int current_dims_prod;
  bool has_minus = false;
  dims = tensor->dims;

  int idx_at = 0;

  
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



  //std::cout << "Get idx of " << tensor_name << "\nCalculateIdxOffset pushing dim: " << first_idx << "\n";

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

    idx_at += (int)(current_dims_prod*idx);

    //std::cout << "CalculateIdxOffset pushing dim: " << idx << "\n";
    

    if (idx!=-1)
      new_dims_no_minus.push_back(idx);
    else
      has_minus=true;
  }
  va_end(args);



  return idx_at;
}




extern "C" DT_tensor *zeros_like(Scope_Struct *scope_struct, DT_tensor *tensor) {

  // DT_tensor *zeros_tensor = new DT_tensor();

  int thread_id = scope_struct->thread_id;
  float *tensor_ptr = tensor->tensor_ptr;
  std::vector<float> dims = tensor->dims;
  float dims_prod = tensor->dims_prod;

  int grid_size, block_size; 
  CalculateGridAndBlockSizes(dims_prod, grid_size, block_size);
  

  float *y = get_from_pool(thread_id, dims_prod, "relu");


  tensor->Sync();
  cudaStream_t stream = ThreadsStream[thread_id];
  set_to_zero_kernel<<<grid_size, block_size, 0, stream>>>(y, dims_prod);

  return customOpTensor(y, dims, DimsProd(dims), "set_to_zero", "", tensor);
}