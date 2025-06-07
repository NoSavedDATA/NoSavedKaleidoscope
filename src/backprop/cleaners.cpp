
#include "../codegen/scope.h"
#include "../common/include.h"
#include "../mangler/scope_struct.h"
#include "../nsk_cuda/pool/include.h"
#include "../nsk_cuda/utils/include.h"
#include "cleaners.h"



// Cleaners
std::map<std::string, float *> var_to_grad;
std::vector<std::tuple<int, float *, std::string>> backprop_tensors_to_pool;
std::vector<float *> tensors_sent_to_pool;
std::vector<DT_tensor *> backprop_Tensors_to_free;
std::vector<DT_tensor *> backprop_Tensors_to_save;

std::map<std::string, std::vector<std::tuple<int, float*,std::string>>> forward_tensors_to_pool;
std::map<std::string, std::vector<float*>> forward_tensors_sent_to_pool;
std::map<std::string, std::vector<DT_tensor*>> forward_Tensors_to_free;
std::map<std::string, std::map<std::string, float*>> scope_tensors; // records last version of a tensor //todo: is this one actually used?

std::map<int, std::map<std::string, std::vector<std::tuple<int, float*,std::string>>>> threaded_tensors_to_pool;
std::map<int, std::map<std::string, std::vector<float*>>> threaded_tensors_sent_to_pool;
std::map<int, std::map<std::string, std::vector<DT_tensor*>>> threaded_Tensors_to_free;
std::map<int, std::map<std::string, std::vector<float*>>> threaded_tensors_to_save;
std::map<int, std::map<std::string, std::vector<DT_tensor*>>> threaded_Tensors_to_save;

void to_free_tensor(DT_tensor *tensor_ptr)
{
  if(!in_tensor_ptr_vec(tensor_ptr, backprop_Tensors_to_free))
    backprop_Tensors_to_free.push_back(tensor_ptr);
}
void to_pool(int dims_prod, float *tensor_ptr, std::string from)
{
  if (!in_float_ptr_vec(tensor_ptr, tensors_sent_to_pool))
  {
    backprop_tensors_to_pool.push_back(std::make_tuple(dims_prod, tensor_ptr, from));
    tensors_sent_to_pool.push_back(tensor_ptr);
  }
}
void save_from_pool(DT_tensor *tensor_ptr)
{
  if(!in_tensor_ptr_vec(tensor_ptr, backprop_Tensors_to_save))
    backprop_Tensors_to_save.push_back(tensor_ptr);
}



void to_free_tensor_forward(DT_tensor *tensor_ptr, std::string scope)
{
  if(!in_tensor_ptr_vec(tensor_ptr, forward_Tensors_to_free[scope]))
    forward_Tensors_to_free[scope].push_back(tensor_ptr);
}
void to_pool_forward(int dims_prod, float *tensor_ptr, std::string scope, std::string from)
{
  if (!in_float_ptr_vec(tensor_ptr, forward_tensors_sent_to_pool[scope]))
  {
    forward_tensors_to_pool[scope].push_back(std::make_tuple(dims_prod, tensor_ptr, from));
    forward_tensors_sent_to_pool[scope].push_back(tensor_ptr);
  }
}


void to_free_tensor_threaded(DT_tensor *tensor_ptr, std::string scope, int thread_id)
{
  if(!in_tensor_ptr_vec(tensor_ptr, threaded_Tensors_to_free[thread_id][scope]) && !in_tensor_ptr_vec(tensor_ptr, threaded_Tensors_to_save[thread_id][scope]))
    threaded_Tensors_to_free[thread_id][scope].push_back(tensor_ptr);
}
void to_pool_threaded(int dims_prod, float *tensor_ptr, std::string scope, int thread_id, std::string from)
{
  if (!in_float_ptr_vec(tensor_ptr, threaded_tensors_sent_to_pool[thread_id][scope]) && !in_float_ptr_vec(tensor_ptr, threaded_tensors_to_save[thread_id][scope]))
  {
    threaded_tensors_to_pool[thread_id][scope].push_back(std::make_tuple(dims_prod, tensor_ptr, from));
    threaded_tensors_sent_to_pool[thread_id][scope].push_back(tensor_ptr);
  }
}



void ThreadedCleanupToPool(DT_tensor *back_node, std::string scope, int thread_id)
{
  if(back_node==nullptr||back_node->weight)
    return;
  //std::cout << "-----Clean threaeded " << back_node->name << "\n";
  

  
  if (!in_str(scope, scopes));
    scopes.push_back(scope);

  ThreadedCleanupToPool(back_node->L_Node, scope, thread_id);
  ThreadedCleanupToPool(back_node->R_Node, scope, thread_id);

  
  to_pool_threaded(back_node->dims_prod, back_node->tensor_ptr, scope, thread_id, "");
  to_free_tensor_threaded(back_node, scope, thread_id);
}

void CleanThreadTensors(std::string scope, int thread_id)
{
  for(DT_tensor *tensor : threaded_Tensors_to_free[thread_id][scope])
    delete tensor;

  std::vector<float*> scope_tensors_ptrs;
  

  for(std::tuple<int, float *, std::string> pair : threaded_tensors_to_pool[thread_id][scope])
    move_to_pool(thread_id, std::get<0>(pair), std::get<1>(pair), std::get<2>(pair));


  threaded_Tensors_to_free[thread_id][scope].clear();
  threaded_tensors_to_pool[thread_id][scope].clear();
  threaded_tensors_sent_to_pool[thread_id][scope].clear();

  threaded_Tensors_to_free[thread_id].erase(scope);
  threaded_tensors_to_pool[thread_id].erase(scope);
  threaded_tensors_sent_to_pool[thread_id].erase(scope);
}



int DoesTreeContainWeight(DT_tensor *back_node)
{
  if(back_node==nullptr)
    return 0;
  
  if(back_node->weight)
    return 1;

  //if(in_int(back_node->op, activation_ops))
  //  return 1;
  
  return DoesTreeContainWeight(back_node->L_Node) + DoesTreeContainWeight(back_node->R_Node);
}


void ForwardCleanupToPool(DT_tensor *back_node, std::string scope)
{
  if(back_node==nullptr||back_node->weight)
    return;
  

  
  if (!in_str(scope, scopes));
    scopes.push_back(scope);

  ForwardCleanupToPool(back_node->L_Node, scope);
  ForwardCleanupToPool(back_node->R_Node, scope);

  
  to_pool_forward(back_node->dims_prod, back_node->tensor_ptr, scope, "");
  to_free_tensor_forward(back_node, scope);
}


void CleanScopeTensors(std::string scope)
{
  for(DT_tensor *tensor : forward_Tensors_to_free[scope])
    delete tensor;

  std::vector<float*> scope_tensors_ptrs;

  //for(auto &pair : scope_tensors[scope])
  //  scope_tensors_ptrs.push_back(pair.second);

  for(std::tuple<int, float *, std::string> pair : forward_tensors_to_pool[scope])
  {
    //if(!in_float_ptr_vec(std::get<1>(pair), scope_tensors_ptrs))
      move_to_pool(0, std::get<0>(pair), std::get<1>(pair), std::get<2>(pair));

    //move_to_pool(pair.first, pair.second);
    //cudaCheck(cudaFree(std::get<1>(pair)));
  }

  forward_Tensors_to_free[scope].clear();
  forward_tensors_to_pool[scope].clear();
  forward_tensors_sent_to_pool[scope].clear();

  forward_Tensors_to_free.erase(scope);
  forward_tensors_to_pool.erase(scope);
  forward_tensors_sent_to_pool.erase(scope);
}

extern "C" float clean_forward(Scope_Struct *scope_struct)
{//TODO: break? clears threaded tensors
  for(std::string _scope : scopes)
    CleanScopeTensors(_scope);
  scopes.clear();
  
  cudaCheck(cudaGetLastError());
  return 0;
}


void CleanTree_Backprop(DT_tensor *back_node) {
  // Avoid calling CleanTree separatly. As this has the overhead of goign throughout the tree multiple times.

  if (back_node==nullptr)
    return;
  if (back_node->weight)
    return;
  CleanTree_Backprop(back_node->L_Node);
  CleanTree_Backprop(back_node->R_Node);

  float dims_prod = back_node->dims_prod;
  to_pool(dims_prod, back_node->tensor_ptr, "CleanTree leaf tensor"); 

  if (!back_node->is_last_version)
    to_free_tensor(back_node);
}


void CleanTreeNow(int thread_id, DT_tensor *tensor, std::string root_name) {
  if (tensor==nullptr)
    return;
  
  if (tensor->weight || (tensor->leaf && tensor->name!=root_name))
    return;

  CleanTreeNow(thread_id, tensor->L_Node, root_name);
  CleanTreeNow(thread_id, tensor->R_Node, root_name);

  float dims_prod = tensor->dims_prod;

  move_to_pool(thread_id, tensor->dims_prod, tensor->tensor_ptr, "eval/thread_id!=0 cleaning");
  free(tensor);
}