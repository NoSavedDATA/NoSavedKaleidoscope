#include <string>


#include "../backprop/include.h"
#include "../char_pool/include.h"
#include "../common/cu_commons.h"
#include "../cuda_threads/include.h"
#include "../compiler_frontend/include.h"
#include "../tensor/tensor_struct.h"



extern "C" char * ConcatScopeStr(char *lc, char *rc)
{
  // std::cout << "ConcatScopeStr: " << lc << " and " << rc << ".\n";
  std::string lstr = lc;


  if (in_str(lstr, globalVars))
  {

    size_t length_lc = strlen(lc) + 1;
    //char* result_cstr = new char[length_lc+length_rc];
    char *result_cstr = get_from_char_pool(length_lc, "concat free left");
    memcpy(result_cstr, lc, length_lc);
    move_to_char_pool(length_lc+1, lc, "concat free left");
    return result_cstr;
  }
  

  size_t length_lc = strlen(lc);
  size_t length_rc = strlen(rc) + 1; // +1 for null terminator
  //char* result_cstr = new char[length_lc+length_rc];
  char *result_cstr = get_from_char_pool(length_lc+length_rc, "concat free left");
  
  memcpy(result_cstr, lc, length_lc);
  memcpy(result_cstr + length_lc, rc, length_rc);


  move_to_char_pool(length_lc+1, lc, "concat free left");
  //delete[] lc;

  //std::cout << "ConcatScopeStr " << result_cstr << "\n";
  
  return result_cstr;
}

extern "C" char * ConcatScopeAtCallExpr(char *lc, char *rc)
{
  //std::cout << "ConcatScopeAtCallExpr of " << lc << " and " << rc << "\n";
  std::string rstr = rc;

  //for (auto &a : globalVars)
  //  std::cout << "" << a << "\n";

  
  

  size_t length_lc = strlen(lc);
  size_t length_rc = strlen(rc) + 1; // +1 for null terminator
  //char* result_cstr = new char[length_lc+length_rc];
  char *result_cstr = get_from_char_pool(length_lc+length_rc, "ConcatScopeAtCallExpr");
  
  memcpy(result_cstr, lc, length_lc);
  memcpy(result_cstr + length_lc, rc, length_rc);

  
  return result_cstr;
}

extern "C" void AddFloatToScopeCleanList(char *scope, char *name)
{
  std::string _name, _scope;
  _name = name;
  _scope = scope;

  
  //std::cout << "will erase " << name << " from scope " << scope << "\n";
  pthread_mutex_lock(&clean_scope_mutex);
  ScopeVarsToClean[_scope].push_back(std::make_pair(_name, "float"));
  pthread_mutex_unlock(&clean_scope_mutex);
  
}

extern "C" void AddToScopeCleanList(char *scope, char *name)
{
  
  pthread_mutex_lock(&clean_scope_mutex);
  std::vector<std::pair<std::string, std::string>> scope_vars = ScopeVarsToClean[scope];
  
  for(auto &pair : scope_vars)
    if (pair.first==name)
    {
      delete[] name;
      return;
    }
    
  ScopeVarsToClean[scope].push_back(std::make_pair(name, "str"));
  pthread_mutex_unlock(&clean_scope_mutex);
  
  delete[] name;
}



extern "C" void CleanScopeVars(char *scope, int thread_id)
{
  
  pthread_mutex_lock(&clean_scope_mutex);

  std::vector<std::pair<std::string, std::string>> scope_vars = ScopeVarsToClean[scope];

  for (auto _it = ScopeVarsToClean[scope].begin(); _it != ScopeVarsToClean[scope].end(); )
  {
    auto &pair = *_it;
    

    if (pair.second=="str")
    {
      NamedStrs.erase(pair.first);

      /*
      auto it = NamedStrs.find(pair.first);
      if (it != NamedStrs.end())
        NamedStrs.erase(it);
      */      
      
    }
    
    if (pair.second=="float")
    {
      NamedClassValues.erase(pair.first);
      /*
      auto it = NamedClassValues.find(pair.first);
      if (it != NamedClassValues.end())
        NamedClassValues.erase(it);
      */
    }
    
    _it = ScopeVarsToClean[scope].erase(_it);
  }

  if(thread_id!=0)
  {
    
    while(ThreadedScopeTensorsToClean[thread_id][scope].size()>0)
    {
      std::string tensor_name = ThreadedScopeTensorsToClean[thread_id][scope].back();
      ThreadedScopeTensorsToClean[thread_id][scope].pop_back();

      ThreadedCleanupToPool(NamedTensorsT[tensor_name], scope, thread_id);      
    }
    CleanThreadTensors(scope, thread_id);
    ThreadedScopeTensorsToClean[thread_id].erase(scope);
    
  }

  
  //ScopeVarsToClean[scope].clear(); // clear does not actually clears it
  auto it = ScopeVarsToClean.find(scope);
  ScopeVarsToClean.erase(it);

  pthread_mutex_unlock(&clean_scope_mutex);
  
}




extern "C" float RemoveTensorScope(char *tensor_name, char *scope, char *tgt_tensorc, char *previous_scope, int thread_id)
{
  // std::cout << "Removing scope of " << tensor_name << " into " << tgt_tensorc << ".\n";
  std::string tgt_tensor = tgt_tensorc;
  tgt_tensor = previous_scope + tgt_tensor;

  std::string scope_tensor_name = scope;
  scope_tensor_name = scope_tensor_name + tensor_name;
 
  Tensor *tensor, *scope_tensor;
  tensor = NamedTensorsT[tgt_tensor];

  if(tensor->thread_id!=thread_id)
  {
    //std::cout << "\n\n\nRETURNING " << scope_tensor_name << " into " << tgt_tensor << "\n";
    std::cout << "Returning from thread id " << thread_id << " into " << tensor->thread_id << "\n\n\n\n";
    cudaStreamSynchronize(ThreadsStream[thread_id]);
    cudaStreamSynchronize(ThreadsStream[tensor->thread_id]);
  }

  scope_tensor = NamedTensorsT[scope_tensor_name];
  std::vector<float> dims = scope_tensor->dims;
  int dims_prod = scope_tensor->dims_prod;


  std::string _name = "remove tensor scope of ";
  _name = _name + tensor_name;
  if(tensor->thread_id==0)
    move_to_pool(tensor->thread_id, tensor->dims_prod, tensor->tensor_ptr, _name);
  else
    ThreadedScopeTensorsToClean[tensor->thread_id][previous_scope].push_back(tensor->name);
  tensor->AttrTensor(scope_tensor->tensor_ptr, scope_tensor->dims, scope_tensor->dims_prod, scope_tensor->cuda_stream, scope_tensor->loader);
  tensor->from_grad_or_load = scope_tensor->from_grad_or_load;
  tensor->is_last_version = true;
  
  

  if(thread_id!=0)
  {
    threaded_Tensors_to_save[thread_id][scope].push_back(scope_tensor);
    threaded_tensors_to_save[thread_id][scope].push_back(scope_tensor->tensor_ptr);
  } 
  else if(nn_mode==eval_mode)//
    to_free_tensor_forward(scope_tensor, scope);//
  else
    to_free_tensor(scope_tensor);
  
  //delete scope_tensor;
  // NamedTensorsT.erase(scope_tensor_name);
  
  return 0;
}



extern "C" float RemoveTensorScopeAttrOnIndex(char *tensor_name, char *scope, char *tgt_tensorc, char *previous_scope, float idx_at, int thread_id)
{
  std::string scope_tensor_name = scope;
  scope_tensor_name = scope_tensor_name + tensor_name;

  std::string tgt_tensor = tgt_tensorc;
  tgt_tensor = previous_scope + tgt_tensor;


  std::cout << "\n\n\nRETURNING " << scope_tensor_name << " into " << tgt_tensor << " at idx\n\n\n\n";  



  Tensor *tensor, *scope_tensor;
  tensor = NamedTensorsT[tgt_tensor];

  scope_tensor = NamedTensorsT[scope_tensor_name];
  std::vector<float> dims = tensor->dims;
  int dims_prod = tensor->dims_prod;

  int scope_dims_prod = scope_tensor->dims_prod;

  
  if (idx_at>(dims_prod-1))
  {
    std::string _error = "\n\t- Idexating at pos: \033[32m"+std::to_string((int)idx_at);
    _error = _error + "\033[0m on tensor \033[95m"+std::string(tgt_tensor);
    _error = _error + "\033[0m;\n\t- Max idx allowed:  \033[32m"+std::to_string(dims_prod)+"\033[0m.";

    LogErrorS(_error);
    std::cout << "Dimensions:" << "\n";
    PrintDims(dims);
    std::cout << "\n";

    return -1;
  }

  if ((idx_at+scope_dims_prod)>(dims_prod))
  {
    std::string _error = "\n\t- Attributing at pos: \033[32m"+std::to_string((int)idx_at)+"\033[0m with a tensor of size \033[32m"+std::to_string(scope_dims_prod)+"\033[0m";
    _error = _error + "\033[0m on tensor \033[95m"+std::string(tgt_tensor);
    _error = _error + "\033[0m;\n\t- Max idx allowed:  \033[32m"+std::to_string(dims_prod)+"\033[0m.";

    LogErrorS(_error);
    std::cout << "Dimensions:" << "\n";
    PrintDims(dims);
    std::cout << "\n";

    return -1;
  }

  float *base_address = tensor->tensor_ptr;
  float *device_x = base_address + static_cast<int>(idx_at);

  cudaCheck(cudaMemcpy(device_x, scope_tensor->tensor_ptr, scope_dims_prod*sizeof(float), cudaMemcpyDeviceToDevice));
  
  return 0;
}


// extern "C" float print_scope(char *scope, char *previous_scope, int thread_id)
// {

//   std::cout << "\n- Scope is: " << scope << ";\n";
//   std::cout << "- Previous scope was: " << previous_scope << ";\n";
//   std::cout << "- Thread id: " << thread_id << ".\n\n";

//   return 0;
// }