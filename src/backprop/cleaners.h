#pragma once

#include <string>
#include <map>
#include <vector>

#include "../mangler/scope_struct.h"
#include "../tensor/include.h"


// Cleaners
extern std::map<std::string, float *> var_to_grad;
extern std::vector<std::tuple<float, float *, std::string>> backprop_tensors_to_pool;
extern std::vector<float *> tensors_sent_to_pool;
extern std::vector<Tensor *> backprop_Tensors_to_free;
extern std::vector<Tensor *> backprop_Tensors_to_save;
extern std::map<std::string, std::vector<std::tuple<float, float*,std::string>>> forward_tensors_to_pool;
extern std::map<std::string, std::vector<float*>> forward_tensors_sent_to_pool;
extern std::map<std::string, std::vector<Tensor*>> forward_Tensors_to_free;
extern std::map<std::string, std::map<std::string, float*>> scope_tensors; // records last version of a tensor //todo: is this one actually used?
extern std::map<int, std::map<std::string, std::vector<std::tuple<float, float*,std::string>>>> threaded_tensors_to_pool;
extern std::map<int, std::map<std::string, std::vector<float*>>> threaded_tensors_sent_to_pool;
extern std::map<int, std::map<std::string, std::vector<Tensor*>>> threaded_Tensors_to_free;
extern std::map<int, std::map<std::string, std::vector<float*>>> threaded_tensors_to_save;
extern std::map<int, std::map<std::string, std::vector<Tensor*>>> threaded_Tensors_to_save;

using backward_tuple = std::tuple<int, int, int, int, int, float *, float *, float *, std::string, std::string, std::string>;





void to_free_tensor(Tensor *tensor_ptr);

void to_pool(float dims_prod, float *tensor_ptr, std::string from);

void save_from_pool(Tensor *tensor_ptr);



void to_free_tensor_forward(Tensor *tensor_ptr, std::string scope);

void to_pool_forward(float dims_prod, float *tensor_ptr, std::string scope, std::string from);



void to_free_tensor_threaded(Tensor *tensor_ptr, std::string scope, int thread_id);

void to_pool_threaded(float dims_prod, float *tensor_ptr, std::string scope, int thread_id, std::string from);


void ForwardCleanupToPool(Tensor *back_node, std::string scope);
int DoesTreeContainWeight(Tensor *back_node);
void CleanScopeTensors(std::string scope);


void ThreadedCleanupToPool(Tensor *back_node, std::string scope, int thread_id);
void CleanThreadTensors(std::string scope, int thread_id);


void CleanScopeTensors(std::string scope);


extern "C" float clean_forward(Scope_Struct *scope_struct);
