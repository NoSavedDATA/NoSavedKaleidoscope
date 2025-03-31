#pragma once


#include <cuda_fp16.h>

#include <string>
#include <vector>
#include <map>

extern std::map<int, std::map<float, std::vector<float *>>> TensorPool;
extern std::map<int, std::map<float, std::vector<half *>>> TensorHalfPool;

float *get_from_pool(int thread_id, float dims_prod, std::string from);


void move_to_pool(int thread_id, float dims_prod, float *tensor_ptr, std::string from);
void move_to_pool(int thread_id, float dims_prod, half *tensor_ptr, std::string from);


half *get_half_from_pool(int thread_id, float dims_prod, std::string from);

void move_to_pool_pow2(int thread_id, float dims_prod, float *tensor_ptr, std::string from);

float *get_from_pool_pow2(int thread_id, float dims_prod, std::string from);


