#pragma once


#include <cuda_fp16.h>

#include <string>
#include <vector>
#include <map>

extern std::map<int, std::map<int, std::vector<float *>>> TensorPool;
extern std::map<int, std::map<int, std::vector<half *>>> TensorHalfPool;



int round_to_nearest_pow2(int x); 

float *get_from_pool(int thread_id, int dims_prod, std::string from, bool is_new=false);


void move_to_pool(int thread_id, int dims_prod, float *tensor_ptr, std::string from);
void move_to_pool(int thread_id, int dims_prod, half *tensor_ptr, std::string from);


half *get_half_from_pool(int thread_id, int dims_prod, std::string from);

void move_to_pool_pow2(int thread_id, int dims_prod, float *tensor_ptr, std::string from);

float *get_from_pool_pow2(int thread_id, int dims_prod, std::string from);


