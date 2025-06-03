#pragma once


#include <cuda_fp16.h>

#include <string>
#include <vector>
#include <map>

extern std::map<int, std::map<int, std::vector<int *>>> TensorPool_i32;


int *get_i32pool(int thread_id, int dims_prod, std::string from);
void move_to_i32pool(int thread_id, int dims_prod, int *tensor_ptr, std::string from);