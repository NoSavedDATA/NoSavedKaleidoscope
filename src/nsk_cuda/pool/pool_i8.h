#pragma once


#include <cuda_fp16.h>

#include <string>
#include <vector>
#include <map>

extern std::map<int, std::map<int, std::vector<int8_t *>>> TensorPool_i8;


int8_t *get_i8pool(int thread_id, int dims_prod, std::string from);
void move_to_i8pool(int thread_id, int dims_prod, int8_t *tensor_ptr, std::string from);