#pragma once


extern std::map<int, std::map<float, std::vector<float *>>> TensorPool;
extern std::map<int, std::map<float, std::vector<half *>>> TensorHalfPool;

float *get_from_pool(int thread_id, float dims_prod, std::string from);
