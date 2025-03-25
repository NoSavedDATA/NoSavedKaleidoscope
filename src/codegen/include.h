#pragma once

#include<random> 

#include"functions.h"
#include"random.h"
#include"string.h"
#include"tensor_functions.h"
#include"tensor_dim_functions.h"
#include"time.h"
#include"vec.h"

using uint32 = uint32_t;


extern std::vector<Tensor *> todo_backward_tensors;


extern std::map<std::string, std::string> AuxRandomStrs;
extern std::map<std::string, std::vector<char *>> StrVecAuxHash;
extern std::map<std::string, std::vector<float>>  FloatVecAuxHash;


extern pthread_mutex_t mutex, clean_scope_mutex, char_pool_mutex, vocab_mutex, random_seed_mutex, aux_mutex;


extern LCG rng;

extern std::random_device rd2; // it is already defined at cu_common.h
extern std::mt19937 MAIN_PRNG;