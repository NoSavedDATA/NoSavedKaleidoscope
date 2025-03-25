#pragma once

#include<random> 
#include <map>

#include"../threads/include.h"
#include"functions.h"
#include"random.h"
#include"string.h"
#include"tensor_dim_functions.h"
#include"time.h"
#include"vec.h"

using uint32 = uint32_t;




extern std::map<std::string, std::string> AuxRandomStrs;
extern std::map<std::string, std::vector<char *>> StrVecAuxHash;
extern std::map<std::string, std::vector<float>>  FloatVecAuxHash;




extern LCG rng;

extern std::random_device rd2; // it is already defined at cu_common.h
extern std::mt19937 MAIN_PRNG;