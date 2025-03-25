
#include<string>
#include<vector>
#include<map>
#include<cstring>
#include<random>
#include<thread>

#include "../backprop/include.h"
#include "../codegen/tensor_dim_functions.h"
#include "../common/include.h"
#include "../tensor/include.h"
#include "include.h"


extern "C" void CreatePinnedTensorOnDemand(char *tensor_name, char *init);
