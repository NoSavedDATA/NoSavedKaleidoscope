
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



extern "C" float pinned_tensor_Create(char *tensor_name, char *scopeless_name, float init_val, AnyVector *notes_vector, int thread_id, char *scope);

extern "C" void CreatePinnedTensorOnDemand(char *tensor_name, char *init);
