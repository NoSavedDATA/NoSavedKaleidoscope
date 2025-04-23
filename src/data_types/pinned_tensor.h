
#include<string>
#include<vector>
#include<map>
#include<cstring>
#include<random>
#include<thread>

#include "../backprop/include.h"
#include "../common/include.h"
#include "../mangler/scope_struct.h"
#include "../tensor/include.h"
#include "include.h"




extern "C" void pinned_tensor_Store_Idx(char *, float, float, Scope_Struct *);



extern "C" float pinned_tensor_CalculateIdx(char *tensor_name, float first_idx, ...);
