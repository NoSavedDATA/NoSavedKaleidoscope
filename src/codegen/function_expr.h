#pragma once

#include "../tensor/tensor_struct.h"
#include "include.h"


extern "C" void StoreArgOnDemand(char *scope, char *name, float value);


extern "C" float CopyArgTensor(Tensor *tensor, char *new_tensor_name, char *previous_scope, char *scope, int thread_id);
