#pragma once

#include "../tensor/include.h"

extern "C" float CreateTensorOnDemand(char *tensor_name, char *scopeless_name, char *init, int is_weight, int thread_id, char *scope);

extern "C" void *gpu(int thread_id, Tensor *tensor, Tensor *pinned_tensor);
extern "C" float gpuw(int thread_id, Tensor *tensor, Tensor *pinned_tensor, float idx);
