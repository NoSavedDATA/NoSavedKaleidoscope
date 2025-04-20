#pragma once

#include <map>
#include <vector>
#include <string>

#include "../mangler/scope_struct.h"
#include "../tensor/tensor_struct.h"

extern std::vector<Tensor *> todo_backward_tensors;
extern std::map<std::string, float *> NamedParamGrads;


void TraversePreOrder(Tensor *back_node, float *device_dy, bool from_gradless, bool from_custom, int parent_op);


extern "C" float backprop(Scope_Struct *);
