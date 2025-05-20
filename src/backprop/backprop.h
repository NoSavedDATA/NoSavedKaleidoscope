#pragma once

#include <functional>
#include <map>
#include <vector>
#include <string>

#include "../mangler/scope_struct.h"
#include "../tensor/tensor_struct.h"

extern std::vector<DT_tensor *> todo_backward_tensors;
extern std::map<std::string, float *> NamedParamGrads;

extern std::map<std::string, std::function<void(float *, int, float *, float *, float *, std::string, DT_tensor *)>> backward_functions;


void TraversePreOrder(DT_tensor *back_node, float *device_dy, bool from_custom, int parent_op);




