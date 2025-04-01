
#include "include.h"

std::vector<Tensor *> todo_backward_tensors;
std::map<std::string, float *> NamedParamGrads;