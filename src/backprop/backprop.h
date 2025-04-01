#pragma once


extern std::vector<Tensor *> todo_backward_tensors;
extern std::map<std::string, float *> NamedParamGrads;