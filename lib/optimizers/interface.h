#pragma once

#include <map>
#include <string>
#include <vector>


class Optimizer {
public:
  virtual ~Optimizer() = default;
  std::map<std::string, float *> NamedV, NamedM;

  int timestep = 1;
  float lr = 0.0f;
  //float eps = 1.5e-4;
  float eps = 1e-8;
    
  virtual void init_states(std::string, float);
  virtual void step(float *, float *, std::vector<int>, std::string, cudaStream_t);
  virtual void sparse_step(float *, float *, float *, std::vector<int>, std::vector<int>, std::string, cudaStream_t);
  virtual void count_step(); 
};