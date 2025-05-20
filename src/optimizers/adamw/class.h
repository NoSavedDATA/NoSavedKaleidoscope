#pragma once


#include <string>
#include <vector>

#include "../../mangler/scope_struct.h"
#include "../interface.h"

class AdamW_optim : public Optimizer {
  float lr, beta1, beta2, weight_decay, grad_clip;

  public:
    AdamW_optim(float lr, float beta1, float beta2, float weight_decay, float grad_clip);
    
  void init_states(std::string param_name, float params_count) override;
  void step(float *param, float *grad, std::vector<int> dims, std::string param_name, cudaStream_t stream) override;
  void sparse_step(float *, float *, float *, std::vector<int>, std::vector<int> dims, std::string param_name, cudaStream_t stream) override;
};



extern "C" float AdamW(Scope_Struct *, float lr, float beta1, float beta2, float weight_decay, float grad_clip);
