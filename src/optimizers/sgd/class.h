#pragma once

#include <string>
#include <vector>

#include "../../mangler/scope_struct.h"
#include "../interface.h"

class SGD_optim : public Optimizer {
    float lr, momentum, weight_decay, grad_clip;

    public:
        SGD_optim(float lr, float momentum, float weight_decay, float grad_clip);
        
    void init_states(std::string param_name, float params_count) override;
    void step(float *param, float *grad, std::vector<float> dims, std::string param_name, cudaStream_t stream) override;
    void sparse_step(float *, float *, float *, std::vector<float>, std::vector<float> dims, std::string param_name, cudaStream_t stream) override;
};


extern "C" float SGD(Scope_Struct *, float lr, float momentum, float weight_decay, float grad_clip);
