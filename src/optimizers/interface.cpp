
#include <cuda_runtime.h>
#include <map>
#include <string>
#include <vector>


#include "interface.h"

void Optimizer::init_states(std::string, float) {}
void Optimizer::step(float *, float *, std::vector<float>, std::string, cudaStream_t) {}
void Optimizer::sparse_step(float *, float *, float *, std::vector<float>, std::vector<float>, std::string, cudaStream_t) {}
void Optimizer::count_step() {
    timestep+=1;
}