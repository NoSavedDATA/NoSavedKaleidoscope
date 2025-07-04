#include <cuda_runtime.h>
#include <string>
#include <vector>

#include "../../common/cu_commons.h"
#include "../../cuda_kernels/calculate_grids.h"
#include "../../mangler/scope_struct.h"
#include "../../nsk_cuda/pool/include.h"
#include "../../tensor/tensor_dim_functions.h"

#include "../common.h"
#include "class.h"
#include "kernels.h"

SGD_optim::SGD_optim(float lr, float momentum, float weight_decay, float grad_clip)
  : lr(lr), momentum(momentum), weight_decay(weight_decay), grad_clip(grad_clip) {}
      


void SGD_optim::init_states(std::string param_name, float params_count)
{
  

  if (NamedM[param_name]==nullptr)
  {
    std::cout << "init_states for param " << param_name << " with params count: " << params_count << "\n";

    float *m, *device_m;

    m = new float[params_count];
    m = make_zeros_float(params_count);

    cudaMalloc(&device_m, round_to_nearest_pow2(params_count)*sizeof(float));
    cudaMemcpy(device_m, m, params_count*sizeof(float), cudaMemcpyHostToDevice);

    delete[] m;

    NamedM[param_name] = device_m;
  }
}

void SGD_optim::step(float *param, float *grad, std::vector<int> dims, std::string param_name, cudaStream_t stream)
{
  float *m = NamedM[param_name];

 
  int params_count = DimsProd(dims);
  int grid_size, block_size; 
  CalculateGridAndBlockSizes(params_count, grid_size, block_size);

  sgd_kernel<<<grid_size, block_size, 0, stream>>>(param, grad, m, params_count,
                                           lr, momentum, weight_decay, grad_clip);
}

void SGD_optim::sparse_step(float *param, float *grad, float *idx, std::vector<int> idx_dims, std::vector<int> dims, std::string param_name, cudaStream_t stream)
{
  float *m = NamedM[param_name];
}


extern "C" float SGD(Scope_Struct *scope_struct, float lr, float momentum, float weight_decay, float grad_clip)
{

  if (optimizer==nullptr)
    optimizer = std::make_unique<SGD_optim>(lr, momentum, weight_decay, grad_clip);

  optimizer = optimize(std::move(optimizer));

  return 0;
}