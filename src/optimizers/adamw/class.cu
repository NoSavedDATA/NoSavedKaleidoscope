
#include <cuda_runtime.h>
#include <string>
#include <vector>

#include "../../common/cu_commons.h"
#include "../../cuda_kernels/calculate_grids.h"
#include "../../mangler/scope_struct.h"
#include "../../tensor/tensor_dim_functions.h"
#include "../../tensor/pool.h"

#include "../common.h"
#include "class.h"
#include "kernels.h"


AdamW_optim::AdamW_optim(float lr, float beta1, float beta2, float weight_decay, float grad_clip)
    : lr(lr), beta1(beta1), beta2(beta2), weight_decay(weight_decay), grad_clip(grad_clip) {}


void AdamW_optim::init_states(std::string param_name, float params_count)
{
  if (NamedV[param_name]==nullptr)
  {
    std::cout << "init_states for param " << param_name << " with params count: " << params_count << "\n";

    float *v, *m, *device_v, *device_m;
    v = new float[params_count];
    m = new float[params_count];

    v = make_zeros_float(params_count);
    m = make_zeros_float(params_count);


    cudaMalloc(&device_v, round_to_nearest_pow2(params_count)*sizeof(float));
    cudaMalloc(&device_m, round_to_nearest_pow2(params_count)*sizeof(float));
    cudaMemcpy(device_v, v, params_count*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(device_m, m, params_count*sizeof(float), cudaMemcpyHostToDevice);

    delete[] v;
    delete[] m;

    NamedV[param_name] = device_v; 
    NamedM[param_name] = device_m;
  }
}

void AdamW_optim::step(float *param, float *grad, std::vector<int> dims, std::string param_name, cudaStream_t stream)
{
  float *v = NamedV[param_name];
  float *m = NamedM[param_name];

  float beta1_correction = 1.0f - powf(beta1, timestep);
  float beta2_correction = 1.0f - powf(beta2, timestep);


  int params_count = DimsProd(dims);
  
  int grid_size, block_size; 
  CalculateGridAndBlockSizes(params_count, grid_size, block_size);

  adamw_kernel<<<grid_size, block_size, 0, stream>>>(param, grad, m, v, params_count,
                                           lr, beta1, beta2, beta1_correction, beta2_correction,
                                           eps, weight_decay, grad_clip);
}


void AdamW_optim::sparse_step(float *param, float *grad, float *idx, std::vector<int> idx_dims, std::vector<int> dims, std::string param_name, cudaStream_t stream)
{
  float *v = NamedV[param_name];
  float *m = NamedM[param_name];

  float beta1_correction = 1.0f - powf(beta1, timestep);
  float beta2_correction = 1.0f - powf(beta2, timestep);


  int leading_dim = dims[dims.size()-1];

  int params_count = DimsProd(idx_dims)*leading_dim;

  
  int grid_size, block_size; 
  CalculateGridAndBlockSizes(params_count, grid_size, block_size);

  //std::cout << "Sparse step " << "\n";
  //PrintDims(idx_dims);
  //PrintDims(dims);
  sparse_adamw_kernel<<<grid_size, block_size, 0, stream>>>(param, grad, idx, m, v, params_count, leading_dim,
                                           lr, beta1, beta2, beta1_correction, beta2_correction,
                                           eps, weight_decay, grad_clip);
}



extern "C" float AdamW(Scope_Struct *scope_struct, float lr, float beta1, float beta2, float weight_decay, float grad_clip)
{

  // std::cout << "AdamW: lr " << lr << " beta1 " << beta1 << " beta2 " << beta2 << " wd " << weight_decay << " grad clip " << grad_clip << ".\n";

  if (optimizer==nullptr)
    optimizer = std::make_unique<AdamW_optim>(lr, beta1, beta2, weight_decay, grad_clip);

  optimizer = optimize(std::move(optimizer));

  return 0;
}