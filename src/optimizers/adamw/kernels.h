#pragma once




__global__ void adamw_kernel(float* params_memory, const float* grads_memory, float* m_memory, float* v_memory, long num_parameters,
                              float learning_rate, float beta1, float beta2, float beta1_correction, float beta2_correction,
                              const float eps, const float weight_decay, const float grad_clip); 






__global__ void sparse_adamw_kernel(float* params_memory, const float* grads_memory, const float *idx_tensor,
                              float* m_memory, float* v_memory, long num_parameters, const int C,
                              const float learning_rate, const float beta1, const float beta2, const float beta1_correction, const float beta2_correction,
                              const float eps, const float weight_decay, const float grad_clip); 