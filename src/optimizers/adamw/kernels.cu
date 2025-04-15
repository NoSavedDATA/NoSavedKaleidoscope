
#include <algorithm>

#include "../../cuda_kernels/elementwise_kernels_inline.cu"



__global__ void adamw_kernel(float* params_memory, const float* grads_memory, float* m_memory, float* v_memory, long num_parameters,
                              float learning_rate, float beta1, float beta2, float beta1_correction, float beta2_correction,
                              const float eps, const float weight_decay, const float grad_clip) {

   int i = blockIdx.x * blockDim.x + threadIdx.x;
   if (i >= num_parameters) return;  // guard
   
  //  float grad = std::clamp(grads_memory[i], -grad_clip, grad_clip);
   float grad = grads_memory[i];
   float m = m_memory[i];
   float v = v_memory[i];
   // update the first moment (momentum)
   m = lerp(grad, m, beta1);
   m_memory[i] = m;
   // update the second moment (RMSprop)
   v = lerp(grad * grad, v, beta2);
   v_memory[i] = v;
   m /= beta1_correction;  // m_hat
   v /= beta2_correction;  // v_hat
   params_memory[i] -= learning_rate * (m / (sqrtf(v) + eps) + weight_decay * params_memory[i]);
}






__global__ void sparse_adamw_kernel(float* params_memory, const float* grads_memory, const float *idx_tensor,
                              float* m_memory, float* v_memory, long num_parameters, const int C,
                              const float learning_rate, const float beta1, const float beta2, const float beta1_correction, const float beta2_correction,
                              const float eps, const float weight_decay, const float grad_clip) {

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= num_parameters) return;  // guard

  int b = i / C;
  int c = i % C;


  int idx = (int)idx_tensor[b]; 


  float grad = std::clamp(grads_memory[i], -grad_clip, grad_clip);
  float m = m_memory[idx*C + c];
  float v = v_memory[idx*C + c];
  // update the first moment (momentum)
  m = lerp(grad, m, beta1);
  // update the second moment (RMSprop)
  v = lerp(grad * grad, v, beta2);
  
  m /= beta1_correction;  // m_hat
  v /= beta2_correction;  // v_hat

  //float *param = params_memory + idx*C + c;
  float p = params_memory[idx*C + c];

  p = p - learning_rate * (m / (sqrtf(v) + eps) + weight_decay * params_memory[i]);
  //atomicAdd(param, -1*(learning_rate * (m / (sqrtf(v) + eps) + weight_decay * params_memory[i])));
  __threadfence();
  m_memory[idx*C + c] = m;
  v_memory[idx*C + c] = v;
  params_memory[idx*C + c] = p;
}