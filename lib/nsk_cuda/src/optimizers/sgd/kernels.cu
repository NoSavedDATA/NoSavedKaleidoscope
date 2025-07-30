


__global__ void sgd_kernel(float* params_memory, const float* grads_memory, float* m_memory, long num_parameters,
                              float learning_rate, float momentum,
                              const float weight_decay, const float grad_clip) {

   int i = blockIdx.x * blockDim.x + threadIdx.x;
   if (i >= num_parameters) return;  // guard
   
  //  float grad = std::clamp(grads_memory[i], -grad_clip, grad_clip) + weight_decay * params_memory[i];
   float grad = grads_memory[i];
   float m = m_memory[i];
   
   // update the first moment (momentum)
   m = m*momentum + grad;
   m_memory[i] = m;
  
   params_memory[i] -= learning_rate * m;
}