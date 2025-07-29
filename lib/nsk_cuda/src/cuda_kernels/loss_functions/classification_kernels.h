#pragma once


// Parallelizes over B, C
__global__ void crossentropy_softmax_backward_kernel1(float* dlogits,
    const float* probs, const float* targets,
    int B, int C, float scale); 


__global__ void crossentropy_idx_backward_kernel(float* dlogits,
                           const float* probs, const float* targets,
                           int B, int C, float scale); 