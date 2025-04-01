#pragma once


// Parallelizes over B, C
__global__ void crossentropy_softmax_backward_kernel1(float* dlogits,
                           const float* probs, const float* targets,
                           int B, int C, float scale) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    //int i = threadIdx.x;
    
    
    if (i < B * C) {
        int b = i / (C);
        int v = i % C;

        float *dlogits_b = dlogits + b * C;
        const float *probs_b = probs + b * C;

        //float ix = targets[v];
        float ix = targets[b * C + v]; // one-hot tensor
        float p = probs_b[v];

        //float indicator = (v==ix) ? 1.0f : 0.0f; // one-hot already
        float indicator = ix;

        dlogits_b[v] += (p - indicator) * scale;
        
    }
}

__global__ void crossentropy_idx_backward_kernel(float* dlogits,
    const float* probs, const float* targets,
    int B, int C, float scale) {

int i = blockIdx.x * blockDim.x + threadIdx.x;
//int i = threadIdx.x;


if (i < B * C) {
int b = i / (C);
int v = i % C;

float *dlogits_b = dlogits + b * C;
const float *probs_b = probs + b * C;


float p = probs_b[v];

float indicator = (v==targets[b]) ? 1.0f : 0.0f;
//float indicator = ix;

dlogits_b[v] += (p - indicator) * scale;

}
}