
#include <iostream> 
#include "quantize_f32_i8.h"


void quantize_f32_to_i8(int8_t *x8, float *x, float quant, int M, int N, cudaStream_t stream)
{
    float idx = quant * (M-1);
    int upper = (int)std::ceilf(idx);
    int lower = (int)std::floorf(idx);
    float fraction = idx - lower;

    const int num_warps = 8;



    const int max_M = 1536;
    // Todo: change to dynamic max_M
    // int max_M = deviceProp.sharedMemPerBlock/sizeof(float)/num_warps;

    // std::floor((OC + (num_warps_x*WMMA_T - 1)) / (float)(num_warps_x*WMMA_T))
    
    dim3 grid_size(std::floor((N + (num_warps - 1)) / (float)(num_warps)), std::floor((M + (max_M - 1)) / (float)(max_M)));
    
    int smem = std::min(M, max_M)*num_warps*sizeof(float);
    

    // std::cout << "fraction: " << fraction << " lower " << lower << " upper " << upper << ".\n";
    
    quantize_f32_i8_kernel<<<grid_size, num_warps*32, smem, stream>>>(x8, x, fraction, lower, upper, M, N, max_M, num_warps, M*N);

}