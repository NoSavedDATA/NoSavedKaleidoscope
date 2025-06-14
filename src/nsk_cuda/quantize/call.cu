
#include <iostream> 
#include <algorithm>

#include "../minimal_tensor.h"

#include "quantize_f32_i4.h"
#include "quantize_f32_i8.h"



void quantize_f32_to_i8(int8_t *x8, float *x, Minimal_Tensor *scale, float quant, int M, int N, cudaStream_t stream)
{
    int base_num_warps = 8;
    const int num_warp_factor = 2;
    const int num_warps = base_num_warps*num_warp_factor;
    const int max_N  = 1536/num_warp_factor;



    float idx = quant * (max_N-1);
    int upper = (int)std::ceilf(idx);
    int lower = (int)std::floorf(idx);
    float fraction = idx - lower;


    
    dim3 grid_size(std::floor((M + (num_warps - 1)) / (float)(num_warps)));

    int smem_N = std::clamp(N, 256, max_N); // 128 * 2 for the xor swap

    
    int smem = smem_N*num_warps*sizeof(float);
    

    // std::cout << "fraction: " << fraction << " lower " << lower << " upper " << upper << ", M " << M << ", N " << N << ", max_N " << max_N << ", smem_N " << smem_N << ", warps " << num_warps <<".\n";
    // std::cout << "g.x " << grid_size.x << "\n";
    
    quantize_f32_i8_kernel<<<grid_size, num_warps*32, smem, stream>>>(x8, x, (float *)scale->tensor, fraction, lower, upper, M, N, max_N, smem_N, num_warps, M*N);

}


void quantize_f32_to_i4(int8_t *x8, float *x, Minimal_Tensor *scale, float quant, int M, int N, cudaStream_t stream)
{
    int base_num_warps = 8;
    const int num_warp_factor = 2;
    const int num_warps = base_num_warps*num_warp_factor;
    const int max_N  = 1536/num_warp_factor;



    float idx = quant * (max_N-1);
    int upper = (int)std::ceilf(idx);
    int lower = (int)std::floorf(idx);
    float fraction = idx - lower;


    
    dim3 grid_size(std::floor((M + (num_warps - 1)) / (float)(num_warps)));

    int smem_N = std::clamp(N, 256, max_N); // 128 * 2 for the xor swap

    
    int smem = smem_N*num_warps*sizeof(float);
    

    // std::cout << "fraction: " << fraction << " lower " << lower << " upper " << upper << ", M " << M << ", N " << N << ", max_N " << max_N << ", smem_N " << smem_N << ", warps " << num_warps <<".\n";
    // std::cout << "g.x " << grid_size.x << "\n";
    
    quantize_f32_i4_kernel<<<grid_size, num_warps*32, smem, stream>>>(x8, x, (float *)scale->tensor, fraction, lower, upper, M, N, max_N, smem_N, num_warps, M*N);

}




// void quantize_f32_to_i8(int8_t *x8, float *x, Minimal_Tensor *scale, float quant, int M, int N, cudaStream_t stream)
// {
//     float idx = quant * (M-1);
//     int upper = (int)std::ceilf(idx);
//     int lower = (int)std::floorf(idx);
//     float fraction = idx - lower;

//     int base_num_warps = 8;


//     // const int num_warp_factor = std::clamp(N/base_num_warps, 1, 4);
//     const int num_warp_factor = 1;
//     const int num_warps = base_num_warps*num_warp_factor;

//     // Todo: change to dynamic max_M
//     // int max_M = deviceProp.sharedMemPerBlock/sizeof(float)/(num_warps);

//     const int max_M = 1536/num_warp_factor;

//     // std::floor((OC + (num_warps_x*WMMA_T - 1)) / (float)(num_warps_x*WMMA_T))
    
//     dim3 grid_size(std::floor((N + (num_warps - 1)) / (float)(num_warps)), std::floor((M + (max_M - 1)) / (float)(max_M)));
    
//     int smem = std::min(M, max_M)*num_warps*sizeof(float);

    
    

//     std::cout << "fraction: " << fraction << " lower " << lower << " upper " << upper << ", M " << M << ", N " << N << ", max_M " << max_M << ", warps " << num_warps <<".\n";
//     std::cout << "g.x " << grid_size.x << " g.y " << grid_size.y << "\n";
    
//     quantize_f32_i8_kernel<<<grid_size, num_warps*32, smem, stream>>>(x8, x, (float*)scale->tensor, fraction, lower, upper, M, N, max_M, num_warps, M*N);

// }