
#include "../smem/gmem_to_smem.h"


__global__ void transpose_kernel(float *y, const float *x, const int M, const int N, const int smem_N, const int num_warps)
{

    const int warpId = threadIdx.x / warpSize;
    const int laneId = threadIdx.x % warpSize;

    const int block_x = blockIdx.x;
    const int B_idx = block_x*num_warps + warpId;

    int xor_swap=0, xor_swap_val=128; // 128 of 32*4 floats of cp.async over the warp

    



    extern __shared__ float smem[];



    float *smem_i = smem + warpId*2*smem_N; // *2 for the xor swap


    const float *x_i = x + B_idx*N;




    int col = laneId*4;
    if(B_idx<M)
        gmem_to_smem_safe(x_i+col, *(smem_i+col), (N-col)*4);
    asm volatile("cp.async.commit_group;");


    for (int tile=0; tile<N; tile+=128)
    {
        int col = tile + laneId*4;
        float *smem_ij = smem_i + xor_swap + laneId*4;

        xor_swap ^= xor_swap_val;

        int next_tile = tile + 128;

        if (next_tile<N) {
            int next_col = next_tile + laneId*4; // jump by 4 copied floats of cp.async (16B)
            const float *x_ij = x_i + next_col;
            
            float *smem_ij_next_tile = smem_i + xor_swap + laneId*4;
            

            if(B_idx<M)
                gmem_to_smem_safe(x_ij, *smem_ij_next_tile, (N-next_col)*4);

            asm volatile("cp.async.commit_group;\n" ::);
            asm volatile("cp.async.wait_group %0;" ::"n"(1));
        } else
            asm volatile("cp.async.wait_all;");


        __syncthreads();
        

        
        #pragma unroll
        for (int i=0; i<4; ++i) {
            if(col+i<N && B_idx<M)
                y[(col+i)*M + B_idx] = smem_ij[i];
        }        
        __syncthreads();
    }
}