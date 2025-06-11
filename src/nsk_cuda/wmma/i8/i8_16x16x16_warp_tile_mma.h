#pragma once

#include <mma.h>

#include "../../structs/i8_wmma_frags.h"
#include "ptx.h"

using namespace nvcuda;



#define LDMATRIX_X2(R0, R1, addr) \
    asm volatile("ldmatrix.sync.aligned.x2.m8n8.shared.b16 {%0, %1}, [%2];\n" : "=r"(R0), "=r"(R1) : "r"(addr))

#define LDMATRIX_X4(R0, R1, R2, R3, addr)                                             \
    asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n" \
                 : "=r"(R0), "=r"(R1), "=r"(R2), "=r"(R3)                             \
                 : "r"(addr))


template<int warp_rows_per_m, int warp_cols_per_n>
__device__ inline void warp_tiled_wmma_i8_16x16x16_mma(int *out, int *reg_A, int *reg_B,
                                              wmma_indexes<warp_rows_per_m, warp_cols_per_n>& wmma_idx,
                                              const int M, const int N, const int WMMA_M, const int WMMA_N)
{


    // for (int wx_tile=0; wx_tile<1; ++wx_tile)
    for (int wx_tile=0; wx_tile<warp_rows_per_m; ++wx_tile)
    { 
        // for (int wy_tile=0; wy_tile<1; ++wy_tile)
        for (int wy_tile=0; wy_tile<warp_cols_per_n; ++wy_tile)
        {
            if ((wmma_idx.block_y*wmma_idx.blocking_size_y + wy_tile*WMMA_M)<M && (wmma_idx.block_x*wmma_idx.blocking_size_x + wx_tile*WMMA_N)<N)
            wmma16x16x16_i8_mma(out+(wx_tile*warp_cols_per_n + wy_tile)*8, reg_A+wy_tile*4, reg_B+wx_tile*4); // 8 is the frag ld  
        }
    }
    // __syncthreads();
    
    // if(threadIdx.x==0)
    // {
    //     printf("out is: ");
    //     for (int i=0;i<8;++i)
    //         printf("%d, ", out[i]);
    //     printf("\n");

    // }
    

}






template<int warp_rows_per_m, int warp_cols_per_n, typename T>
__device__ inline void wmma_i8_m16n16k16_mma(int *out,
                                        wmma_indexes<warp_rows_per_m, warp_cols_per_n>& wmma_idx,
                                        smem_cpasync_wmma_loader<warp_rows_per_m, warp_cols_per_n, T>& smem_loader,
                                        float *x_smem, float *w_smem,
                                        const int M, const int N, const int K,
                                        const int WMMA_M, const int WMMA_N,
                                        int chunks)
{
    // chunks = 1;
    size_t reg_store_idx = 0;
    size_t reg_load_idx = 0;

    // int reg_A[2*warp_cols_per_n*4];
    // int reg_B[2*warp_rows_per_m*4];

    int reg_A[2*warp_cols_per_n*4];
    int reg_B[2*warp_rows_per_m*4];



    for (int k_stride=0; k_stride<chunks; ++k_stride)
    {
        // smem_loader.store_frag_A(frag_loader, x_smem, WMMA_M, k_stride);
        // smem_loader.store_frag_B(frag_loader, w_smem, WMMA_N, k_stride);


        // if (threadIdx.x<32)
        // {
            

        for (int i=0; i<warp_cols_per_n; ++i)
        {

            uint32_t x_smem_casted = __cvta_generic_to_shared(x_smem +\
                smem_loader.xor_load_offset + \
                (wmma_idx.warp_y*wmma_idx.wy + i*WMMA_N)*8 + \
                k_stride*64 + \ 
                ((wmma_idx.laneId/16)*16 + wmma_idx.laneId%16)*4);
                
            asm volatile("ldmatrix.sync.aligned.x2.m8n8.shared.b16 {%0, %1}, [%2];\n" \
                    : "=r"(reg_A[(reg_store_idx*warp_cols_per_n + i)*4]), "=r"(reg_A[(reg_store_idx*warp_cols_per_n + i)*4+1])
                    : "r"(x_smem_casted));
        }
                    


        for (int i=0; i<warp_rows_per_m; ++i)
        {

            uint32_t w_smem_casted = __cvta_generic_to_shared(w_smem +\
                smem_loader.xor_load_offset + \
                (wmma_idx.warp_x*wmma_idx.wx + i*WMMA_M)*8 + \
                k_stride*64 + \
                ((wmma_idx.laneId/16)*16 + wmma_idx.laneId%16)*4);
                //   (((wmma_idx.laneId/8) %2)*16 + wmma_idx.laneId%8)*4);
                
            asm volatile("ldmatrix.sync.aligned.x2.m8n8.shared.b16 {%0, %1}, [%2];\n" \
                    : "=r"(reg_B[(reg_store_idx*warp_rows_per_m + i)*4]), "=r"(reg_B[(reg_store_idx*warp_rows_per_m + i)*4+1])
                    : "r"(w_smem_casted));
        }

                    
                    

        // reg_load_idx^=1;
        // reg_store_idx^=1;

        // __syncthreads();

        // if (threadIdx.x==0)
        // {
        //     printf("chunk_A thread 0:\n");
        //     for(int k=0; k<2; ++k)
        //     {
        //         printf("load at %d-%d-%d\n", reg_store_idx, i, k);
        //         int8_t *i8_regA = (int8_t *)(&reg_A[k]);

        //         printf("\tchunk %d/%d: %d, %d, %d, %d\n", k_stride, k, (int)i8_regA[0], (int)i8_regA[1], (int)i8_regA[2], (int)i8_regA[3]);
        //     }
        //     // printf("Chunk B %d loaded: %d, %d, %d, %d\n", k_stride, (int)frag_loader.x_frag[0], (int)frag_loader.x_frag[1], (int)frag_loader.x_frag[2], (int)frag_loader.x_frag[3]);
        //     printf("\n");
        // }
        // __syncthreads();

        // if (threadIdx.x==1)
        // {
        //     printf("chunk_A thread 1:\n");
        //     for(int k=0; k<2; ++k)
        //     {
        //         printf("load at %d-%d-%d\n", reg_store_idx, i, k);
        //         int8_t *i8_regA = (int8_t *)(&reg_A[k]);

        //         printf("\tchunk %d/%d: %d, %d, %d, %d | %d, %d\n", k_stride, k, (int)i8_regA[0], (int)i8_regA[1], (int)i8_regA[2], (int)i8_regA[3], (int)i8_regA[8], (int)i8_regA[9]);
        //     }
        //     // printf("Chunk B %d loaded: %d, %d, %d, %d\n", k_stride, (int)frag_loader.x_frag[0], (int)frag_loader.x_frag[1], (int)frag_loader.x_frag[2], (int)frag_loader.x_frag[3]);
        //     printf("\n");
        // }
        // __syncthreads();
        // if (threadIdx.x==31)
        // {
        //     printf("chunk_A thread 31:\n");
        //     for(int k=0; k<2; ++k)
        //     {
        //         printf("load at %d-%d-%d\n", reg_store_idx, i, k);
        //         int8_t *i8_regA = (int8_t *)(&reg_A[k]);

        //         printf("\tchunk %d/%d: %d, %d, %d, %d | %d, %d\n", k_stride, k, (int)i8_regA[0], (int)i8_regA[1], (int)i8_regA[2], (int)i8_regA[3], (int)i8_regA[8], (int)i8_regA[9]);
        //     }
        //     // printf("Chunk B %d loaded: %d, %d, %d, %d\n", k_stride, (int)frag_loader.x_frag[0], (int)frag_loader.x_frag[1], (int)frag_loader.x_frag[2], (int)frag_loader.x_frag[3]);
        //     printf("\n");
        // }
        // __syncthreads();

        warp_tiled_wmma_i8_16x16x16_mma(out, reg_A, reg_B, wmma_idx, M, N, WMMA_M, WMMA_N);
        // }
    }
}
