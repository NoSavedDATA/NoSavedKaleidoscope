#pragma once

#include <mma.h>

using namespace nvcuda;


__device__ void smem_xor_to_reg_A_ec(wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> &frag,
                                     wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> &frag_delta,
                                     const float *smem, const int ld, const int k_stride)
{
  const auto func = [&](const unsigned* frag_index_list,
        const unsigned fragment_index_count,
        const unsigned i,
        const unsigned j) {
      

          int wi = i/4;
          int xi = i%4;

          int xj = j/4;
          int wj = j%4;

          int offset = smem_dexor_from_cp_async(xi, xj*2 + k_stride)+wj;

        
          float _fp32 = *(smem + (wi*4)*ld + offset);
          __half _fp16 = __float2half(_fp32);
          __half _delta = __float2half((_fp32 - __half2float(_fp16))*2048);

  #pragma unroll
          for (unsigned f = 0; f < fragment_index_count; f++)
          {
            frag.x[frag_index_list[f]] = _fp16;
            frag_delta.x[frag_index_list[f]] = _delta;
          }
    };

  wmma_foreach_ij(
      frag,
      func
    );
  __syncwarp();
}


__device__ void smem_xor_to_reg_B_ec(wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> &frag,
                                     wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> &frag_delta,
                                     const float *smem, const int ld, const int k_stride)
{
  const auto func = [&](const unsigned* frag_index_list,
        const unsigned fragment_index_count,
        const unsigned i,
        const unsigned j) {
      

          int wj = j/4;
          int xj = j%4;
        
          int xi = i/4;
          int wi = i%4;


          int offset = smem_dexor_from_cp_async(xj, xi*2+k_stride)+wi;

          float _fp32 = *(smem + (wj*4)*ld + offset);
          __half _fp16 = __float2half(_fp32);
          __half _delta = __float2half((_fp32 - __half2float(_fp16))*2048);


  #pragma unroll
          for (unsigned f = 0; f < fragment_index_count; f++)
          {
            frag.x[frag_index_list[f]] = _fp16;
            frag_delta.x[frag_index_list[f]] = _delta;
          }
    };

  wmma_foreach_ij(
      frag,
      func
    );
  __syncwarp();
}











__inline__ __device__ void smem_xor_to_reg_A(wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> &frag,
                                  const float *smem, const int ld, const int k_stride)
{
  const auto func = [&](const unsigned* frag_index_list,
        const unsigned fragment_index_count,
        const unsigned i,
        const unsigned j) {
      

          int wi = i/4;
          int xi = i%4;

          int xj = j/4;
          int wj = j%4;


          int offset = smem_dexor_from_cp_async(xi, xj*2 + k_stride)+wj;

        
          __half tmp = __float2half(*(smem + (wi*4)*ld + offset));
  #pragma unroll
          for (unsigned f = 0; f < fragment_index_count; f++)
            frag.x[frag_index_list[f]] = tmp;
    };

  wmma_foreach_ij(
      frag,
      func
    );
  __syncwarp();
}


__inline__ __device__ void smem_xor_to_reg_B(wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> &frag,
                                  const float *smem, const int ld, const int k_stride)
{
  const auto func = [&](const unsigned* frag_index_list,
        const unsigned fragment_index_count,
        const unsigned i,
        const unsigned j) {
      

          int wj = j/4;
          int xj = j%4;
        
          int xi = i/4;
          int wi = i%4;


          int offset = smem_dexor_from_cp_async(xj, xi*2+k_stride)+wi;

        
          __half tmp = __float2half(*(smem + (wj*4)*ld + offset));
  #pragma unroll
          for (unsigned f = 0; f < fragment_index_count; f++)
            frag.x[frag_index_list[f]] = tmp;
    };

  wmma_foreach_ij(
      frag,
      func
    );
  __syncwarp();
}
























__inline__ __device__ void smem_xor_to_reg_A(int8_t *frag,
                                  const int8_t *smem, const int k_stride)
{
  const auto func = [&](const unsigned* frag_index_list,
        const unsigned fragment_index_count,
        const unsigned i,
        const unsigned j) {
      

      int8_t tmp = *(smem + k_stride*256 + i*16 + j);


      // frag.x[frag_index_list[0]] = tmp;
      frag[frag_index_list[0]] = tmp;
    };

  wmma_foreach_ij_a(
      func
    );
  __syncwarp();
}


__inline__ __device__ void smem_xor_to_reg_B(int8_t *frag,
                                  const int8_t *smem, const int k_stride)
{
  const auto func = [&](const unsigned* frag_index_list,
        const unsigned fragment_index_count,
        const unsigned i,
        const unsigned j) {
              
      
      int8_t tmp = *(smem + k_stride*256 + i*16 + j);

      
      // frag.x[frag_index_list[0]] = tmp;
      frag[frag_index_list[0]] = tmp;
    };

  wmma_foreach_ij_b(
      func
    );
  __syncwarp();
}