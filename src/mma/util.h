#pragma once

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <mma.h>


using namespace nvcuda;




struct Grid {

  dim3 g;
  dim3 b;
  dim3 w;


  int smem;

  int wx_per_bx;
  int wy_per_by;

  inline void NewGrid(int gx, int gy, int bx, int by);
  inline void SetWarpSize(int wx, int wy);
};


inline Grid CalculateBlockingSize(int M, int N);




struct Grid2 {

  
  dim3 g;
  dim3 b;
  dim3 w;


  int smem;

  int bx_per_w;
  int by_per_w;

  int wx, wy, bx_per_wx, by_per_wy, wx_per_mma_m, wy_per_mma_n;

  inline void NewGrid(int gx, int gy, int bx, int by);

  inline void SetWarpSize(int warps, int wx, int wy);
};


inline Grid2 CalculateBlockingSize2(int M, int N);







__inline__ __device__ uint32_t cast_smem_ptr_to_uint(void *smem_ptr);





__inline__ __device__ int smem_xor_cp_async(int lane_id);



__inline__ __device__ int smem_dexor_from_cp_async(int strided, int contiguous);










// matrix_a row_major
template <class Func>
__device__ inline void
wmma_foreach_ij(wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> &frag,
           Func func);


// matrix_a row_major other warp info
template <class Func>
__device__ inline void
wmma_foreach_ij(wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> &frag,
           Func func, int other_warp, const int other_warp_max);


// matrix_b col_major
template <class Func>
__device__ inline void
wmma_foreach_ij(wmma::fragment<wmma::matrix_b, 16, 16, 16, half,
                                  wmma::col_major> &frag,
           Func func);

// accumulator
template <class T, class Func>
__device__ inline void
wmma_foreach_ij(wmma::fragment<wmma::accumulator, 16, 16, 16, T> &frag,
           Func func);
// accumulator
template <class Func>
__device__ inline void
wmma_foreach_ij_acc(Func func);






__device__ __inline__ void smem_xor_to_reg_A(wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> &frag,
                                  const float *smem, const int ld, const int k_stride);

__device__ __inline__  void smem_xor_to_reg_B(wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> &frag,
                                  const float *smem, const int ld, const int k_stride);



__device__ void smem_xor_to_reg_A_ec(wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> &frag,
                                     wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> &frag_delta,
                                     const float *smem, const int ld, const int k_stride);



__device__ void smem_xor_to_reg_B_ec(wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> &frag,
                                     wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> &frag_delta,
                                     const float *smem, const int ld, const int k_stride);











__inline__ __device__ void frag_to_mem(const float *frag, float *gmem, const int ld);


__inline__ __device__ void frag_to_mem_ec(const float *frag, const float *delta_frag, float *gmem, const int ld);




