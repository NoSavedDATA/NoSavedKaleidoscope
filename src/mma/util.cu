#include "util.h"
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <mma.h>


using namespace nvcuda;



void Grid::NewGrid(int gx, int gy, int bx, int by)
{
  this->g.x = gx;
  this->g.y = gy;
  this->g.z = 1;

  this->b.x = bx;
  this->b.y = by;
  this->b.z = 1;

  smem = (bx+by)*32*sizeof(float);
}

void Grid::SetWarpSize(int wx, int wy)
{
  wx_per_bx = b.x / (wx*16);
  wy_per_by = b.y / (wy*16);

  this->w.x = wx*32;
  this->w.y = wy;
  this->w.z = 1;
}

Grid CalculateBlockingSize(int M, int N)
{

  int bx = 256;
  int by = 128;


  while(bx>M && bx>64)
    bx = bx/2;

  while(by>N && by>64)
    by = by/2;

  int gx = std::floor((M+bx-1)/(float)bx);
  int gy = std::floor((N+by-1)/(float)by);

  Grid grid;

  // std::cout << gx << ", " << gy << ", " << bx << ", " << by << "\n";
  grid.NewGrid(gx, gy, bx, by);


  int wx = fminf(fmaxf(M/16,1),4);
  int wy = fminf(fmaxf(N/16,1),4);

  wx = 4;
  wy = 2;

  grid.SetWarpSize(wx, wy);

  return grid;
}











  



void Grid2::NewGrid(int gx, int gy, int bx, int by)
{
  this->g.x = gx;
  this->g.y = gy;
  this->g.z = 1;

  this->b.x = bx;
  this->b.y = by;
  this->b.z = 1;

  smem = (bx+by)*64*sizeof(float);
}

void Grid2::SetWarpSize(int warps, int wx, int wy)
{
  bx_per_w = b.x / warps;
  by_per_w = b.y / warps;

  this->w.x = warps*32;


  this->wx = wx;
  this->wy = wy;


  bx_per_wx = this->b.x/wx;
  by_per_wy = this->b.y/wy;

  wx_per_mma_m = wx / 16;
  wy_per_mma_n = wy / 16;

}


Grid2 CalculateBlockingSize2(int M, int N)
{

  int bx = 128;
  int by = 64;


  // while(bx>M && bx>64)
  //   bx = bx/2;

  // while(by>N && by>64)
  //   by = by/2;

  int gx = std::floor((M+bx-1)/(float)bx);
  int gy = std::floor((N+by-1)/(float)by);

  Grid2 grid;

  // std::cout << gx << ", " << gy << ", " << bx << ", " << by << "\n";
  grid.NewGrid(gx, gy, bx, by);


  // int wx = fminf(fmaxf(M/16,1),4);
  // int wy = fminf(fmaxf(N/16,1),4);

  int warps = 8;
  int wx = 32;
  int wy = 32;

  grid.SetWarpSize(warps, wx, wy);

  return grid;
}








__inline__ __device__ uint32_t cast_smem_ptr_to_uint(void *smem_ptr) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
}






__inline__ __device__ int smem_xor_cp_async(int lane_id) {
    // *4 means we calculate the storage for 16B/128b words (4 floats), done for each thread
    return ((lane_id % 8) ^ (lane_id / 8) + (lane_id / 8)*8) * 4;
}




__inline__ __device__ int smem_dexor_from_cp_async(int strided, int contiguous) {
    int stride=8;
    
    int tc = contiguous / 8;
    int ts = strided / 4;

    int c = contiguous % 8;
    int s = strided % 4;

    int k_index = c / 2;

    int bank = ((c & 1) * 4) | (s ^ k_index); // e [0, 7]
    
    int offset = tc * 32 + bank + (ts * 4 + k_index) * stride;


    // *4 means we calculate the storage for 16B/128b words (4 floats), done for each thread
    return offset*4;
}










// matrix_a row_major
template <class Func>
__device__ inline void
wmma_foreach_ij(wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> &frag,
           Func func) {

  const unsigned lane_id = threadIdx.x & 0x1f;
  const auto i_offset = lane_id / 4;
  const auto j_offset = (lane_id & 0b11) * 2;
  for (unsigned x = 0; x < frag.num_elements / 2; x++) {
    const unsigned i = i_offset + (x & 0b10) * 4;
    const unsigned j = j_offset + (x & 0b1) + (x & 0b100) * 2;
    const unsigned frag_index_list[2] = {x, x + 8};
    func(frag_index_list, 2, i, j);
  }
}



// matrix_a row_major other warp info
template <class Func>
__device__ inline void
wmma_foreach_ij(wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> &frag,
           Func func, int other_warp, const int other_warp_max) {

  const unsigned lane_id = threadIdx.x & 0x1f;
  const auto i_offset = lane_id / 4;
  const auto j_offset = (lane_id & 0b11) * 2;

  for (unsigned tile = 0; tile < (unsigned)ceilf((frag.num_elements/2)/other_warp_max); tile++) {
    unsigned x = tile*other_warp_max + other_warp;

    const unsigned i = i_offset + (x & 0b10) * 4;
    const unsigned j = j_offset + (x & 0b1) + (x & 0b100) * 2;
    const unsigned frag_index_list[2] = {x, x + 8};
    func(frag_index_list, 2, i, j);
  }
}



// matrix_b col_major
template <class Func>
__device__ inline void
wmma_foreach_ij(wmma::fragment<wmma::matrix_b, 16, 16, 16, half,
                                  wmma::col_major> &frag,
           Func func) {
  const unsigned lane_id = threadIdx.x & 0x1f;
  const auto i_offset = (lane_id & 0b11) * 2;
  const auto j_offset = lane_id / 4;
  for (unsigned x = 0; x < frag.num_elements / 2; x++) {
    const unsigned i = i_offset + (x & 0b1) + (x & 0b10) * 4;
    const unsigned j = j_offset + (x & 0b100) * 2;
    const unsigned frag_index_list[2] = {x, x + 8};
    func(frag_index_list, 2, i, j);
  }
}



// accumulator
template <class T, class Func>
__device__ inline void
wmma_foreach_ij(wmma::fragment<wmma::accumulator, 16, 16, 16, T> &frag,
           Func func) {
  const unsigned lane_id = threadIdx.x & 0x1f;
  const unsigned i_offset = (lane_id >> 2);
  const unsigned j_offset = (lane_id & 0b11) * 2;
  for (unsigned x = 0; x < frag.num_elements; x++) {
    const unsigned j = j_offset + (x & 0b100) * 2 + (x & 0b1);
    const unsigned i = i_offset + (x & 0b10) * 4;
    const unsigned frag_index_list[1] = {x};
    func(frag_index_list, 1, i, j);
  }
}

// accumulator
template <class Func>
__device__ inline void
wmma_foreach_ij_acc(Func func) {
  const unsigned lane_id = threadIdx.x & 0x1f;
  const unsigned i_offset = (lane_id >> 2);
  const unsigned j_offset = (lane_id & 0b11) * 2;
  for (unsigned x = 0; x < 8; x++) {
    const unsigned j = j_offset + (x & 0b100) * 2 + (x & 0b1);
    const unsigned i = i_offset + (x & 0b10) * 4;
    const unsigned frag_index_list[1] = {x};
    func(frag_index_list, 1, i, j);
  }
}








__device__ __inline__ void smem_xor_to_reg_A(wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> &frag,
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


__device__ __inline__  void smem_xor_to_reg_B(wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> &frag,
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










__inline__ __device__ void frag_to_mem(const float *frag, float *gmem, const int ld)
{
  const auto func_y = [&](const unsigned* frag_index_list,
        const unsigned fragment_index_count,
        const unsigned i,
        const unsigned j) {
                
        
    #pragma unroll
        for (unsigned f = 0; f < fragment_index_count; f++)
          gmem[i*ld + j] = frag[frag_index_list[f]];
    };

  __syncwarp();
  wmma_foreach_ij_acc(func_y);
  __syncwarp();
}

__inline__ __device__ void frag_to_mem_ec(const float *frag, const float *delta_frag, float *gmem, const int ld)
{
  const auto func_y = [&](const unsigned* frag_index_list,
        const unsigned fragment_index_count,
        const unsigned i,
        const unsigned j) {
                
        
    #pragma unroll
        for (unsigned f = 0; f < fragment_index_count; f++)
          gmem[i*ld + j] = frag[frag_index_list[f]] + delta_frag[frag_index_list[f]]/2048;
    };

  __syncwarp();
  wmma_foreach_ij_acc(func_y);
  __syncwarp();
}


