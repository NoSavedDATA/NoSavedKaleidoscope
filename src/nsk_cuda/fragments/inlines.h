#pragma once

#include <mma.h>

#include "../smem/include.h"
#include "include.h"


using namespace nvcuda;


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


__inline__ __device__ void frag_to_mem(const float *frag, float *gmem, const int ld);


__inline__ __device__ void frag_to_mem_ec(const float *frag, const float *delta_frag, float *gmem, const int ld);





