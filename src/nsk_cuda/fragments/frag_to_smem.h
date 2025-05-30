#pragma once

#include <mma.h>
#include "inlines.h"

using namespace nvcuda;



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