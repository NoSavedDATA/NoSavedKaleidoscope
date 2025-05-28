#pragma once

#include <mma.h>

using namespace nvcuda;

__device__ __inline__ void gmem_to_smem_xor(const float *gmem_ptr, float &smem, const int trunc)
{
  uint32_t smem_int_ptr = cast_smem_ptr_to_uint(&smem);
  
  asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], %2, %3;"
                :: "r"(smem_int_ptr),
                   "l"(gmem_ptr),
                   "n"(16),
                   "r"(trunc)); // incorrect 0 padding yet
}