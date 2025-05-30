
#include "util.h"
#include "../nsk_cuda/include.h"

// Cuda
#include <cuda_runtime.h>
#include <cublasLt.h>
#include <curand_kernel.h>
#include <cudnn.h>

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <mma.h>


// asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0, %1}, {%2, %3, %4, %5}, {%6, %7}, {%8, %9};\n" \
//             : "=r"(RD0), "=r"(RD1)                                                                                \
//             : "r"(RA0), "r"(RA1), "r"(RA2), "r"(RA3), "r"(RB0), "r"(RB1), "r"(RC0), "r"(RC1))


// using namespace nvcuda;

// __device__ __inline__ void gmem_to_smem_xor(const float *gmem_ptr, float &smem, const int trunc)
// {
//   uint32_t smem_int_ptr = cast_smem_ptr_to_uint(&smem);
  
//   asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], %2, %3;"
//                 :: "r"(smem_int_ptr),
//                    "l"(gmem_ptr),
//                    "n"(16),
//                    "r"(trunc)); // incorrect 0 padding yet
// }





__inline__ __device__ void ld_smem_to_reg_A(__half *frag, const float *smem)
{

    asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];"
                 : "=r"(frag[0]), "=r"(frag[1]), "=r"(frag[2]), "=r"(frag[3])
                 : "r"(smem));
}




__inline__ __device__ void ld_smem_to_reg_B(__half *frag, const float *smem)
{

    asm volatile("ldmatrix.sync.aligned.x2.m8n8.shared.b16 {%0, %1}, [%2];"
                 : "=r"(frag[0]), "=r"(frag[1])
                 : "r"(smem));
}



  
