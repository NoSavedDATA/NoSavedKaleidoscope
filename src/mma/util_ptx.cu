
#include "util.cu"

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



  
__inline__ __device__ void wmma16x16x16(wmma::fragment<wmma::accumulator, 16, 16, 16, float> &y_frag,
                                        wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> &x_frag,
                                        wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> &w_frag)
{
        const __half *X = reinterpret_cast<const __half *>(&x_frag.x);
        const __half *W = reinterpret_cast<const __half *>(&w_frag.x);
        
        asm volatile("wmma.mma.sync.aligned.m16n16k16.row.col.f32.f32"
                     "{%0,  %1,  %2,  %3,  %4,  %5,  %6,  %7},"
                     "{%8,  %9,  %10, %11, %12, %13, %14, %15},"
                     "{%16, %17, %18, %19, %20, %21, %22, %23},"
                     "{%24, %25, %26, %27, %28, %29, %30, %31};"
                     : "=f"(y_frag.x[0]), "=f"(y_frag.x[1]), "=f"(y_frag.x[2]), "=f"(y_frag.x[3]), "=f"(y_frag.x[4]), "=f"(y_frag.x[5]), "=f"(y_frag.x[6]), "=f"(y_frag.x[7])
                     :  "r"(X[0]),  "r"(X[2]),  "r"(X[4]),  "r"(X[6]),  "r"(X[8]),  "r"(X[10]), "r"(X[12]), "r"(X[14]), \
                        "r"(W[0]),  "r"(W[2]),  "r"(W[4]),  "r"(W[6]),  "r"(W[8]),  "r"(W[10]), "r"(W[12]), "r"(W[14]), \
                       "f"(y_frag.x[0]), "f"(y_frag.x[1]), "f"(y_frag.x[2]), "f"(y_frag.x[3]), "f"(y_frag.x[4]), "f"(y_frag.x[5]), "f"(y_frag.x[6]), "f"(y_frag.x[7]));
        
        asm volatile("wmma.mma.sync.aligned.m16n16k16.row.col.f32.f32"
                     "{%0,  %1,  %2,  %3,  %4,  %5,  %6,  %7},"
                     "{%8,  %9,  %10, %11, %12, %13, %14, %15},"
                     "{%16, %17, %18, %19, %20, %21, %22, %23},"
                     "{%24, %25, %26, %27, %28, %29, %30, %31};"
                     : "=f"(y_frag.x[0]), "=f"(y_frag.x[1]), "=f"(y_frag.x[2]), "=f"(y_frag.x[3]), "=f"(y_frag.x[4]), "=f"(y_frag.x[5]), "=f"(y_frag.x[6]), "=f"(y_frag.x[7])
                     :  "r"(X[1]),  "r"(X[3]),  "r"(X[5]),  "r"(X[7]),  "r"(X[9]),  "r"(X[11]), "r"(X[13]), "r"(X[15]), \
                        "r"(W[1]),  "r"(W[3]),  "r"(W[5]),  "r"(W[7]),  "r"(W[9]),  "r"(W[11]), "r"(W[13]), "r"(W[15]), \
                       "f"(y_frag.x[0]), "f"(y_frag.x[1]), "f"(y_frag.x[2]), "f"(y_frag.x[3]), "f"(y_frag.x[4]), "f"(y_frag.x[5]), "f"(y_frag.x[6]), "f"(y_frag.x[7]));
}




__inline__ __device__ void wmma16x16x16(float *O,
                                        wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> &x_frag,
                                        wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> &w_frag)
{                       
  const __half *X = reinterpret_cast<const __half *>(&x_frag.x);
  const __half *W = reinterpret_cast<const __half *>(&w_frag.x);

  asm volatile("wmma.mma.sync.aligned.m16n16k16.row.col.f32.f32"
               "{%0,  %1,  %2,  %3,  %4,  %5,  %6,  %7},"
               "{%8,  %9,  %10, %11, %12, %13, %14, %15},"
               "{%16, %17, %18, %19, %20, %21, %22, %23},"
               "{%24, %25, %26, %27, %28, %29, %30, %31};"
               : "=f"(O[0]), "=f"(O[1]), "=f"(O[2]), "=f"(O[3]), "=f"(O[4]), "=f"(O[5]), "=f"(O[6]), "=f"(O[7])
               :  "r"(X[0]),  "r"(X[2]),  "r"(X[4]),  "r"(X[6]),  "r"(X[8]),  "r"(X[10]), "r"(X[12]), "r"(X[14]), \
                  "r"(W[0]),  "r"(W[2]),  "r"(W[4]),  "r"(W[6]),  "r"(W[8]),  "r"(W[10]), "r"(W[12]), "r"(W[14]), \
                  "f"(O[0]),  "f"(O[1]),  "f"(O[2]),  "f"(O[3]),  "f"(O[4]),  "f"(O[5]),  "f"(O[6]),  "f"(O[7]));

                  
  asm volatile("wmma.mma.sync.aligned.m16n16k16.row.col.f32.f32"
               "{%0,  %1,  %2,  %3,  %4,  %5,  %6,  %7},"
               "{%8,  %9,  %10, %11, %12, %13, %14, %15},"
               "{%16, %17, %18, %19, %20, %21, %22, %23},"
               "{%24, %25, %26, %27, %28, %29, %30, %31};"
               : "=f"(O[0]), "=f"(O[1]), "=f"(O[2]), "=f"(O[3]), "=f"(O[4]), "=f"(O[5]), "=f"(O[6]), "=f"(O[7])
               :  "r"(X[1]),  "r"(X[3]),  "r"(X[5]),  "r"(X[7]),  "r"(X[9]),  "r"(X[11]), "r"(X[13]), "r"(X[15]), \
                  "r"(W[1]),  "r"(W[3]),  "r"(W[5]),  "r"(W[7]),  "r"(W[9]),  "r"(W[11]), "r"(W[13]), "r"(W[15]), \
                  "f"(O[0]),  "f"(O[1]),  "f"(O[2]),  "f"(O[3]),  "f"(O[4]),  "f"(O[5]),  "f"(O[6]),  "f"(O[7]));


  // asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
  //       "{%0, %1, %2, %3},"
  //       "{%4, %5, %6, %7},"
  //       "{%8, %9},"
  //       "{%10, %11, %12, %13};"
  //       : "=f"(O[0]), "=f"(O[1]), "=f"(O[2]), "=f"(O[3])
  //       :  "r"(X[0]),  "r"(X[2]),  "r"(X[4]),  "r"(X[6]), \
  //          "r"(W[0]),  "r"(W[2]),                         \
  //          "f"(O[0]),  "f"(O[1]),  "f"(O[2]),  "f"(O[3]));
  // __syncwarp();

  // asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
  //       "{%0, %1, %2, %3},"
  //       "{%4, %5, %6, %7},"
  //       "{%8, %9},"
  //       "{%10, %11, %12, %13};"
  //       : "=f"(O[0]), "=f"(O[1]), "=f"(O[2]), "=f"(O[3])
  //       :  "r"(X[0]),  "r"(X[2]),  "r"(X[4]),  "r"(X[6]), \
  //          "r"(W[4]),  "r"(W[6]),                         \
  //          "f"(O[0]),  "f"(O[1]),  "f"(O[2]),  "f"(O[3]));
  // __syncwarp();

  // asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
  //       "{%0, %1, %2, %3},"
  //       "{%4, %5, %6, %7},"
  //       "{%8, %9},"
  //       "{%10, %11, %12, %13};"
  //       : "=f"(O[4]), "=f"(O[5]), "=f"(O[6]), "=f"(O[7])
  //       :  "r"(X[8]),  "r"(X[10]),  "r"(X[12]),  "r"(X[14]), \
  //          "r"(W[8]),  "r"(W[10]),                         \
  //          "f"(O[4]),  "f"(O[5]),  "f"(O[6]),  "f"(O[7]));
  // __syncwarp();

  // asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
  //       "{%0, %1, %2, %3},"
  //       "{%4, %5, %6, %7},"
  //       "{%8, %9},"
  //       "{%10, %11, %12, %13};"
  //       : "=f"(O[4]), "=f"(O[5]), "=f"(O[6]), "=f"(O[7])
  //       :  "r"(X[8]),  "r"(X[10]),  "r"(X[12]),  "r"(X[14]), \
  //          "r"(W[12]),  "r"(W[14]),                         \
  //          "f"(O[4]),  "f"(O[5]),  "f"(O[6]),  "f"(O[7]));


  // __syncwarp();

  // asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
  //       "{%0, %1, %2, %3},"
  //       "{%4, %5, %6, %7},"
  //       "{%8, %9},"
  //       "{%10, %11, %12, %13};"
  //       : "=f"(O[0]), "=f"(O[1]), "=f"(O[2]), "=f"(O[3])
  //       :  "r"(X[1]),  "r"(X[3]),  "r"(X[5]),  "r"(X[7]), \
  //          "r"(W[1]),  "r"(W[3]),                         \
  //          "f"(O[0]),  "f"(O[1]),  "f"(O[2]),  "f"(O[3]));
  // __syncwarp();

  // asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
  //       "{%0, %1, %2, %3},"
  //       "{%4, %5, %6, %7},"
  //       "{%8, %9},"
  //       "{%10, %11, %12, %13};"
  //       : "=f"(O[0]), "=f"(O[1]), "=f"(O[2]), "=f"(O[3])
  //       :  "r"(X[1]),  "r"(X[3]),  "r"(X[5]),  "r"(X[7]), \
  //          "r"(W[5]),  "r"(W[7]),                         \
  //          "f"(O[0]),  "f"(O[1]),  "f"(O[2]),  "f"(O[3]));
  // __syncwarp();

  // asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
  //       "{%0, %1, %2, %3},"
  //       "{%4, %5, %6, %7},"
  //       "{%8, %9},"
  //       "{%10, %11, %12, %13};"
  //       : "=f"(O[4]), "=f"(O[5]), "=f"(O[6]), "=f"(O[7])
  //       :  "r"(X[9]),  "r"(X[11]),  "r"(X[13]),  "r"(X[15]), \
  //          "r"(W[9]),  "r"(W[11]),                         \
  //          "f"(O[4]),  "f"(O[5]),  "f"(O[6]),  "f"(O[7]));
  // __syncwarp();

  // asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
  //       "{%0, %1, %2, %3},"
  //       "{%4, %5, %6, %7},"
  //       "{%8, %9},"
  //       "{%10, %11, %12, %13};"
  //       : "=f"(O[4]), "=f"(O[5]), "=f"(O[6]), "=f"(O[7])
  //       :  "r"(X[9]),  "r"(X[11]),  "r"(X[13]),  "r"(X[15]), \
  //          "r"(W[13]),  "r"(W[15]),                         \
  //          "f"(O[4]),  "f"(O[5]),  "f"(O[6]),  "f"(O[7]));
  // __syncwarp();
}
