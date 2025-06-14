#pragma once


#define LDMATRIX_X2(R0, R1, addr) \
    asm volatile("ldmatrix.sync.aligned.x2.m8n8.shared.b16 {%0, %1}, [%2];\n" : "=r"(R0), "=r"(R1) : "r"(addr))


#define I8_MMA(O0, O1, O2, O3, X0, X1, W0)                      \
   asm volatile(                                                \
      "mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 "        \
      "{%0,%1,%2,%3}, "                                         \
      "{%4,%5}, "                                               \
      "{%6}, "                                                  \
      "{%7, %8, %9, %10};\n"                                    \
      : "=r"(O0), "=r"(O1), "=r"(O2) , "=r"(O3)                 \
      : "r"(X0), "r"(X1),                                       \
         "r"(W0),                                               \ 
         "r"(O0),  "r"(O1),  "r"(O2),  "r"(O3));




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


}