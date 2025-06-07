#pragma once




__inline__ __device__ void wmma16x16x16_i8(int *O,
                                        wmma::fragment<wmma::matrix_a, 16, 16, 16, int8_t, wmma::row_major> &x_frag,
                                        wmma::fragment<wmma::matrix_b, 16, 16, 16, int8_t, wmma::col_major> &w_frag)
{                       
  const int *X = reinterpret_cast<const int *>(&x_frag.x);
  const int *W = reinterpret_cast<const int *>(&w_frag.x);





   // 1 lane handles 16 columns

   asm volatile(
      "wmma.mma.sync.aligned.m16n16k16.row.col.s32.s8.s8.s32 "
      "{%0,%1,%2,%3,%4,%5,%6,%7}, "    // D matrix
      "{%8,%9}, "                     // A matrix
      "{%10,%11}, "                   // B matrix
      "{%12,%13,%14,%15,%16,%17,%18,%19};\n"
      : "=r"(O[0]), "=r"(O[1]), "=r"(O[2]), "=r"(O[3]), "=r"(O[4]), "=r"(O[5]), "=r"(O[6]), "=r"(O[7])
      : "r"(X[0]), "r"(X[1]), 
         "r"(W[0]), "r"(W[1]),
         "r"(O[0]),  "r"(O[1]),  "r"(O[2]),  "r"(O[3]),  "r"(O[4]),  "r"(O[5]),  "r"(O[6]),  "r"(O[7]));




//  const int8_t *_X = reinterpret_cast<const int8_t *>(&x_frag.x);
//  const int8_t *_W = reinterpret_cast<const int8_t *>(&w_frag.x);

//  if(threadIdx.x==0&&blockIdx.x==0&&blockIdx.y==0)
//    {
//       printf("\nX: %d, %d, %d, %d\n", (int)_X[0], (int)_X[1], (int)_X[2], (int)_X[3]);
//       printf("X: %d, %d, %d, %d\n", (int)_X[4], (int)_X[5], (int)_X[6], (int)_X[7]);
//       printf("\nW: %d, %d, %d, %d\n", (int)_W[0], (int)_W[1], (int)_W[2], (int)_W[3]);
//       printf("W: %d, %d, %d, %d\n\n", (int)_W[4], (int)_W[5], (int)_W[6], (int)_W[7]);
//       for(int i=0; i<8; ++i)
//       {
//          printf("%d, ", O[i]);
//       }
//       printf("\n------------------------------\n");
//    }
//   __syncwarp();

//  if(threadIdx.x==1&&blockIdx.x==0&&blockIdx.y==0)
//    {
//       printf("\nX: %d - %d - %d - %d\n", (int)_X[0], (int)_X[1], (int)_X[2], (int)_X[3]);
//       printf("X: %d - %d - %d - %d\n", (int)_X[4], (int)_X[5], (int)_X[6], (int)_X[7]);
//       printf("\nW: %d - %d - %d - %d\n", (int)_W[0], (int)_W[1], (int)_W[2], (int)_W[3]);
//       printf("W: %d - %d - %d - %d\n\n", (int)_W[4], (int)_W[5], (int)_W[6], (int)_W[7]);
//       for(int i=0; i<8; ++i)
//       {
//          printf("%d, ", O[i]);
//       }
//       printf("\n==============================\n");
//    }
//   __syncwarp();

//    if(threadIdx.x==2&&blockIdx.x==0&&blockIdx.y==0)
//    {
//       printf("\nX: %d - %d - %d - %d\n", (int)_X[0], (int)_X[1], (int)_X[2], (int)_X[3]);
//       printf("X: %d - %d - %d - %d\n", (int)_X[4], (int)_X[5], (int)_X[6], (int)_X[7]);
//       printf("\nW: %d - %d - %d - %d\n", (int)_W[0], (int)_W[1], (int)_W[2], (int)_W[3]);
//       printf("W: %d - %d - %d - %d\n\n", (int)_W[4], (int)_W[5], (int)_W[6], (int)_W[7]);
//       for(int i=0; i<8; ++i)
//       {
//          printf("%d, ", O[i]);
//       }
//       printf("\n==============================\n");
//    }

//   __syncwarp();

//    if(threadIdx.x==3&&blockIdx.x==0&&blockIdx.y==0)
//    {
//       printf("\nX: %d - %d - %d - %d\n", (int)_X[0], (int)_X[1], (int)_X[2], (int)_X[3]);
//       printf("X: %d - %d - %d - %d\n", (int)_X[4], (int)_X[5], (int)_X[6], (int)_X[7]);
//       printf("\nW: %d - %d - %d - %d\n", (int)_W[0], (int)_W[1], (int)_W[2], (int)_W[3]);
//       printf("W: %d - %d - %d - %d\n\n", (int)_W[4], (int)_W[5], (int)_W[6], (int)_W[7]);
//       for(int i=0; i<8; ++i)
//       {
//          printf("%d, ", O[i]);
//       }
//       printf("\n==============================\n");
//    }
//   __syncwarp();
                  


}