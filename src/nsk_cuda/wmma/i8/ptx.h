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

