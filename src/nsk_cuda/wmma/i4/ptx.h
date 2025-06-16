#pragma once



#define LDMATRIX_X2(R0, R1, addr) \
    asm volatile("ldmatrix.sync.aligned.x2.m8n8.shared.b16 {%0, %1}, [%2];\n" : "=r"(R0), "=r"(R1) : "r"(addr))




#define I4_MMA(O0, O1, X0, W0)                                  \
   asm volatile(                                                \
      "wmma.mma.sync.aligned.m8n8k32.row.col.s32.s4.s4.s32 "    \
      "{%0,%1}, "                                               \
      "{%2}, "                                                  \
      "{%3}, "                                                  \
      "{%4,%5};\n"                                              \
      : "=r"(O0), "=r"(O1)                                      \
      :  "r"(X0),                                               \
         "r"(W0),                                               \
         "r"(O0),  "r"(O1));