#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <mma.h>


#include "../tensor/include.h"

extern "C" void *rl_discounted_return(int thread_id, Tensor *reward, Tensor *terminated, float gamma);
