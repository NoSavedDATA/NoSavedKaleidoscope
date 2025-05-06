#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <mma.h>


#include "../tensor/include.h"

extern "C" void *rl_discounted_return(int thread_id, data_type_tensor *reward, data_type_tensor *terminated, float gamma);
