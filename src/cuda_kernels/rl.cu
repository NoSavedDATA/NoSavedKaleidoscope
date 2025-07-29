#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <cstdarg>
#include <iostream>
#include <vector>


#include "../backprop/include.h"
#include "../common/include.h"
#include "../compiler_frontend/logging.h"
#include "../nsk_cuda/pool/include.h"
#include "../tensor/include.h"
#include "calculate_grids.h"
#include "rl_kernels.h"
#include "handles.h"



extern "C" DT_tensor *rl_discounted_return(Scope_Struct *scope_struct, DT_tensor *reward, DT_tensor *terminated, float gamma)
{
  //std::cout << "rl_discounted_return THREAD IS: " << thread_id << "\n";

  int thread_id = scope_struct->thread_id;
  std::vector<int> dims = reward->dims;

  if (reward->dims.size()!=2||terminated->dims.size()!=2)
    LogErrorS(scope_struct->code_line, "rl_discounted_return requires dims [B, T]");

  int B = dims[0];
  int T = dims[1];
  

  int grid_size, block_size; 
  CalculateGridAndBlockSizes(B, grid_size, block_size);
  

  float *G = get_from_pool(thread_id, B, "rl_discounted_return");

  reward->Sync();
  terminated->Sync();

  cudaStream_t stream = ThreadsStream[thread_id];
  rl_discounted_return_kernel<<<grid_size, block_size, 0, stream>>>(G, reward->tensor_ptr, terminated->tensor_ptr, T, gamma, B);



  DT_tensor *new_tensor = createTensor(G, {B}, B, false, "");
  new_tensor->AttrNodes(reward, terminated, detach_op);
  return new_tensor;
}