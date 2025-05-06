
#include "../../tensor/include.h"
#include "../handles.h"

#include "kernel.h"



inline data_type_tensor *float_to_half(data_type_tensor *tensor, int thread_id, cudaStream_t stream)
{

  half *tensor_ptr = get_half_from_pool(thread_id, tensor->dims_prod, "float to half");


  data_type_tensor *half_tensor = createTensorHalf(tensor_ptr, tensor->dims, tensor->dims_prod, true, tensor->name+"half");

  float_to_half_kernel<<<std::ceil(tensor->dims_prod/(float)THREADS_PER_BLOCK), THREADS_PER_BLOCK, 0, stream>>>(tensor->tensor_ptr, tensor_ptr, tensor->dims_prod);

  return half_tensor;
}
inline half *float_to_half(float *tensor, int thread_id, int dims_prod, cudaStream_t stream)
{
  half *tensor_ptr = get_half_from_pool(thread_id, dims_prod, "float to half");


  float_to_half_kernel<<<std::ceil(dims_prod/(float)THREADS_PER_BLOCK), THREADS_PER_BLOCK, 0, stream>>>(tensor, tensor_ptr, dims_prod);

  return tensor_ptr;
}


inline data_type_tensor *half_to_float(data_type_tensor *tensor, int thread_id, cudaStream_t stream)
{

  float *tensor_ptr = get_from_pool(thread_id, tensor->dims_prod, "half to float");
  


  data_type_tensor *float_tensor = createTensor(tensor_ptr, tensor->dims, tensor->dims_prod, true, tensor->name+"half");

  half_to_float_kernel<<<std::ceil(tensor->dims_prod/(float)THREADS_PER_BLOCK), THREADS_PER_BLOCK, 0, stream>>>(tensor->half_ptr, tensor_ptr, tensor->dims_prod);

  return float_tensor;
}

inline float *half_to_float(half *tensor, int thread_id, int dims_prod, cudaStream_t stream)
{

  float *tensor_ptr = get_from_pool(thread_id, dims_prod, "half to float");
  

  half_to_float_kernel<<<std::ceil(dims_prod/(float)THREADS_PER_BLOCK), THREADS_PER_BLOCK, 0, stream>>>(tensor, tensor_ptr, dims_prod);

  return tensor_ptr;
}

inline float *half_to_float_overwrite(half *tensor, float *float_tensor, int dims_prod, cudaStream_t stream)
{
  half_to_float_kernel<<<std::ceil(dims_prod/(float)THREADS_PER_BLOCK), THREADS_PER_BLOCK, 0, stream>>>(tensor, float_tensor, dims_prod);

  return float_tensor;
}