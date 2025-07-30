#include <cuda_runtime.h>

#include "../backprop/backprop.h"
#include "../cuda_kernels/include.h"
#include "../cuda_threads/include.h"
#include "../common/include.h"
#include "../tensor/tensor_struct.h"


extern "C" void initialize__nsk_cuda() {
  for (int i=0;i<10;++i)
  {
    cudaStream_t thread_stream = createCudaStream();
    ThreadsStream[i] = thread_stream;
  }
  
  int deviceIdx = 0;
  cudaCheck(cudaSetDevice(deviceIdx));
  cudaGetDeviceProperties(&deviceProp, deviceIdx);

  std::cout << "\n\n";
  std::cout << "CuDNN Version: " << CUDNN_MAJOR << "." << CUDNN_MINOR << "." << CUDNN_PATCHLEVEL << std::endl;
  printf("Device %d: %s\n", deviceIdx, deviceProp.name);
  std::cout << "Device Max Compute Capability (SM): " << deviceProp.major << "." << deviceProp.minor << std::endl;

  std::cout << "Shared-Memory per thread-block size: " << deviceProp.sharedMemPerBlock << ".\n";
  

    
  cudaDeviceGetAttribute(&WARP_SIZE, cudaDevAttrWarpSize, 0); 
  cublasCheck(cublasCreate(&cublas_handle));
  cublasCheck(cublasLtCreate(&cublaslt_handle));


  int enable_tf32 = deviceProp.major >= 8 ? 1 : 0;


  printf("enable_tf32: %d\n", enable_tf32);
  
  cublas_compute_type = enable_tf32 ? CUBLAS_COMPUTE_32F_FAST_TF32 : CUBLAS_COMPUTE_32F;
  cublasMath_t cublas_math_mode = enable_tf32 ? CUBLAS_TF32_TENSOR_OP_MATH : CUBLAS_DEFAULT_MATH;
  cublasCheck(cublasSetMathMode(cublas_handle, cublas_math_mode));
  // setup the (global) cuBLASLt workspace
  cudaCheck(cudaMalloc(&cublaslt_workspace, cublaslt_workspace_size));
  
  cudnnCreate(&cudnn);

  std::cout << "Tile size is: " << TILE_SIZE << ".\n\n";
  main_stream = createCudaStream();


  tensor_inits = {"binary", "arange", "ints", "randu", "zeros", "ones", "xavu", "xavu_relu", "xavu_tanh", "he_normal_relu", "init_gpt", "xavn", "normal", "fixed8i", "fixed42i"};


  leaf_ops = {leaf, tensor_leaf, weight_leaf, bias_leaf};
  activation_ops = {relu_op, gelu_op, softmax_op, tanh_op, sigmoid_op, cudnn_relu_op};
  loss_ops = {cross_entropy_op, cross_entropy_idx_op, mse_op, mse_is_w_op};

  custom_ops = {sigmoid_add2weights_op, embedding_op};

  tensor_scalar_ops = {scalar_add_op, scalar_sub_op, scalar_mult_op, scalar_div_op};

  weightless_ops = {add_op, lgrad_op, dropout_op};
  weightless_ops = concat_int_vec(weightless_ops, tensor_scalar_ops);

  preprocessing_ops = {gpu_op, crop_op, random_horizontal_flip_op, normalize_img_op, jitter_op};
  gradless_ops = {randu_like_op, onehot_op, max_op, argmax_op, equal_op,
                  create_tensor_from_brackets_op, detach_op};
  gradless_ops = concat_int_vec(gradless_ops, preprocessing_ops);




  backward_functions["scalarmult_backward"] = scalarmult_backward;
	backward_functions["relu_backward"] = relu_backward;
	backward_functions["gelu_backward"] = gelu_backward;
	backward_functions["sigmoid_backward"] = sigmoid_backward;
	backward_functions["tanh_backward"] = tanh_backward;
	backward_functions["mean_over_semilast_dim_backward"] = mean_over_semilast_dim_backward;
	backward_functions["gather_last_dim_backward"] = gather_last_dim_backward;
}