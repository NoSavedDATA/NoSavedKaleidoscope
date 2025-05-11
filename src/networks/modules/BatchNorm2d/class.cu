
#include <cuda_runtime.h>
#include <cublasLt.h>
#include <cublas_v2.h>
#include <curand_kernel.h>
#include <cuda_fp16.h>
#include <cudnn.h>
#include <mma.h>

#include <string>
#include <vector>


#include "../../../backprop/include.h"
#include "../../../common/cu_commons.h"
#include "../../../cuda_kernels/calculate_grids.h"
#include "../../../cuda_kernels/elementwise_kernels_inline.cu"
#include "../../../cuda_kernels/handles.h"
#include "../../../tensor/include.h"
#include "class.h"




BatchNorm2dCPP::BatchNorm2dCPP(int C, std::string Name)
    : C(C), Name(Name) {
  // NamedTensorsT[Name+"W"] = new DT_tensor();
  // NamedTensorsT[Name+"B"] = new DT_tensor();
}



void BatchNorm2dCPP::SetDescriptors(int H, int W, int B, DT_tensor *tensor)
{
  this->H = H;
  this->W = W;
  this->B = B;

  /*
  switch(tensor->op)
  {
    case conv2d:
      input_desc = NamedConv2d[tensor->from_cudnn]->output_desc;
      break;
    case bn2drelu:
      input_desc = NamedBN2dRelu[tensor->from_cudnn]->output_desc;
      break;
    case cudnn_relu_op:
      input_desc = NamedRelu[tensor->from_cudnn]->output_desc;
      break;
    case batchnorm2d:
      input_desc = NamedBatchNorm2d[tensor->from_cudnn]->output_desc;
      break;
    case maxpool2d:
      input_desc = NamedMaxPool2d[tensor->from_cudnn]->output_desc;
      break;
    default:
      checkCUDNN(cudnnCreateTensorDescriptor(&input_desc));
      checkCUDNN(cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, B, C, H, W));
      break;
  }*/
  checkCUDNN(cudnnCreateTensorDescriptor(&input_desc));
  checkCUDNN(cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, B, C, H, W));
  
  
  checkCUDNN(cudnnCreateTensorDescriptor(&output_desc));
  checkCUDNN(cudnnSetTensor4dDescriptor(output_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, B, C, H, W));
  
  checkCUDNN(cudnnCreateTensorDescriptor(&scale_bias_mean_var_desc));
  //checkCUDNN(cudnnSetTensor4dDescriptor(scale_bias_mean_var_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, C, 1, 1));
  checkCUDNN(cudnnDeriveBNTensorDescriptor(scale_bias_mean_var_desc, input_desc, CUDNN_BATCHNORM_SPATIAL_PERSISTENT));
}

void BatchNorm2dCPP::InitMovingAverages()
{
  float *aux;

  aux = make_ones_float(C);
  cudaCheck(cudaMalloc(&scale, C*sizeof(float)));
  cudaCheck(cudaMemcpy(scale, aux, C*sizeof(float), cudaMemcpyHostToDevice));
  delete[] aux;
  
  aux = make_zeros_float(C);
  cudaCheck(cudaMalloc(&bias, C*sizeof(float)));
  cudaCheck(cudaMemcpy(bias, aux, C*sizeof(float), cudaMemcpyHostToDevice));
  delete[] aux;
  

  aux = make_zeros_float(C);
  cudaCheck(cudaMalloc(&running_mean, C*sizeof(float)));
  cudaCheck(cudaMemcpy(running_mean, aux, C*sizeof(float), cudaMemcpyHostToDevice));
  delete[] aux;

  aux = make_zeros_float(C);
  cudaCheck(cudaMalloc(&saved_mean, C*sizeof(float)));
  cudaCheck(cudaMemcpy(saved_mean, aux, C*sizeof(float), cudaMemcpyHostToDevice));
  delete[] aux;
  
  
  aux = make_ones_float(C);
  cudaCheck(cudaMalloc(&running_var, C*sizeof(float)));
  cudaCheck(cudaMemcpy(running_var, aux, C*sizeof(float), cudaMemcpyHostToDevice));
  delete[] aux;

  aux = make_ones_float(C);
  cudaCheck(cudaMalloc(&saved_var, C*sizeof(float)));
  cudaCheck(cudaMemcpy(saved_var, aux, C*sizeof(float), cudaMemcpyHostToDevice));
  delete[] aux;


  DT_tensor *scale_tensor, *bias_tensor;
  scale_tensor = new DT_tensor();
  scale_tensor->NewTensor(scale, {(float)C}, C, true, Name);

  bias_tensor = new DT_tensor();
  bias_tensor->NewTensor(bias, {(float)C}, C, true, Name);

  NamedTensorsT[Name+"W"] = scale_tensor;
  NamedTensorsT[Name+"B"] = bias_tensor;
}

float *BatchNorm2dCPP::Forward(DT_tensor *tensor, int H, int W, int B, int C, int thread_id)
{

  if (H != this->H || W != this->W || B != this->B)
    this->SetDescriptors(H, W, B, tensor);

  // Initialize weights.
  if (scale==nullptr)
    this->InitMovingAverages();


  // Forward
  int grid_size, block_size, shared_mem_size; 
  std::vector<int> grid_block_mem_sizes = CalculateGridAndBlockSizes(B*C);
  grid_size = grid_block_mem_sizes[0];
  block_size = grid_block_mem_sizes[1];
  
  

  float *output = get_from_pool(thread_id, B * H * W * C, "batchnorm2d");
  //set_to_one_kernel<<<grid_size, block_size>>>(output, B * H * W * C);
  
  
  constexpr float one = 1.0f;
  constexpr float zero = 0.0f;
  float gamma = 0.9f;
  float eps = 0.00001f;

  

  if(nn_mode==training_mode)
  {
    checkCUDNN(cudnnBatchNormalizationForwardTraining(
      cudnn,
      CUDNN_BATCHNORM_SPATIAL_PERSISTENT,
      &one,
      &zero,
      input_desc,
      tensor->tensor_ptr,
      output_desc,
      output,
      scale_bias_mean_var_desc,
      scale,
      bias,
      gamma,
      running_mean,
      running_var,
      eps,
      saved_mean,
      saved_var
    ));
  }
  else
  {
    checkCUDNN(cudnnDeriveBNTensorDescriptor(scale_bias_mean_var_desc, input_desc, CUDNN_BATCHNORM_SPATIAL));
    checkCUDNN(cudnnBatchNormalizationForwardInference(
      cudnn,
      CUDNN_BATCHNORM_SPATIAL,
      &one,
      &zero,
      input_desc,
      tensor->tensor_ptr,
      output_desc,
      output,
      scale_bias_mean_var_desc,
      scale,
      bias,
      running_mean,
      running_var,
      eps
    ));
  }
  
  return output;
}



void BatchNorm2dCPP::FirstBackward() {
  if (first_backward) {

    dW = get_from_pool(0, C, "BatchNorm2d dW");
    dB = get_from_pool(0, C, "BatchNorm2d dB");

    
    set_to_zero_kernel<<<std::ceil(C/(float)TILE_SIZE_SQ), TILE_SIZE_SQ, 0, main_stream>>>(dW, C);
    set_to_zero_kernel<<<std::ceil(C/(float)TILE_SIZE_SQ), TILE_SIZE_SQ, 0, main_stream>>>(dB, C);

    NamedParamGrads[Name+"W"] = dW;
    NamedParamGrads[Name+"B"] = dB;

    first_backward=false;
  }
}




void BatchNorm2dCPP::Backward(float *tensor, float *dx, float *dy)
{
  constexpr float one = 1.0f;
  constexpr float zero = 0.0f;
  float eps = 0.00001f;
  
  FirstBackward();
  

  checkCUDNN(cudnnBatchNormalizationBackward(
    cudnn,
    CUDNN_BATCHNORM_SPATIAL_PERSISTENT,
    &one,
    &zero,
    &one,
    &one,
    input_desc,
    tensor,
    output_desc,
    dy,
    input_desc,
    dx,
    scale_bias_mean_var_desc,
    scale,
    dW,
    dB,
    eps,
    saved_mean,
    saved_var
  ));
}