
#include <cuda_runtime.h>
#include <cublasLt.h>
#include <cublas_v2.h>
#include <curand_kernel.h>
#include <cuda_fp16.h>
#include <cudnn.h>
#include <mma.h>

#include <string>
#include <vector>


#include "../../../common/cu_commons.h"
#include "../../../cuda_kernels/handles.h"
#include "../../../tensor/include.h"
#include "../globals.h"
#include "class.h"




Relu::Relu(std::string Name)
    : Name(Name) {}


void Relu::SetDescriptors(int C, int H, int W, int B, Tensor *tensor)
{
  this->C = C;
  this->H = H;
  this->W = W;
  this->B = B;

  
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
  }

  
  cudnnCreateActivationDescriptor(&activation_desc);
  cudnnSetActivationDescriptor(activation_desc, CUDNN_ACTIVATION_RELU, CUDNN_NOT_PROPAGATE_NAN, 0.0);

  checkCUDNN(cudnnCreateTensorDescriptor(&output_desc));
  checkCUDNN(cudnnSetTensor4dDescriptor(output_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, B, C, H, W));
}


float *Relu::Forward(Tensor *tensor, int H, int W, int B, int C)
{

  if (H != this->H || W != this->W || B != this->B)
    this->SetDescriptors(C, H, W, B, tensor);



  float *output = get_from_pool(0, B * H * W * C, "Relu");
  //set_to_one_kernel<<<grid_size, block_size>>>(output, B * H * W * C);
  
  
  constexpr float one = 1.0f;
  constexpr float zero = 0.0f;
  



  checkCUDNN(cudnnActivationForward(
    cudnn,
    activation_desc,
    &one,
    input_desc,
    tensor->tensor_ptr,
    &zero,
    output_desc,
    output
  ));
  
  return output;
}

void Relu::Backward(float *tensor, float *out, float *dx, float *dy)
{
  constexpr float one = 1.0f;
  constexpr float zero = 0.0f;
  float eps = 0.00001f;
  
  

  checkCUDNN(cudnnActivationBackward(
                        cudnn,
                        activation_desc,
                        &one,
                        output_desc,
                        out,
                        output_desc,
                        dy,
                        input_desc,
                        tensor,
                        &zero,
                        input_desc,
                        dx
  ));
}