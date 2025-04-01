
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
#include "class.h"


MaxPool2d::MaxPool2d(int ks, int stride, int padding, std::string Type)
    : ks(ks), stride(stride), padding(padding), Type(Type) {}



void MaxPool2d::SetDescriptors(int H, int W, int B, int C, Tensor *tensor)
{
  this->H = H;
  this->W = W;
  this->B = B;


  out_H = std::floor((H - ks + 2 * padding) / stride) + 1;
  out_W = std::floor((W - ks + 2 * padding) / stride) + 1;

  this->out_H=out_H;
  this->out_W=out_W;
  //std::cout << "Out H: " << out_H << " out W: " << out_W << "\n";

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
  


  // Initialize pooling descriptor
  checkCUDNN(cudnnCreatePoolingDescriptor(&pooling_desc));
  checkCUDNN(cudnnSetPooling2dDescriptor(pooling_desc,            //descriptor handle
                                         (Type=="max") ? CUDNN_POOLING_MAX : CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING,       //mode - max pooling
                                         CUDNN_NOT_PROPAGATE_NAN, //NaN propagation mode
                                         ks,                       //window height
                                         ks,                       //window width
                                         padding,                       //vertical padding
                                         padding,                       //horizontal padding
                                         stride,                       //vertical stride
                                         stride));                     //horizontal stride
  
  // Initialize output tensor descriptor
  checkCUDNN(cudnnCreateTensorDescriptor(&output_desc));
  checkCUDNN(cudnnSetTensor4dDescriptor(output_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, B, C, out_H, out_W));
}



float *MaxPool2d::Forward(Tensor *tensor, int H, int W, int B, int C, int thread_id)
{
  // Initialize descriptors.
  //std::cout << "\nPool2d Forward with H: " << H << " W: " << W << "\n";
  //std::cout << "Type: " << Type << "\n";


  if (H != this->H || W != this->W || B != this->B || C != this->C)
    this->SetDescriptors(H, W, B, C, tensor);


  
  // Forward
  float *d_output = get_from_pool(thread_id, B * out_H * out_W * C, "maxpool2d");
  

  constexpr float one = 1.0f;
  constexpr float zero = 0.0f;

  checkCUDNN(cudnnPoolingForward(
        cudnn,
        pooling_desc,
        &one,
        input_desc,
        tensor->tensor_ptr,
        &zero,
        output_desc,
        d_output
    ));
  

  return d_output;
}


void MaxPool2d::Backward(float *tensor, float *out, float *dx, float *dy)
{
  //std::cout << "\nMaxPool2d Backward with H: " << H << " W: " << W << "\n";


  constexpr float one = 1.0f;
  constexpr float zero = 0.0f;


  // Backward to input
  checkCUDNN(cudnnPoolingBackward(
    cudnn,
    pooling_desc,
    &one,
    output_desc,
    out,
    output_desc, // output grad tensor descriptor
    dy,
    input_desc,
    tensor,
    &zero,
    input_desc, // input descriptor
    dx
  ));
  
}