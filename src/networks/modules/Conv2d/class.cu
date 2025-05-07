
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
#include "../../../cuda_kernels/elementwise_kernels_inline.cu"
#include "../../../cuda_kernels/handles.h"
#include "../../../tensor/include.h"
#include "class.h"


Conv2dCPP::Conv2dCPP(int C, int OC, int ks, int stride, int padding, std::string Init, std::vector<std::string> Notes, std::string Name)
    : C(C), OC(OC), ks(ks), stride(stride), padding(padding), Init(Init), Notes(Notes), Name(Name) {
    NamedTensorsT[Name+"W"] = new DT_tensor();
    d_filter=nullptr;
    d_workspace=nullptr;
    d_workspace_w_back=nullptr;
    d_workspace_y_back=nullptr;
    workspace_size=0;
    workspace_size_w_back=0;
    workspace_size_y_back=0;
}


void Conv2dCPP::SetDescriptors(int H, int W, int B, DT_tensor *tensor)
{
  this->H = H;
  this->W = W;
  this->B = B;


  //std::cout << "\nConv2d Set Descriptors\nC: " << C << " OC " << OC << " ks " << ks << " stride " << stride << " padding " << padding << " H " << H << " W " << W << "\n";


  out_H = std::floor((H - ks + 2 * padding) / stride) + 1;
  out_W = std::floor((W - ks + 2 * padding) / stride) + 1;
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
      // Initialize input tensor descriptor
      checkCUDNN(cudnnCreateTensorDescriptor(&input_desc));
      checkCUDNN(cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, B, C, H, W));
      break;
  }*/

  checkCUDNN(cudnnCreateTensorDescriptor(&input_desc));
  checkCUDNN(cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, B, C, H, W));
  
  // Initialize filter descriptor
  cudnnFilterDescriptor_t filter_desc;
  checkCUDNN(cudnnCreateFilterDescriptor(&filter_desc));
  checkCUDNN(cudnnSetFilter4dDescriptor(filter_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, OC, C, ks, ks));
  this->filter_desc = filter_desc;

  // Initialize convolution descriptor
  cudnnConvolutionDescriptor_t conv_desc;
  checkCUDNN(cudnnCreateConvolutionDescriptor(&conv_desc));
  checkCUDNN(cudnnSetConvolution2dDescriptor(conv_desc, padding, padding, stride, stride, 1, 1,
                                           CUDNN_CONVOLUTION, CUDNN_DATA_FLOAT));
  this->conv_desc = conv_desc;

  // Initialize output tensor descriptor
  cudnnTensorDescriptor_t output_desc;
  checkCUDNN(cudnnCreateTensorDescriptor(&output_desc));
  checkCUDNN(cudnnSetTensor4dDescriptor(output_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, B, OC, out_H, out_W));
  this->output_desc = output_desc;

  
  int requested_algo_count;
  int algo_count;




  // Forward
  checkCUDNN(cudnnGetConvolutionForwardAlgorithmMaxCount(cudnn, &requested_algo_count));
  std::vector<cudnnConvolutionFwdAlgoPerf_t> perf_results(requested_algo_count);
  checkCUDNN(cudnnFindConvolutionForwardAlgorithm(
        cudnn,
        input_desc,
        filter_desc,
        conv_desc,
        output_desc,
        requested_algo_count,
        &algo_count,
        perf_results.data()
  ));

  this->fwd_algo = perf_results.front().algo;


  
  if (d_workspace!=nullptr)
    move_to_pool_pow2(0, workspace_size, d_workspace, "d workspace");
  checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(
        cudnn,
        input_desc,
        filter_desc,
        conv_desc,
        output_desc,
        fwd_algo,
        &workspace_size
  ));
  d_workspace = get_from_pool_pow2(0, workspace_size, "d workspace");
  
  




  // Backward to input
  checkCUDNN(cudnnGetConvolutionBackwardDataAlgorithmMaxCount(cudnn, &requested_algo_count));
  std::vector<cudnnConvolutionBwdDataAlgoPerf_t> perf_results_back_y(requested_algo_count);
  checkCUDNN(cudnnFindConvolutionBackwardDataAlgorithm(
        cudnn,
        filter_desc,
        output_desc,
        conv_desc,
        input_desc,
        requested_algo_count,
        &algo_count,
        perf_results_back_y.data()
  ));

  y_bwd_algo = perf_results_back_y.front().algo;

  
  if(d_workspace_y_back!=nullptr)
    move_to_pool_pow2(0, workspace_size_y_back, d_workspace_y_back, "d workspace y back");
  checkCUDNN(cudnnGetConvolutionBackwardDataWorkspaceSize(
        cudnn,
        filter_desc,
        output_desc,
        conv_desc,
        input_desc,
        y_bwd_algo,
        &workspace_size_y_back
  ));

  d_workspace_y_back = get_from_pool_pow2(0, workspace_size_y_back, "d workspace y back");
  




  // Backward to weight
  checkCUDNN(cudnnGetConvolutionBackwardFilterAlgorithmMaxCount(cudnn, &requested_algo_count));
  std::vector<cudnnConvolutionBwdFilterAlgoPerf_t> perf_results_back_w(requested_algo_count);
  checkCUDNN(cudnnFindConvolutionBackwardFilterAlgorithm(
        cudnn,
        input_desc,
        output_desc,
        conv_desc,
        filter_desc,
        requested_algo_count,
        &algo_count,
        perf_results_back_w.data()
  ));

  w_bwd_algo = perf_results_back_w.front().algo;

  
  
  if (d_workspace_w_back!=nullptr)
    move_to_pool_pow2(0, workspace_size_w_back, d_workspace_w_back, "conv d workspace w back");
  checkCUDNN(cudnnGetConvolutionBackwardFilterWorkspaceSize(
        cudnn,
        input_desc,
        output_desc,
        conv_desc,
        filter_desc,
        w_bwd_algo,
        &workspace_size_w_back
  ));
  d_workspace_w_back = get_from_pool_pow2(0, workspace_size_w_back, "conv d workspace w back");
  
  
}





void Conv2dCPP::InitFilters()
{
  std::vector<float> h_filter;
  float *filter;
  for (std::size_t idx = 0; idx < C * OC; ++idx) {

    if (Init=="xavu_relu")
      filter = make_xavier_uniform_float_relu(ks*ks, ks*ks*C, ks*ks*OC);
    if (Init == "xavu_tanh")
      filter = make_xavier_uniform_float_tanh(ks*ks, ks*ks*C, ks*ks*OC);
    if (Init=="he_normal_relu")
      filter = make_he_normal_float_relu(ks*ks, ks*ks*C);
    if (Init == "init_gpt")
      filter = make_gpt_init(ks*ks);
    if (Init=="xavu")
      filter = make_xavier_uniform_float(ks*ks, ks*ks*C, ks*ks*OC);
    if (Init=="zeros")
      filter = make_zeros_float(ks*ks);
    if (Init=="ones")
      filter = make_ones_float(ks*ks);
    if (Init=="randu")
      filter = make_random_float_uniform(ks*ks);


    for (int i=0; i < ks*ks; i++)
      h_filter.emplace_back(filter[i]);

    delete[] filter;
    //for (const auto& val : filter) 
    //  h_filter.emplace_back(val);
  }
    
  float* d_filter = nullptr;
  const std::size_t filter_size = h_filter.size();
  cudaCheck(cudaMalloc(&d_filter, filter_size * sizeof(float)));

  cudaCheck(cudaMemcpy(d_filter, h_filter.data(), filter_size * sizeof(float), cudaMemcpyDefault));
  this->d_filter = d_filter;
  

  std::vector<float> kernel_dims = {(float)OC, (float)C, (float)ks, (float)ks}; 

  DT_tensor *tensor_W = createTensor(d_filter, kernel_dims, DimsProd(kernel_dims), true, Name+"W");
  tensor_W->SetIsWeight();

  NamedTensorsT[Name+"W"] = tensor_W;
}




void Conv2dCPP::FirstBackward()
{
  if (first_backward)
  {
    dW = get_from_pool(0, OC*C*ks*ks, "conv2d gradient");
    set_to_zero_kernel<<<std::ceil((OC*C*ks*ks)/(float)TILE_SIZE_SQ), TILE_SIZE_SQ, 0, main_stream->stream>>>(dW, OC*C*ks*ks);

    NamedParamGrads[Name+"W"] = dW;
    first_backward = false;
  }
  
}


float *Conv2dCPP::Forward(DT_tensor *tensor, int H, int W, int B, int thread_id)
{
  // Initialize descriptors.
  //std::cout << "\nConv2d Forward with H: " << H << " W: " << W << "\n";

  if (H != this->H || W != this->W || B != this->B)
    this->SetDescriptors(H, W, B, tensor);

  // Initialize weights.
  if (d_filter==nullptr)
    this->InitFilters();
  

  // Forward
  float *d_output = get_from_pool(thread_id, B * out_H * out_W * OC, "conv2d");




  constexpr float one = 1.0f;
  constexpr float zero = 0.0f;

  
 

  checkCUDNN(cudnnConvolutionForward(
        cudnn,
        &one,
        input_desc,
        tensor->tensor_ptr,
        filter_desc,
        d_filter,
        conv_desc,
        fwd_algo,
        d_workspace,
        workspace_size,
        &zero,
        output_desc,
        d_output
    ));
  



  return d_output;
}


void Conv2dCPP::Backward(float *tensor, float *dx, float *dy)
{
  //std::cout << "\nConv2d Backward with H: " << H << " W: " << W << "\n";


  constexpr float one = 1.0f;
  constexpr float zero = 0.0f;
  
  FirstBackward();

  // Backward to input
  checkCUDNN(cudnnConvolutionBackwardData(
    cudnn,
    &one,
    filter_desc, // input tensor descriptor
    d_filter,
    output_desc, // output grad tensor descriptor
    dy,
    conv_desc, // convolution descriptor
    y_bwd_algo, //Obtained with getConvolutionBackwardDataAlgorithm
    d_workspace_y_back, 
    workspace_size_y_back, //Obtained with getConvolutionBackwardDataWorkspaceSize
    &zero,
    input_desc, // filter descriptor
    dx
  ));



  // Backward to weight
  checkCUDNN(cudnnConvolutionBackwardFilter(
    cudnn,
    &one,
    input_desc, // input tensor descriptor
    tensor,
    output_desc, // output grad tensor descriptor
    dy,
    conv_desc, // convolution descriptor
    w_bwd_algo, //Obtained with getConvolutionBackwardFilterAlgorithm
    d_workspace_w_back, 
    workspace_size_w_back, //Obtained with getConvolutionBackwardFilterWorkspaceSize
    &one,
    filter_desc, // filter descriptor
    dW
  ));

  

  /*
  std::cout << "d_w is:\n";
  PrintTensorF(dW, C*OC, ks*ks);
  std::cout << "\n";
  */

}