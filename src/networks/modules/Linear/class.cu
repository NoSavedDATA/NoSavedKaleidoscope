
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
#include "../../../common/include.h"
#include "../../../cuda_kernels/elementwise_kernels_inline.cu"
#include "../../../cuda_kernels/handles.h"
#include "../../../mma/include.h"
#include "../../../notators/notators.h"
#include "../../../tensor/include.h"
#include "class.h"




LinearCPP::LinearCPP(int C, int OC, std::string Init, std::vector<std::string> Notes, std::string Name)
    : C(C), OC(OC), Init(Init), Notes(Notes), Name(Name) {
    B = 0;

    _fp32 = true;
    // if (in_int_ptr(fp16, Notators->vec, Notators->size))
    //   _fp32 = false;

    if (in_str("fp16", Notes))
      _fp32 = false;



    
    float *W_cpu;
    int product = OC*C;


    if (Init=="randu")
      W_cpu = make_random_float_uniform(product);
    if (Init=="zeros")
      W_cpu = make_zeros_float(product);
    if (Init=="ones")
      W_cpu = make_ones_float(product);
    if (Init=="normal")
      W_cpu = make_normal(product);
    if (Init=="xavu")
      W_cpu = make_xavier_uniform_float(product, C, OC);
    if (Init=="xavu_relu")
      W_cpu = make_xavier_uniform_float_relu(product, C, OC);
    if (Init=="xavu_tanh")
      W_cpu = make_xavier_uniform_float_tanh(product, C, OC);
    if (Init=="he_normal_relu")
      W_cpu = make_he_normal_float_relu(product, C);
    if (Init=="init_gpt")
      W_cpu = make_gpt_init(product);
    if (Init=="int")
      W_cpu = make_random_int(product, 10);
    if (Init=="binary")
      W_cpu = make_random_int(product, 1);


    cudaMalloc(&W,       product * sizeof(float));
    cudaMemcpy(W, W_cpu, product * sizeof(float), cudaMemcpyHostToDevice);

    data_type_tensor *tensor_W = createTensor(W, {(float)OC*(float)C}, product, true, Name+"W");
    tensor_W->SetIsWeight();

    NamedTensorsT[Name+"W"] = tensor_W;

    delete[] W_cpu;


    first_backward = true;
    changed_descriptors = false;
}


void LinearCPP::SetDescriptors(int B, int thread_id)
{
  this->B=B;
  changed_descriptors=true;
}



float *LinearCPP::Forward(data_type_tensor *x, int thread_id)
{

  std::vector<float> dims = format_LinearLayer_Dims(x->dims);
  int B = dims[0];

  if (this->B!=B)
    SetDescriptors(B, thread_id);


  float *out = get_from_pool(thread_id, B*OC, "linear fwd");

  cudaStream_t stream = ThreadsStream[thread_id];  


  if (_fp32)
  {
    dim3 block_size(TILE_SIZE, TILE_SIZE);
    dim3 grid_size(std::ceil(OC/(float)TILE_SIZE), std::ceil(B/(float)TILE_SIZE));
    int shared_mem_size = 2*TILE_SIZE*TILE_SIZE*sizeof(float);

    mult_kernel<<<grid_size, block_size, shared_mem_size, stream>>>(x->tensor_ptr, W, out, TILE_SIZE, TILE_SIZE*TILE_SIZE, B, C, OC);
  } else {
    
    constexpr int num_warps_x{4};
    constexpr int num_warps_y{4};
    

    constexpr int WMMA_T{16};
    dim3 block_size(num_warps_x * WARP_SIZE, num_warps_y);
    dim3 block_size_pp(num_warps_x * WARP_SIZE*2, num_warps_y);
    dim3 grid_size(std::floor((OC + (num_warps_x*WMMA_T - 1)) / (float)(num_warps_x*WMMA_T)), std::floor((B + (num_warps_y*WMMA_T - 1)) / (float)(num_warps_y*WMMA_T)));

    // dim3 grid_size_pp(std::ceil((OC + ((num_warps_x/2)*WMMA_T - 1)) / (float)((num_warps_x/2)*WMMA_T)), std::ceil((B + (num_warps_y*WMMA_T - 1)) / (float)(num_warps_y*WMMA_T)));

    
    // int shared_mem_pp   = (num_warps_y*WMMA_T*WMMA_T*(num_warps_x/2))*sizeof(float) + 2*((num_warps_x/2)+num_warps_y)*WMMA_T*WMMA_T*sizeof(__half);
    int shared_mem_pp   = (num_warps_y*WMMA_T*WMMA_T*num_warps_x)*sizeof(float) + 2*(num_warps_x+num_warps_y)*WMMA_T*WMMA_T*sizeof(__half);





    // int shared_mem_cf = (num_warps_x+num_warps_y)*WMMA_T*WMMA_T*2*sizeof(float);
    // wmma_cp_async<WMMA_T,num_warps_x,num_warps_y><<<grid_size, block_size, shared_mem_cf, stream>>>(x->tensor_ptr, W, out, B, C, OC);


    
    blocking_mma<WMMA_T>(x->tensor_ptr, W, out, B, C, OC, stream);
    



    // float *bank;
    // cudaMalloc(&bank, 16*32*4);
    
    // std::cout << "\n\n\n";
    // std::cout << "B: " << B << ", C: " << C << ", OC: " << OC << "\nbx " << grid_size.x << ", by " << grid_size.y << "\n\n";

    // int shared_mem_size = (num_warps_y*WMMA_T*WMMA_T*num_warps_x)*sizeof(float) +   (num_warps_x+num_warps_y)*WMMA_T*WMMA_T*sizeof(__half);
    // wmma_mult_kernel<WMMA_T,num_warps_x,num_warps_y><<<grid_size, block_size, shared_mem_size, stream>>>(x->tensor_ptr, W, out, B, C, OC);
    // wmma_pingpong<WMMA_T,num_warps_x,num_warps_y><<<grid_size, block_size_pp, shared_mem_pp, stream>>>(x->tensor_ptr, W, out, B, C, OC);

    // PrintTensorF(bank, 32, 16);

    // wmma_mult_kernel<WMMA_T,num_warps_x,num_warps_y><<<grid_size, block_size, shared_mem_size, stream>>>(x->tensor_ptr, W, out, B, C, OC);


    // PrintTensorF(out, 16, 16);

    cudaCheck(cudaGetLastError());

  }
  
  return out;
}



void LinearCPP::SetBackwardDescriptors()
{
  changed_descriptors=false;
}

void LinearCPP::FirstBackward()
{
  dW = get_from_pool(0, OC*C, "MHSA dW");

  set_to_zero_kernel<<<std::ceil((OC*C)/(float)TILE_SIZE_SQ), TILE_SIZE_SQ, 0, main_stream->stream>>>(dW, OC*C);

  NamedParamGrads[Name+"W"] = dW;

  first_backward=false;
}


void LinearCPP::Backward(float *x, float *dx, float *dy)
{
  float one = 1.0f, zero = 0.0f;
  
  
  if(first_backward)
    FirstBackward();


  if (_fp32)
  {
  // backwad to dx
  cublasCheck(cublasGemmEx(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, C, B, OC, &one,
                             W, CUBLAS_LOWP, C, dy, CUBLAS_LOWP, OC, &zero,
                             dx, CUBLAS_LOWP, C, cublas_compute, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
  
  
  // backward to weight, uses += in the backward pass (accumulate the gradient) by setting alpha=one
  cublasCheck(cublasGemmEx(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T, C, OC, B, &one,
                             x, CUBLAS_LOWP, C, dy, CUBLAS_LOWP, OC, &one,
                             dW, CUBLAS_LOWP, C, cublas_compute, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
  } else {

    constexpr int num_warps_x{4};
    constexpr int num_warps_y{4};
    

    constexpr int WMMA_T{16};
    dim3 block_size(num_warps_x * WARP_SIZE, num_warps_y);
    dim3 grid_size_dx(std::ceil((C + (num_warps_x*WMMA_T - 1)) / (float)(num_warps_x*WMMA_T)), std::ceil((B + (num_warps_y*WMMA_T - 1)) / (float)(num_warps_y*WMMA_T)));
    dim3 grid_size_dw(std::ceil((C + (num_warps_x*WMMA_T - 1)) / (float)(num_warps_x*WMMA_T)), std::ceil((OC + (num_warps_y*WMMA_T - 1)) / (float)(num_warps_y*WMMA_T)));
    
    int shared_mem_size = num_warps_y*WMMA_T*WMMA_T*num_warps_x*sizeof(float) + (num_warps_x+num_warps_y)*WMMA_T*WMMA_T*sizeof(__half);

    int shared_mem_cf = (num_warps_y*WMMA_T*WMMA_T*num_warps_x)*sizeof(float) +   (num_warps_x+num_warps_y)*WMMA_T*WMMA_T*2*sizeof(float);

    // wmma_dx_cp_async<WMMA_T,num_warps_x,num_warps_y><<<grid_size_dx, block_size, shared_mem_cf, main_stream->stream>>>(dx, W, dy, B, C, OC);
    wmma_backwarddx_kernel<WMMA_T,num_warps_x,num_warps_y><<<grid_size_dx, block_size, shared_mem_size, main_stream->stream>>>(dx, W, dy, B, C, OC);
    wmma_backwarddw_kernel<WMMA_T,num_warps_x,num_warps_y><<<grid_size_dw, block_size, shared_mem_size, main_stream->stream>>>(dW, x, dy, B, C, OC);




    /*
    // backward to weight, uses += in the backward pass (accumulate the gradient) by setting alpha=one
    cublasCheck(cublasGemmEx(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T, C, OC, B, &one,
                             x, CUBLAS_LOWP, C, dy, CUBLAS_LOWP, OC, &one,
                             dW, CUBLAS_LOWP, C, cublas_compute, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    */
  }
}




