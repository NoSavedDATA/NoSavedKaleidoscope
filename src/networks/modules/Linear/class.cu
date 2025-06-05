
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
#include "../../../nsk_cuda/include.h"
#include "../../../tensor/include.h"
#include "class.h"




LinearCPP::LinearCPP(int C, int OC, std::string Init, std::vector<std::string> Notes, std::string Name)
    : C(C), OC(OC), Init(Init), Notes(Notes), Name(Name) {
    B = 0;


    

    
    // if (in_int_ptr(fp16, Notators->vec, Notators->size))
    //   _fp32 = false;

    if (in_str("fp16", Notes))
      precision = 1;
    if (in_str("i8", Notes))
    {
      precision = 2;
      w8 = get_i8pool(0, OC*C, "Linear w8");

      scale_N = new Minimal_Tensor();
      scale_M = new Minimal_Tensor();
      scale_K = new Minimal_Tensor();
      scale_N->tensor = (void *)get_from_pool(0, OC, "Linear i8 scale_N");
      scale_K->tensor = (void *)get_from_pool(0, C, "Linear i8 scale_K");
    }



    
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
    if (Init=="fixed8i")
      W_cpu = make_xavier_uniform_float_fixed(product, C, OC, 8);
    if (Init=="fixed42i")
      W_cpu = make_xavier_uniform_float_fixed(product, C, OC, 42);
    if (Init=="xavu_relu")
      W_cpu = make_xavier_uniform_float_relu(product, C, OC);
    if (Init=="xavu_tanh")
      W_cpu = make_xavier_uniform_float_tanh(product, C, OC);
    if (Init=="he_normal_relu")
      W_cpu = make_he_normal_float_relu(product, C);
    if (Init=="init_gpt")
      W_cpu = make_gpt_init(product);
    if (Init=="ints")
      W_cpu = make_random_int(product, 10);
    if (Init=="binary")
      W_cpu = make_random_int(product, 1);


    cudaMalloc(&W,       product * sizeof(float));
    cudaMemcpy(W, W_cpu, product * sizeof(float), cudaMemcpyHostToDevice);

    DT_tensor *tensor_W = createTensor(W, {OC*C}, product, true, Name+"W");
    tensor_W->SetIsWeight();

    NamedTensorsT[Name+"W"] = tensor_W;

    delete[] W_cpu;


    first_backward = true;
    changed_descriptors = false;
}


void LinearCPP::SetDescriptors(int B, int thread_id)
{
  if(precision==2)
  {
    std::cout << "Thread id: " << thread_id << ".\n";
    if(x8!=nullptr)
      move_to_i8pool(thread_id, this->B*C, x8, "Linear x8 on set descriptors");
    if(scale_M->tensor!=nullptr)
      move_to_pool(thread_id, B, (float *)scale_M->tensor, "Linear i8 scale_M");

    x8 = get_i8pool(thread_id, B*C, "Linear x8 on set descriptors");
    scale_M->tensor = (void*)get_from_pool(thread_id, B, "Linear i8 scale_M");

    // printf("MALLOCING %d - %d - %d\n", B, C, B*C);
  }

  this->B=B;
  changed_descriptors=true;
}



float *LinearCPP::Forward(DT_tensor *x, int thread_id)
{

  std::vector<int> dims = format_LinearLayer_Dims(x->dims);
  int B = dims[0];

  if (this->B!=B)
    SetDescriptors(B, thread_id);



  cudaStream_t stream = ThreadsStream[thread_id];  

  float *out = get_from_pool(thread_id, B*OC, "linear fwd");
  if (precision==0)
  {
    // std::cout << "Linear is fp32" << ".\n";
    // dim3 block_size(TILE_SIZE, TILE_SIZE);
    // dim3 grid_size(std::ceil(OC/(float)TILE_SIZE), std::ceil(B/(float)TILE_SIZE));
    // int shared_mem_size = 2*TILE_SIZE*TILE_SIZE*sizeof(float);

    // mult_kernel<<<grid_size, block_size, shared_mem_size, stream>>>(x->tensor_ptr, W, out, TILE_SIZE, TILE_SIZE*TILE_SIZE, B, C, OC);
    const float alpha = 1.0f;
    const float beta = 0.0f;
    cublasCheck(cublasSgemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, OC, B, C, &alpha, W, C, x->tensor_ptr, C, &beta, out, OC));
  } else if (precision==1) { 

    constexpr int WMMA_T{16};
    
    // constexpr int num_warps_x{4};
    // constexpr int num_warps_y{4};
    

    // dim3 block_size(num_warps_x * WARP_SIZE, num_warps_y);
    // dim3 block_size_pp(num_warps_x * WARP_SIZE*2, num_warps_y);
    // dim3 grid_size(std::floor((OC + (num_warps_x*WMMA_T - 1)) / (float)(num_warps_x*WMMA_T)), std::floor((B + (num_warps_y*WMMA_T - 1)) / (float)(num_warps_y*WMMA_T)));

    // dim3 grid_size_pp(std::ceil((OC + ((num_warps_x/2)*WMMA_T - 1)) / (float)((num_warps_x/2)*WMMA_T)), std::ceil((B + (num_warps_y*WMMA_T - 1)) / (float)(num_warps_y*WMMA_T)));

    
    // int shared_mem_pp   = (num_warps_y*WMMA_T*WMMA_T*(num_warps_x/2))*sizeof(float) + 2*((num_warps_x/2)+num_warps_y)*WMMA_T*WMMA_T*sizeof(__half);
    // int shared_mem_pp   = (num_warps_y*WMMA_T*WMMA_T*num_warps_x)*sizeof(float) + 2*(num_warps_x+num_warps_y)*WMMA_T*WMMA_T*sizeof(__half);





    // int shared_mem_cf = (num_warps_x+num_warps_y)*WMMA_T*WMMA_T*2*sizeof(float);
    // wmma_cp_async<WMMA_T,num_warps_x,num_warps_y><<<grid_size, block_size, shared_mem_cf, stream>>>(x->tensor_ptr, W, out, B, C, OC);




    blocking_mma<WMMA_T>(x->tensor_ptr, W, out, B, OC, C, stream);
    

    cudaCheck(cudaGetLastError());
  } else if (precision==2) {


    // x8 = get_i8pool(0, B*C, "x8");
    // w8 = get_i8pool(0, OC*C, "w8");



    // std::cout << "Quantize x-------------------------"  << ".\n";
    quantize_f32_to_i8(x8, x->tensor_ptr, scale_M, 0.99, B, C, stream);
    // cudaCheck(cudaGetLastError());
    // std::cout << "Quantize w-------------------------"  << ".\n";
    quantize_f32_to_i8(w8, W, scale_N, 0.99, OC, C, stream);

    cudaCheck(cudaGetLastError());
    // PrintTensorI8(x8, 8, 8);
    // PrintTensorF(x->tensor_ptr, 8, 8);

    // PrintTensorI8(x8+(31*B/32)*C, B/32, C);
    // PrintTensorI8(w8, OC, 4);


    // cudaCheck(cudaGetLastError());

    constexpr int WMMA_T{16};
    blocking_mma_i8<WMMA_T>(x8, w8, out, (float*)scale_M->tensor, (float*)scale_N->tensor, B, OC, C, stream);

    // blocking_mma<WMMA_T>(x->tensor_ptr, W, out, B, OC, C, stream);


    // std::cout << "PRINTING" << ".\n";
    // PrintTensorF(out, B, OC);
    // std::cout << "Printed.." << ".\n";



    cudaCheck(cudaGetLastError());


    // std::cout << "Precision is int8"  << ".\n";
  } else {
    std::cout << "Unknown precision type" << ".\n";
    std::exit(0);
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

  set_to_zero_kernel<<<std::ceil((OC*C)/(float)TILE_SIZE_SQ), TILE_SIZE_SQ, 0, main_stream>>>(dW, OC*C);

  NamedParamGrads[Name+"W"] = dW;

  first_backward=false;
}


void LinearCPP::Backward(float *x, float *dx, float *dy)
{
  float one = 1.0f, zero = 0.0f;
  
  
  if(first_backward)
    FirstBackward();


  if (precision==0)
  {

  // backwad to dx
  cublasCheck(cublasGemmEx(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, C, B, OC, &one,
                             W, CUBLAS_LOWP, C, dy, CUBLAS_LOWP, OC, &zero,
                             dx, CUBLAS_LOWP, C, cublas_compute, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
  
  
  // backward to weight, uses += in the backward pass (accumulate the gradient) by setting alpha=one
  cublasCheck(cublasGemmEx(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T, C, OC, B, &one,
                             x, CUBLAS_LOWP, C, dy, CUBLAS_LOWP, OC, &one,
                             dW, CUBLAS_LOWP, C, cublas_compute, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
  } else if (precision==1) {

    // constexpr int num_warps_x{4};
    // constexpr int num_warps_y{4};
    

    constexpr int WMMA_T{16};
    // dim3 block_size(num_warps_x * WARP_SIZE, num_warps_y);
    // dim3 grid_size_dx(std::ceil((C + (num_warps_x*WMMA_T - 1)) / (float)(num_warps_x*WMMA_T)), std::ceil((B + (num_warps_y*WMMA_T - 1)) / (float)(num_warps_y*WMMA_T)));
    // dim3 grid_size_dw(std::ceil((C + (num_warps_x*WMMA_T - 1)) / (float)(num_warps_x*WMMA_T)), std::ceil((OC + (num_warps_y*WMMA_T - 1)) / (float)(num_warps_y*WMMA_T)));
    
    // int shared_mem_size = num_warps_y*WMMA_T*WMMA_T*num_warps_x*sizeof(float) + (num_warps_x+num_warps_y)*WMMA_T*WMMA_T*sizeof(__half);

    // int shared_mem_cf = (num_warps_y*WMMA_T*WMMA_T*num_warps_x)*sizeof(float) + (num_warps_x+num_warps_y)*WMMA_T*WMMA_T*2*sizeof(float);

    // wmma_dx_cp_async<WMMA_T,num_warps_x,num_warps_y><<<grid_size_dx, block_size, shared_mem_cf, main_stream>>>(dx, W, dy, B, C, OC);
    
    // wmma_backwarddw_kernel<WMMA_T,num_warps_x,num_warps_y><<<grid_size_dw, block_size, shared_mem_size, main_stream>>>(dW, x, dy, B, C, OC);
    // wmma_backwarddx_kernel<WMMA_T,num_warps_x,num_warps_y><<<grid_size_dx, block_size, shared_mem_size, main_stream>>>(dx, W, dy, B, C, OC);


    

    blocking_mma_dw<WMMA_T>(dy, x, dW, OC, C, B, main_stream);
    blocking_mma_dx<WMMA_T>(dy, W, dx, B, C, OC, main_stream);








    /*
    // backward to weight, uses += in the backward pass (accumulate the gradient) by setting alpha=one
    cublasCheck(cublasGemmEx(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T, C, OC, B, &one,
                             x, CUBLAS_LOWP, C, dy, CUBLAS_LOWP, OC, &one,
                             dW, CUBLAS_LOWP, C, cublas_compute, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    */
  } else {
    // std::cout << "backward precision int8" << ".\n";

    
    constexpr int WMMA_T{16};

    float *w_T = get_from_pool(0, OC*C, "Linear w_T");
    float *x_T = get_from_pool(0, B*C, "Linear x_T");
    float *dy_T = get_from_pool(0, B*OC, "Linear w_T");

    transpose_tensor(x_T, x, B, C, main_stream);
    transpose_tensor(w_T, W, OC, C, main_stream);
    transpose_tensor(dy_T, dy, B, OC, main_stream);


    // printf("\n\n%d - %d\n", OC, C);
    PrintTensorF(w_T, 8, std::min(OC,8));
    // PrintTensorF(dy, 8, 8);

    int8_t *w8_T = get_i8pool(0, C*OC, "Linear w8_T");
    int8_t *x8_T = get_i8pool(0, B*C, "linear fwd");
    int8_t *dy8 = get_i8pool(0, B*OC, "linear fwd");
    int8_t *dy8_T = get_i8pool(0, B*OC, "linear fwd");

    quantize_f32_to_i8(dy8, dy, scale_M, 0.99, B, OC, main_stream);
    quantize_f32_to_i8(w8_T, w_T, scale_K, 0.99, C, OC, main_stream);

    PrintTensorI8(w8_T, 8, std::min(OC,8));
    // PrintTensorI8(dy8, 8, 8);
    
    blocking_mma_i8<WMMA_T>(dy8, w8_T, dx, (float*)scale_M->tensor, (float*)scale_K->tensor, B, C, OC, main_stream);

    // quantize_f32_to_i8(dy8_T, dy_T, scale_N, 0.99, OC, B, main_stream);
    // quantize_f32_to_i8(x8_T, x_T, scale_K, 0.99, C, B, main_stream);

    // blocking_mma_i8<WMMA_T>(dy8_T, x8_T, dW, (float*)scale_N->tensor, (float*)scale_K->tensor, OC, C, B, main_stream);



    // quantize_f32_to_i8(w8, W, scale_N, 0.99, OC, C, stream);




    
    blocking_mma_dw<WMMA_T>(dy, x, dW, OC, C, B, main_stream);
    // blocking_mma<WMMA_T>(dy, w_T, dx, B, C, OC, main_stream);

    move_to_pool(0, B*C, x_T, "Linear x_T");
    move_to_pool(0, OC*C, w_T, "Linear w_T");
    move_to_pool(0, B*OC, dy_T, "Linear dy_T");
        
    // quantize_f32_to_i8(dy8, dy, 0.99, B, OC, main_stream);
    
    // printf("dy - %d, %d\n", B, OC);
    // PrintTensorF(dy, 8, OC);
    // printf("dy8\n");
    // PrintTensorI8(dy8, 8, OC);

    // blocking_mma_i8_dw<WMMA_T>(dy8, x8, dW, OC, C, B, main_stream);
    // blocking_mma_i8_dx<WMMA_T>(dy8, w8, dx, B, C, OC, main_stream);

    move_to_i8pool(0, OC*C, w8_T, "Linear w8");
    move_to_i8pool(0, B*C, x8_T, "Linear w8");
    move_to_i8pool(0, B*OC, dy8, "Linear w8");
    move_to_i8pool(0, OC*B, dy8_T, "Linear w8");
  }

}




