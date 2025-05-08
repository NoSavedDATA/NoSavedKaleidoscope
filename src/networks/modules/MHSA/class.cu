
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
#include "../../../compiler_frontend/logging.h"
#include "../../../cuda_kernels/handles.h"
#include "../../../cuda_kernels/elementwise_kernels_inline.cu"
#include "../../../mma/general.h"
#include "../../../mma/wmma_blocking.cu"
#include "../../../notators/notators.h"
#include "../../../tensor/include.h"
#include "class.h"
#include "kernels.h"
#include "template_kernels.h"


MHSA::MHSA(int nh, int C, int maxT, std::string Init, int_vec *Notators, std::string Name)
    : nh(nh), C(C), maxT(maxT), Init(Init), Notators(Notators), Name(Name) {
    B = 0;
    T = 0;
    d = C/nh;
    M = deviceProp.sharedMemPerBlock;


    _fp32 = true;
    _fp32_back = true;
    _causal = false;

    if (in_int_ptr(fp16, Notators->vec, Notators->size))
    {
    _fp32 = false;
    _fp32_back = false;
    }
    if (in_int_ptr(causal, Notators->vec, Notators->size))
    _causal = true;

    
    float *W_cpu, *W_proj_cpu;

    //W_cpu = make_gpt_init(3*C*C);
    //W_proj_cpu = make_gpt_init(C*C);
    W_cpu = make_xavier_uniform_float(3*C*C, C, 3*C);
    W_proj_cpu = make_xavier_uniform_float(C*C, C, C);

    cudaMalloc(&W,       3*C*C*sizeof(float));
    cudaMalloc(&W_proj,  C*C*sizeof(float));
    cudaMemcpy(W, W_cpu, 3*C*C * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(W_proj, W_proj_cpu, C*C * sizeof(float), cudaMemcpyHostToDevice);

    DT_tensor *tensor_W = createTensor(W, {3*(float)C*(float)C}, 3*C*C, true, Name+"W");
    DT_tensor *tensor_W_proj = createTensor(W_proj, {(float)C*(float)C}, C*C, true, Name+"W_proj");
    tensor_W->SetIsWeight();
    tensor_W_proj->SetIsWeight();

    NamedTensorsT[Name+"W"] = tensor_W;
    NamedTensorsT[Name+"W_proj"] = tensor_W_proj;

    delete[] W_cpu;
    delete[] W_proj_cpu;
    delete[] Notators;


    first_backward = true;
    changed_descriptors = false;
}


void MHSA::SetDescriptors(int B, int T, int thread_id)
{
  
  if(B!=0)
  {
    //move_to_pool();
  }

  
  //std::cout << "M: " << M << "\n";

  if (!_fp32)
  {
    Bc = 16;
    Br = 16;

    while (((2*(Br+Bc)*d) + 32*32 + 3*Br)*sizeof(float) < M && Br<32 && Bc<32)
    {
      if(Br>Bc/2)
        Br=Br*2;
      else
        Bc=Bc*2;
    }
  } else {
    
    Bc = std::ceil(  M / ((float)(4*d * sizeof(float)))  );
    Br = (int)fminf(Bc, d);
    Bc = fminf(Bc, 32);
    Br = fminf(Br, 32);    
  }


  
  
  while (((2*(Br+Bc)*d) + Br*Bc + 3*Br)*sizeof(float) > M && Br>1 && Bc>1)
  {
    if(Br>Bc/2)
      Br=(int)Br/2;
    else
      Bc=(int)Bc/2;
  }


  if (!_fp32 && (Bc<16 || Br<16))
  {
    std::string _err = "fp16 is not supported for head dimension " + std::to_string(d) + ", got Br = " + std::to_string(Br) + " and Bc = " + std::to_string(Bc) + ".\n     Falling back into floating point precision 32 (fp32).";
    LogErrorS(_err);
    _fp32 = true;
  }

    if (!_fp32 && (d%16!=0))
  {
    std::string _err = "fp16 is not supported for head dimension " + std::to_string(d) + ". It must be a multiple of 16.\n     Falling back into floating point precision 32 (fp32).";
    LogErrorS(_err);
    _fp32 = true;
  }
  

  Tc = std::ceil(T/(float)Bc);
  Tr = std::ceil(T/(float)Br);


  int last_idx = ((2*(Br+Bc)*d) + Br*Bc + 3*Br)*sizeof(float);
  std::cout << "MHSA::SetDescriptors\n - Bc: " << Bc << ";\n - Br: " << Br << ";\n - Tc: " << Tc << ";\n - Tr: " << Tr << ".\n";
  std::cout << "- M: " << M << ";\n- Last idx: " << last_idx << ".\n";
  

  this->B=B;
  this->T=T;

  qkv = get_from_pool(thread_id, B*T*3*C, "mhsa qkv");
  out = get_from_pool(thread_id, B*T*C, "mhsa out");
  l = get_from_pool(thread_id, B*T*nh, "mhsa l");

  changed_descriptors = true;
}


float *MHSA::Forward(DT_tensor *x, int B, int T, int thread_id)
{
  //std::cout << "MHSA::Forward" << "\n";


  

  float *proj_out = get_from_pool(thread_id, B*T*C, "mhsa out");

  cudaStream_t stream = ThreadsStream[thread_id];  

  if (this->B!=B || this->T!=T)
    SetDescriptors(B, T, thread_id);



  
  //std::cout << "" << main_stream->stream==stream << "\n";

  if (_fp32)
  {
    // Get qkv

    dim3 block_size(TILE_SIZE, TILE_SIZE);
    dim3 grid_size(std::ceil((3*C)/(float)TILE_SIZE), std::ceil((B*T)/(float)TILE_SIZE));
    int shared_mem_size = 2*TILE_SIZE*TILE_SIZE*sizeof(float);


    const float alpha = 1.0f;
    const float beta = 0.0f;


    // std::cout << "" << shared_mem_cf << ", " << (num_warps_y*WMMA_T*WMMA_T*num_warps_x) <<  ", " << (num_warps_x+num_warps_y)*WMMA_T*WMMA_T*2 << "\n";
    //if (thread_id==0)
      cublasSgemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, 3*C, B*T, C, &alpha, W, C, x->tensor_ptr, C, &beta, qkv, 3*C);
    //else
    //mult_kernel<<<grid_size, block_size, shared_mem_size, stream>>>(x->tensor_ptr, W, qkv, TILE_SIZE, TILE_SIZE*TILE_SIZE, B*T, C, 3*C);




    
    
    //cudaStreamSynchronize(stream);
    //PrintTensorF(qkv, B*T, 3*C);

    // Attention

    dim3 grid_size_mhsa(nh, B);
    dim3 block_size_mhsa(8, WARP_SIZE);

    int threads_per_block = block_size_mhsa.x*block_size_mhsa.y;
    int warps_per_block = threads_per_block/WARP_SIZE;

    

    flash_attn_kernel<<<grid_size_mhsa, block_size_mhsa, M, stream>>>(out, qkv, l,
                                                            B, nh, T, d, C, sqrtf(d), Bc, Br, Tc, Tr, TILE_SIZE, warps_per_block, threads_per_block);
    
    //cudaStreamSynchronize(stream);
    //PrintTensorF(out, B*T, C);
    


    // Out Proj
    

    //if (thread_id==0)
      cublasSgemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, C, B*T, C, &alpha, W_proj, C, out, C, &beta, proj_out, C);
    //else
    // dim3 grid_size_proj(std::ceil(C/(float)TILE_SIZE), std::ceil((B*T)/(float)TILE_SIZE));
    //mult_kernel<<<grid_size_proj, block_size, shared_mem_size, stream>>>(out, W_proj, proj_out, TILE_SIZE, TILE_SIZE*TILE_SIZE, B*T, C, C);

  } else {
    // Get qkv

    // constexpr int num_warps_x{4};
    // constexpr int num_warps_y{4};
    

    constexpr int WMMA_T{16};
    // dim3 block_size_wmma(num_warps_x * WARP_SIZE, num_warps_y);
    // dim3 grid_size_wmma_proj(std::floor((3*C + (num_warps_x*WMMA_T - 1)) / (float)(num_warps_x*WMMA_T)), std::floor((B*T + (num_warps_y*WMMA_T - 1)) / (float)(num_warps_y*WMMA_T)));
    // int shared_mem_wmma = (num_warps_y*WMMA_T*WMMA_T*num_warps_x+ WMMA_T*WMMA_T)*sizeof(float) + (num_warps_x+num_warps_y)*WMMA_T*WMMA_T*sizeof(__half);
    // wmma_mult_kernel<WMMA_T,num_warps_x,num_warps_y><<<grid_size_wmma_proj, block_size_wmma, shared_mem_wmma, stream>>>(x->tensor_ptr, W, qkv, B*T, C, 3*C);

    // int shared_mem_cf = (num_warps_y*WMMA_T*WMMA_T*num_warps_x)*sizeof(float);
    // wmma_cp_async<WMMA_T,num_warps_x,num_warps_y><<<grid_size_wmma_proj, block_size_wmma, shared_mem_cf, stream>>>(x->tensor_ptr, W, qkv, B*T, C, 3*C);


    blocking_mma<WMMA_T>(x->tensor_ptr, W, qkv, B*T, C, 3*C, stream);



    // DT_tensor *x_half = float_to_half(x, thread_id, stream);
    // DT_tensor *w_half = createTensor(W, {(float)3*(float)C*(float)C}, 3*C*C, true, "w transpose");
    // // transpose(w_half, thread_id, stream);
    // w_half = float_to_half(w_half, thread_id, stream);
    // half *qkv_half = get_half_from_pool(thread_id, B*T*3*C, "qkv half");
    // mmaAsyncStage3(x_half->half_ptr, w_half->half_ptr, qkv_half, B*T, C, 3*C);
    // qkv = half_to_float_overwrite(qkv_half, qkv, B*T*3*C, stream);





    // cudaCheck(cudaGetLastError());
    
    
    //cudaStreamSynchronize(stream);
    //PrintTensorF(qkv, B*T, 3*C);

    // Attention

    constexpr int num_warps{8};

    dim3 grid_size_mhsa(nh, B);
    dim3 block_size_mhsa(num_warps, WARP_SIZE);

    int threads_per_block = block_size_mhsa.x*block_size_mhsa.y;
    int warps_per_block = threads_per_block/WARP_SIZE;

    
    
    // std::cout << "\n\nB: " << B << ", T: " << T << ", nh: " << nh << ", d: " << d << ", C: " << C << "\n";
    // std::cout << "Launching flash attention with Bc: " << Bc << ", Br: " << Br << ", Tc " << Tc << ", Tr: " << Tr << "\n";
    // std::cout << "TILE_SIZE " << TILE_SIZE  << "\n";
    // int res = ((2*(Br+Bc)*d) + 32*32 + 3*Br);
    // std::cout << "last idx: " << res*sizeof(float) << ", M: " << M << ", warps_per_block: " << warps_per_block <<  "\n\n\n";
    


    flash_attn_kernel<<<grid_size_mhsa, block_size_mhsa, M, stream>>>(out, qkv, l,
                                                           B, nh, T, d, C, sqrtf(d), Bc, Br, Tc, Tr, TILE_SIZE, warps_per_block, threads_per_block);

    // flash_attn_fp16_kernel<WMMA_T, num_warps><<<grid_size_mhsa, block_size_mhsa, M, stream>>>(out, qkv, l,
    //                                                         B, nh, T, d, C, sqrtf(d), Bc, Br, Tc, Tr, TILE_SIZE, warps_per_block, threads_per_block);
    
    //cudaStreamSynchronize(stream);
    //PrintTensorF(out, B*T, C);
    

    // Out Proj

    // dim3 grid_size_wmma(std::floor((C + (num_warps_x*WMMA_T - 1)) / (float)(num_warps_x*WMMA_T)), std::floor((B*T + (num_warps_y*WMMA_T - 1)) / (float)(num_warps_y*WMMA_T)));
    // wmma_cp_async<WMMA_T,num_warps_x,num_warps_y><<<grid_size_wmma, block_size_wmma, shared_mem_cf, stream>>>(out, W_proj, proj_out, B*T, C, C);

    blocking_mma<WMMA_T>(out, W_proj, proj_out, B*T, C, C, stream);
                          
    // cudaCheck(cudaGetLastError());
    
    // move_to_pool(thread_id, B*T*C, x_half->half_ptr, "MHSA qkv");
    // move_to_pool(thread_id, 3*C*C, w_half->half_ptr, "MHSA qkv");
    // move_to_pool(thread_id, B*T*3*C, qkv_half, "MHSA qkv");
    // // delete[] x_half;
  }
  



  //add_forward<<<std::ceil((B*T*C)/THREADS_PER_BLOCK), THREADS_PER_BLOCK, 0, stream>>>(proj_out, x->tensor_ptr, proj_out, B*T*C);




  if (thread_id==0 && nn_mode==training_mode)
  {
    qkv_back = qkv;
    out_back = out;
    B_back = B;
    T_back = T;
    l_back = l;
  } else {
    move_to_pool(thread_id, B*T*3*C, qkv, "MHSA qkv");
    move_to_pool(thread_id, B*T*C, out, "MHSA pre out-proj");
    move_to_pool(thread_id, B*T*nh, l, "MHSA l");
  }
  
  return proj_out;
}




void MHSA::SetBackwardDescriptors()
{


  _fp32_back=true;

  if (!_fp32_back)
  {
    Bc_back = 16;
    Br_back = 16;

    while (((2*(Br_back+Bc_back)*d) + Br_back*Bc_back + 3*Br_back)*sizeof(float) < M && Br_back<32 && Bc_back<32)
    {
      if(Br_back>Bc_back/2)
        Br_back=Br_back*2;
      else
        Bc_back=Bc_back*2;
    }
  } else {
    Bc_back = std::ceil(  M / ((float)(4*d * sizeof(float)))  );
    Br_back = (int)fminf(Bc_back, d);
    Bc_back = fminf(Bc_back, 32);
    Br_back = fminf(Br_back, 32);
  }
    

  
  

  while (((4*Bc_back + 4*Br_back)*d + 3*Br_back*Bc_back + 2*Br_back)*sizeof(float) > M && Br_back>1 && Bc_back>1)
  {
    if(Br_back>Bc_back/2)
      Br_back=(int)Br_back/2;
    else
      Bc_back=(int)Bc_back/2;
  }
  

  if (!_fp32_back && (Bc_back<16 || Br_back<16))
  {
    std::string _err = "fp16 is not supported for head dimension " + std::to_string(d) + ", got Br = " + std::to_string(Br_back) + " and Bc = " + std::to_string(Bc_back) + ".\n     Falling back into floating point precision 32 (fp32) at the backward mode.";
    LogErrorS(_err);
    _fp32_back = true;
  }

  Tc_back = std::ceil(T_back/(float)Bc_back);
  Tr_back = std::ceil(T_back/(float)Br_back);


  int last_idx = ((4*Bc_back + 4*Br_back)*d + 3*Br_back*Bc_back + 2*Br_back)*sizeof(float);
  std::cout << "MHSA::SetBackwardDescriptors\n - Bc: " << Bc_back << ";\n - Br: " << Br_back << ";\n - Tc: " << Tc_back << ";\n - Tr: " << Tr_back << ".\n";
  std::cout << "- M: " << M << ";\n- Last idx: " << last_idx << ".\n";

  changed_descriptors=false;
}


void MHSA::FirstBackward()
{
  dW = get_from_pool(0, 3*C*C, "MHSA dW");
  dW_proj = get_from_pool(0, C*C, "MHSA dW");

  set_to_zero_kernel<<<std::ceil((3*C*C)/(float)TILE_SIZE_SQ), TILE_SIZE_SQ, 0, main_stream->stream>>>(dW, 3*C*C);
  set_to_zero_kernel<<<std::ceil((C*C)/(float)TILE_SIZE_SQ), TILE_SIZE_SQ, 0, main_stream->stream>>>(dW_proj, C*C);

  NamedParamGrads[Name+"W"] = dW;
  NamedParamGrads[Name+"W_proj"] = dW_proj;

  first_backward=false;
}



void MHSA::Backward(float *x, float *dx, float *dy)
{
  if (changed_descriptors)
    SetBackwardDescriptors();

  if (first_backward)
    FirstBackward();

  float *d_out = get_from_pool(0, B_back*T_back*C, "MHSA d_attn");
  float *d_qkv = get_from_pool(0, B_back*T_back*3*C, "MHSA d_qkv");
  float *D = get_from_pool(0, B_back*T_back*nh, "MHSA backward D");
  float *D_aux = get_from_pool(0, B_back*T_back*T_back*nh, "MHSA backward D");


  float one = 1.0f, zero = 0.0f;
  

  if (_fp32_back)
  {
    
    dim3 block_size(TILE_SIZE, TILE_SIZE);
    dim3 grid_size_dwproj(std::ceil(C/(float)TILE_SIZE), std::ceil(C/(float)TILE_SIZE));
    dim3 grid_size_dxproj(std::ceil(C/(float)TILE_SIZE), std::ceil((B*T)/(float)TILE_SIZE));
    int shared_mem_size = 2*TILE_SIZE_SQ*sizeof(float);


    //cudaStream_t dw_proj_stream, dw_stream;
    //cudaStreamCreate(&dw_proj_stream);
    //cudaStreamCreate(&dw_stream);


    
    //StreamAwaitStreamB(dw_proj_stream, main_stream->stream);
    


    //mult_backwarddw<<<grid_size_dwproj, block_size, shared_mem_size, main_stream->stream>>>(out, dW_proj, dy, TILE_SIZE, TILE_SIZE_SQ, B*T, C, C);
    //mult_backwarddx<<<grid_size_dwproj, block_size, shared_mem_size, main_stream->stream>>>(W_proj, d_out, dy, TILE_SIZE, TILE_SIZE_SQ, B*T, C, C);


    
    cublasGemmEx(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T, C, C, B*T, &one,
                              out, CUBLAS_LOWP, C, dy, CUBLAS_LOWP, C, &one,
                              dW_proj, CUBLAS_LOWP, C, cublas_compute, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    cublasGemmEx(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, C, B*T, C, &one,
                              W_proj, CUBLAS_LOWP, C, dy, CUBLAS_LOWP, C, &zero,
                              d_out, CUBLAS_LOWP, C, cublas_compute, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    



    //PrintTensorF(d_out, T, C);


    dim3 grid_size_mhsa(nh, B);
    dim3 block_size_mhsa(8, WARP_SIZE);

    int threads_per_block = block_size_mhsa.x*block_size_mhsa.y;
    int warps_per_block = threads_per_block/WARP_SIZE;


    //int last_id = ((4*Bc_back + 4*Br_back)*d + 3*Br_back*Bc_back + 2*Br_back)*sizeof(float);
    //std::cout << "backward last_id: " << last_id << ", M: " << M << "\n";
    //std::cout << "Bc: " << Bc_back << ", Br: " << Br_back << ", Tc: " << Tc_back << ", Tr: " << Tr_back << "\n";
    flash_attn_backward_kernel<<<grid_size_mhsa, block_size_mhsa, M, main_stream->stream>>>(d_qkv, d_out, qkv, out, l, D,
                                                                                B, nh, T, d, C, sqrtf(d),
                                                                                Bc_back, Br_back, Tc_back, Tr_back,
                                                                                warps_per_block, threads_per_block);

    //PrintTensorF(d_qkv, 3, C);
    //PrintTensorF(d_qkv, T, 3*C);
    
    //StreamAwaitStreamB(dw_stream, main_stream->stream);

    dim3 grid_size_dx(std::ceil(C/(float)TILE_SIZE), std::ceil((B*T)/(float)TILE_SIZE));
    dim3 grid_size_dw(std::ceil(C/(float)TILE_SIZE), std::ceil((3*C)/(float)TILE_SIZE));
    //mult_backwarddw<<<grid_size_dw, block_size, shared_mem_size, main_stream->stream>>>(x, dW, d_qkv, TILE_SIZE, TILE_SIZE_SQ, B*T, C, 3*C);
    //mult_backwarddx<<<grid_size_dx, block_size, shared_mem_size, main_stream->stream>>>(W, dx, d_qkv, TILE_SIZE, TILE_SIZE_SQ, B*T, C, 3*C);
    
    
    cublasGemmEx(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T, C, 3*C, B*T, &one,
                              x, CUBLAS_LOWP, C, d_qkv, CUBLAS_LOWP, 3*C, &one,
                              dW, CUBLAS_LOWP, C, cublas_compute, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    cublasGemmEx(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, C, B*T, 3*C, &one,
                              W, CUBLAS_LOWP, C, d_qkv, CUBLAS_LOWP, 3*C, &zero,
                              dx, CUBLAS_LOWP, C, cublas_compute, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    

    //StreamAwaitStreamB(main_stream->stream, dw_proj_stream);
    //StreamAwaitStreamB(main_stream->stream, dw_stream);
    //cudaStreamDestroy(dw_proj_stream);
    //cudaStreamDestroy(dw_stream);
  } else {
    

    //cudaStream_t dw_proj_stream, dw_stream;
    //cudaStreamCreate(&dw_proj_stream);
    //cudaStreamCreate(&dw_stream);


    
    //StreamAwaitStreamB(dw_proj_stream, main_stream->stream);
    
    constexpr int num_warps_x{4};
    constexpr int num_warps_y{4};
    
    constexpr int WMMA_T{16};
    
    int shared_mem_size = num_warps_y*WMMA_T*WMMA_T*num_warps_x*sizeof(float) + (num_warps_x+num_warps_y)*WMMA_T*WMMA_T*sizeof(__half);

    dim3 block_size(num_warps_x * WARP_SIZE, num_warps_y);

    dim3 grid_size_dx_proj(std::ceil((C + (num_warps_x*WMMA_T - 1)) / (float)(num_warps_x*WMMA_T)), std::ceil((B*T + (num_warps_y*WMMA_T - 1)) / (float)(num_warps_y*WMMA_T)));
    dim3 grid_size_dw_proj(std::ceil((C + (num_warps_x*WMMA_T - 1)) / (float)(num_warps_x*WMMA_T)), std::ceil((C + (num_warps_y*WMMA_T - 1)) / (float)(num_warps_y*WMMA_T)));

    wmma_backwarddx_kernel<WMMA_T,num_warps_x,num_warps_y><<<grid_size_dx_proj, block_size, shared_mem_size, main_stream->stream>>>(d_out, W_proj, dy, B*T, C, C);
    wmma_backwarddw_kernel<WMMA_T,num_warps_x,num_warps_y><<<grid_size_dw_proj, block_size, shared_mem_size, main_stream->stream>>>(dW_proj, out, dy, B*T, C, C);
    
    


    dim3 grid_size_mhsa(nh, B);
    dim3 block_size_mhsa(8, WARP_SIZE);

    int threads_per_block = block_size_mhsa.x*block_size_mhsa.y;
    int warps_per_block = threads_per_block/WARP_SIZE;


    //int last_id = ((4*Bc_back + 4*Br_back)*d + 3*Br_back*Bc_back + 2*Br_back)*sizeof(float);
    //std::cout << "backward last_id: " << last_id << ", M: " << M << "\n";
    //std::cout << "Bc: " << Bc_back << ", Br: " << Br_back << ", Tc: " << Tc_back << ", Tr: " << Tr_back << "\n";
    flash_attn_backward_kernel<<<grid_size_mhsa, block_size_mhsa, M, main_stream->stream>>>(d_qkv, d_out, qkv, out, l, D,
                                                                                B, nh, T, d, C, sqrtf(d),
                                                                                Bc_back, Br_back, Tc_back, Tr_back,
                                                                                warps_per_block, threads_per_block);

    //PrintTensorF(d_qkv, 3, C);
    //PrintTensorF(d_qkv, T, 3*C);
    
    //StreamAwaitStreamB(dw_stream, main_stream->stream);



    dim3 grid_size_dx(std::ceil((C + (num_warps_x*WMMA_T - 1)) / (float)(num_warps_x*WMMA_T)), std::ceil((B*T + (num_warps_y*WMMA_T - 1)) / (float)(num_warps_y*WMMA_T)));
    dim3 grid_size_dw(std::ceil((C + (num_warps_x*WMMA_T - 1)) / (float)(num_warps_x*WMMA_T)), std::ceil((3*C + (num_warps_y*WMMA_T - 1)) / (float)(num_warps_y*WMMA_T)));

    wmma_backwarddx_kernel<WMMA_T,num_warps_x,num_warps_y><<<grid_size_dx, block_size, shared_mem_size, main_stream->stream>>>(dx, W, d_qkv, B*T, C, 3*C);
    wmma_backwarddw_kernel<WMMA_T,num_warps_x,num_warps_y><<<grid_size_dw, block_size, shared_mem_size, main_stream->stream>>>(dW, x, d_qkv, B*T, C, 3*C);
    
    
  

    //StreamAwaitStreamB(main_stream->stream, dw_proj_stream);
    //StreamAwaitStreamB(main_stream->stream, dw_stream);
    //cudaStreamDestroy(dw_proj_stream);
    //cudaStreamDestroy(dw_stream);

  }







  //add_forward<<<std::ceil((B*T*C)/THREADS_PER_BLOCK), THREADS_PER_BLOCK, 0, main_stream->stream>>>(dx, dx, dy, B*T*C);


  // Clean-up
  move_to_pool(0, B_back*T_back*C, d_out, "MHSA d_attn");
  move_to_pool(0, B_back*T_back*3*C, d_qkv, "MHSA d_qkv");
  move_to_pool(0, B_back*T_back*nh, D, "MHSA backward D");
  move_to_pool(0, B_back*T_back*T_back*nh, D_aux, "MHSA backward D");
}