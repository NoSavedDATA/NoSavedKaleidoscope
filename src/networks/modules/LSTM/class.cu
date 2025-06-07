
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
#include "../../../cuda_kernels/handles.h"
#include "../../../cuda_kernels/elementwise_kernels_inline.cu"
#include "../../../mma/general.h"
#include "../../../nsk_cuda/pool/include.h"
#include "../../../tensor/include.h"
#include "class.h"
#include "kernels.h"


DT_LSTM::DT_LSTM(int C, int OC, std::string Init, std::string Name)
    : C(C), OC(OC), Init(Init), Name(Name) {
    B = 0;
    T = 0;

    x_out = nullptr;
    fused_out = nullptr;

    float *w_cpu, *u_cpu, *b_cpu;
    w_cpu = make_N_orthogonals(4, OC, OC);
    //w_cpu = make_lstm_init_xavier(OC, OC);
    u_cpu = make_lstm_init_xavier(OC,  C);

    //w_cpu = make_lstm_torch(OC, OC);
    //u_cpu = make_lstm_torch(OC, C);

    b_cpu = make_lstm_bias(OC);


    cudaMalloc(&W, round_to_nearest_pow2(4*OC*OC)*sizeof(float));
    cudaMalloc(&U, round_to_nearest_pow2(4*OC* C)*sizeof(float));
    cudaMalloc(&b, round_to_nearest_pow2(4*OC   )*sizeof(float));
    cudaMemcpy(W, w_cpu, 4*OC*OC*sizeof(float), cudaMemcpyHostToDevice); // ht weight
    cudaMemcpy(U, u_cpu, 4*OC* C*sizeof(float), cudaMemcpyHostToDevice); // x weight
    cudaMemcpy(b, b_cpu, 4*OC*   sizeof(float), cudaMemcpyHostToDevice); // bias

    DT_tensor *tensor_W = createTensor(W, {4*OC, OC}, 4*OC*OC, true, Name+"W");
    DT_tensor *tensor_U = createTensor(U, {4*OC, C},  4*OC* C, true, Name+"U");
    DT_tensor *tensor_B = createTensor(b, {4*OC},            4*OC   , true, Name+"b");
    tensor_W->SetIsWeight();
    tensor_U->SetIsWeight();


    NamedTensorsT[Name+"W"] = tensor_W;
    NamedTensorsT[Name+"U"] = tensor_U;
    NamedTensorsT[Name+"B"] = tensor_B;

    delete[] w_cpu;
    delete[] u_cpu;

    changed_descriptors = false;
    first_backward = true;
}



void DT_LSTM::SetDescriptors(int B, int T, int thread_id)
{

  if (x_out!=nullptr)
  {
    cudaFree(x_out);
    cudaFree(fused_out);
    cudaFree(all_ht);
    cudaFree(all_ct);
  }

  x_out = get_from_pool(thread_id, B*T*4*OC, "lstm x@U out");
  fused_out = get_from_pool(thread_id, T*B*4*OC, "lstm ht@W");

  all_ht = get_from_pool(thread_id, T*B*OC, "lstm all ht");
  all_ct = get_from_pool(thread_id, T*B*OC, "lstm all ct");


  int grid_size, block_size; 
  CalculateGridAndBlockSizes(T*B*OC, grid_size, block_size);


  //set_to_zero_kernel<<<grid_size, block_size, 0, main_stream>>>(all_ht, B*T*OC);


  this->B=B;
  this->T=T;
  changed_descriptors=true;
}


float *DT_LSTM::Forward(DT_tensor *tensor_x, DT_tensor *tensor_ht, DT_tensor *tensor_ct, int B, int T, int thread_id)
{

  dim3 block_size(TILE_SIZE, TILE_SIZE);
  dim3 grid_size(  std::ceil( (4*OC) / (float)TILE_SIZE),   std::ceil( (B*T) / (float)TILE_SIZE)  );
  int shared_mem_size = 2*TILE_SIZE_SQ*sizeof(float);


  
  
  if (B!=this->B || T!=this->T)
    SetDescriptors(B,T,thread_id);
  
  cudaStream_t stream = ThreadsStream[thread_id];
  //cudaStream_t stream = main_stream;





  

  //mult_kernel<<<grid_size, block_size, shared_mem_size, stream>>>(tensor_x->tensor_ptr, U, x_out, TILE_SIZE, TILE_SIZE_SQ, B*T, C, 4*OC);


  
  constexpr int num_warps_x{8};
  constexpr int num_warps_y{4};
  

  constexpr int WMMA_T{16};
  dim3 block_size_wmma(num_warps_x * WARP_SIZE, num_warps_y);
  dim3 grid_size_wmma(std::ceil((4*OC + (num_warps_x*WMMA_T - 1)) / (float)(num_warps_x*WMMA_T)), std::ceil((B*T + (num_warps_y*WMMA_T - 1)) / (float)(num_warps_y*WMMA_T)));
  int shared_mem_wmma = num_warps_y*WMMA_T*WMMA_T*num_warps_x*sizeof(float) + (num_warps_x+num_warps_y)*WMMA_T*WMMA_T*sizeof(__half);
  wmma_mult_kernel<WMMA_T,num_warps_x,num_warps_y><<<grid_size_wmma, block_size_wmma, shared_mem_wmma, stream>>>(tensor_x->tensor_ptr, U, x_out, B*T, C, 4*OC);
  





  //move_to_pool(tensor_ht->dims_prod, tensor_ht->tensor_ptr, "input ht");
  //move_to_pool(tensor_ct->dims_prod, tensor_ct->tensor_ptr, "input ct");
  
  

  dim3 grid_size_lstm(  std::ceil( (4*OC) / (float)TILE_SIZE),   std::ceil( B / (float)TILE_SIZE)  );
  dim3 grid_size_elementwises(  std::ceil( (OC) / (float)TILE_SIZE),   std::ceil( B / (float)TILE_SIZE)  );



  //std::cout << "\nx out"  << "\n";
  //PrintTensorF(x_out, B*T, 4*OC);
  //std::cout << "\n";



  int f_offset =     OC;
  int o_offset = 2 * OC;
  int c_offset = 3 * OC;

  for (int t=0; t<T; ++t)
  {
    //std::cout << "Forward t: " << t << "\n";


    lstm_single_step_kernel<<<grid_size_lstm, block_size, shared_mem_size, stream>>>(fused_out, x_out, W, all_ht, b,
                                                                                      t, T, TILE_SIZE, TILE_SIZE_SQ, B, OC, 4*OC, 3*OC);

    //std::cout << "\nFused out"  << "\n";
    //PrintTensorF(fused_out, B, 4*OC);
    //std::cout << "\n";

    lstm_elementwise_ops_kernel<<<grid_size_elementwises, block_size, 0, stream>>>(fused_out,
                                                                                      all_ht, all_ct,
                                                                                      TILE_SIZE, TILE_SIZE_SQ,
                                                                                      t, T,
                                                                                      B, OC, 4*OC,
                                                                                      f_offset, o_offset, c_offset);
  }


  tensor_ht->tensor_ptr = all_ht + (int)((T-1)*B*OC);
  tensor_ct->tensor_ptr = all_ct + (int)((T-1)*B*OC);

  return tensor_ht->tensor_ptr;
}

void DT_LSTM::SetBackwardDescriptors()
{
  std::cout << "Changed LSTM descriptors." << "\n";


  d_ht   = get_from_pool(0, B*OC, "d_ht");
  d_ct   = get_from_pool(0, B*OC, "d_ct");
  d_ifoc = get_from_pool(0, T*B*4*OC, "d_ct");

  changed_descriptors=false;
}

void DT_LSTM::FirstBackward()
{
  std::cout << "First LSTM backward." << "\n";

  dW = get_from_pool(0, 4*OC*OC, "lstm dW");
  dU = get_from_pool(0, 4*OC* C, "lstm dU");
  dB = get_from_pool(0, 4*OC,    "lstm dB");

  set_to_zero_kernel<<<std::ceil((4*OC*OC)/(float)TILE_SIZE_SQ), TILE_SIZE_SQ, 0, main_stream>>>(dW, 4*OC*OC);
  set_to_zero_kernel<<<std::ceil((4*OC* C)/(float)TILE_SIZE_SQ), TILE_SIZE_SQ, 0, main_stream>>>(dU, 4*OC*C);
  set_to_zero_kernel<<<std::ceil((4*OC)   /(float)TILE_SIZE_SQ), TILE_SIZE_SQ, 0, main_stream>>>(dB, 4*OC);

  NamedParamGrads[Name+"W"] = dW;
  NamedParamGrads[Name+"U"] = dU;
  NamedParamGrads[Name+"B"] = dB;

  first_backward=false;
}


void DT_LSTM::Backward(float *x, float *dx, float *dy)
{
  dim3 block_size(TILE_SIZE, TILE_SIZE);
  int shared_mem_size = 2*TILE_SIZE_SQ*sizeof(float);



  if (first_backward)
    FirstBackward();
  if (changed_descriptors)
    SetBackwardDescriptors();

  

  //std::cout << "Copy dy to d_ht" << "\n";
  copy_tensor_kernel<<<std::ceil(((float)B*(float)OC)/(float)TILE_SIZE_SQ), TILE_SIZE_SQ, 0, main_stream>>>(d_ht, dy, B*OC);
  set_to_zero_kernel<<<std::ceil(((float)B*(float)OC)/(float)TILE_SIZE_SQ), TILE_SIZE_SQ, 0, main_stream>>>(d_ct,     B*OC); // TODO: check if removing this one is safe
  set_to_zero_kernel<<<std::ceil(((float)T*(float)B*4*(float)OC)/(float)TILE_SIZE_SQ), TILE_SIZE_SQ, 0, main_stream>>>(d_ifoc, T*B*4*OC);


  

  //PrintTensorF(d_ht, B, OC);



  dim3 grid_size_elementwises(  std::ceil( (float)OC / (float)TILE_SIZE),   std::ceil( (float)B / (float)TILE_SIZE)  );
  dim3 grid_size_d_ht(          std::ceil( (float)OC / (float)TILE_SIZE),   std::ceil( (float)B / (float)TILE_SIZE)  );


  int f_offset =     OC;
  int o_offset = 2 * OC;
  int c_offset = 3 * OC;

  int reversed_t_;

  for (int t=0; t<T; ++t)
  {
    reversed_t_ = T-t-1;
    
    //std::cout << "backward t: " << t << ", reversed t: " << reversed_t_ << "\n";

    lstm_elementwise_ops_backward_kernel<<<grid_size_elementwises, block_size, shared_mem_size, main_stream>>>(fused_out,
                                                                                      all_ct,
                                                                                      d_ht, d_ct, d_ifoc, dB,
                                                                                      W,
                                                                                      TILE_SIZE, TILE_SIZE_SQ,
                                                                                      t, reversed_t_, T,
                                                                                      B, OC, 4*OC,
                                                                                      f_offset, o_offset, c_offset);

    //PrintTensorF(fused_out+reversed_t_*B*4*OC, B, 4*OC);
    //PrintTensorF(d_ifoc, B, OC);
    /*
    lstm_single_step_backward_dht_kernel<<<grid_size_d_ht, block_size, shared_mem_size, main_stream>>>(d_ifoc,
                                                                                      d_ht, W,
                                                                                      t, reversed_t_, T,
                                                                                      TILE_SIZE, TILE_SIZE_SQ,
                                                                                      B, OC, 4*OC);
    */
    //PrintTensorF(d_ht, B, OC);
  }
  //PrintTensorF(d_ht, B, OC);

  dim3 grid_size_dx(  std::ceil( (float)C  / (float)TILE_SIZE),   std::ceil( (float)B*(float)T   / (float)TILE_SIZE)  );
  //dim3 grid_size_dw(  std::ceil( 4*(float)OC*(float)OC / (float)TILE_SIZE_SQ)  );
  //dim3 grid_size_du(  std::ceil( 4*(float)OC*(float)C  / (float)TILE_SIZE_SQ)  );

  dim3 grid_size_dw(std::ceil(OC/(float)TILE_SIZE), std::ceil((4*OC)/(float)TILE_SIZE));
  dim3 grid_size_du(std::ceil(C /(float)TILE_SIZE), std::ceil((4*OC)/(float)TILE_SIZE));


  // all_ht    e [T, B,    OC]
  // all_ct    e [T, B,    OC]
  // fused out e [T, B,  4*OC]
  // d_ifoc    e [T, B,  4*OC]

  // x         e [B, T,    OC]



  
  cudaStream_t dx_stream, dw_stream;
  cudaStreamCreate(&dx_stream);
  cudaStreamCreate(&dw_stream);

  cudaStreamSynchronize(main_stream);

  
  lstm_backward_dx_kernel<<<grid_size_dx, block_size, shared_mem_size, dx_stream>>>(d_ifoc, dx, U,
                                                                                      TILE_SIZE, TILE_SIZE_SQ, B, T, C, 4*OC);
  RegisterEvent(dx_stream);
  
  

  
  mult_backwarddw<<<grid_size_dw, block_size, shared_mem_size, dw_stream>>>(all_ht, dW, d_ifoc, TILE_SIZE, TILE_SIZE_SQ, B*T, OC, 4*OC);
  RegisterEvent(dw_stream);

  
  mult_backwarddw<<<grid_size_du, block_size, shared_mem_size, main_stream>>>(x, dU, d_ifoc, TILE_SIZE, TILE_SIZE_SQ, B*T, C, 4*OC);

  WaitForAllEvents();
  cudaStreamDestroy(dx_stream);
  cudaStreamDestroy(dw_stream);
  

  /*
  move_to_pool(B*OC, d_ht, "d_ht");
  move_to_pool(B*OC, d_ct, "d_ct");
  move_to_pool(T*B*4*OC, d_ifoc, "d_ht");
  */
} 