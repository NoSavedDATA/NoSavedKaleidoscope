#pragma once

#include <cuda_runtime.h>
#include <cublasLt.h>
#include <cublas_v2.h>
#include <curand_kernel.h>
#include <cuda_fp16.h>
#include <cudnn.h>
#include <mma.h>




__global__ void lstm_single_step_kernel(float *fused_out, const float *x_out, const float *W, const float *ht, const float *b,
                      const int t, const int T, const int tile_size, const int tile_offset,
                      const int B, const int OC, const int fourX_OC, const int tanh_offset) {
  // x_out  e [B,  4*OC]
  // ht     e [T, B, OC]
  // W      e [4*OC, OC]

  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int x_block = blockIdx.x;
  int y_block = blockIdx.y;

  

  int row = y_block*tile_size + ty;
  int col = x_block*tile_size + tx;



  int offset = tile_offset;

  float y = 0.0f;


  extern __shared__ float smem[];


  

  if (t>0)
  {
    for (int i=0; i < ceilf(OC/(float)tile_size); ++i)
    {
      // each tile has a subset of columns to work with
      // tile_tid tells which exact column to use from the subset
      // it assumes that W is transposed already

      int _col  = i * tile_size + tx;
      int _col2 = i * tile_size + ty;
      
      if(row<B && _col<OC)
        smem[tx* tile_size +ty] = ht[((t-1)*B + row)*OC + _col];
      else
        smem[tx* tile_size +ty] = 0;
      
      if (col<fourX_OC && _col2<OC)
        smem[offset+ty* tile_size +tx] = W[col*OC + _col2];
      else
        smem[offset+ty* tile_size +tx] = 0;
      
      __syncthreads();


      for(int j=0; j<tile_size; ++j)
        y += smem[j* tile_size +ty] * smem[offset+j* tile_size +tx];
      
      __syncthreads();
      
    }
  }



  if(row<B && col<fourX_OC)
  {
    
    if (col<tanh_offset)
    {
      if (t==0) //TODO: maybe create a separate kernel to solve the ifs?
        y = 1/(1+exp(-(     x_out[(row*T + t)*fourX_OC + col] +b[col]) ));
      else
        y = 1/(1+exp(-( y + x_out[(row*T + t)*fourX_OC + col] +b[col]) ));
    }
    else
    {
      if (t==0)
        y = tanhf(     x_out[(row*T + t)*fourX_OC + col] +b[col]);
      else
        y = tanhf( y + x_out[(row*T + t)*fourX_OC + col] +b[col]);
    }

    // Now we have tensors i, f, o and c_
    // Output dim is: [T, B, 4*OC]
    // Continuing on this kernel will result on partial usage of this kernel threads, we therefore move the result to the global memory and call another kernel

    fused_out[(t*B + row)*fourX_OC + col] = y;
  }
}






__global__ void lstm_elementwise_ops_kernel(const float *fused_out,
                      float *ht, float *ct,
                      const int tile_size, const int tile_offset,
                      const int t, const int T,
                      const int B, const int OC, const int fourX_OC,
                      const int f_offset, const int o_offset, const int c_offset) {
  // ht        e [T, B,    OC]
  // ct        e [T, B,    OC]
  // fused out e [T, B,  4*OC]

  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int x_block = blockIdx.x;
  int y_block = blockIdx.y;
  

  int row = y_block*tile_size + ty;
  int col = x_block*tile_size + tx;


  if(row<B && col<OC)
  {
    float _ct;
    float idx = (t*B + row);
    int tb_offset = idx*fourX_OC; //TODO: factorize index
    int ht_tb_offset = idx*OC;

    // ct = f*ct + i*c_
    if(t==0)
      _ct = fused_out[tb_offset + col]*fused_out[tb_offset + c_offset + col];
    else
      _ct = fused_out[tb_offset + f_offset + col]*ct[((t-1)*B + row)*OC + col] + fused_out[tb_offset + col]*fused_out[tb_offset + c_offset + col];

    // ht = o*tanh(ct)
    ht[ht_tb_offset + col] = fused_out[tb_offset + o_offset + col]*tanhf(_ct);
    ct[ht_tb_offset + col] = _ct;
  }
}


__global__ void lstm_single_step_backward_dht_kernel(const float *d_ifoc,
                      float *d_ht, const float *w,
                      const int t, const int _t, const int T,
                      const int tile_size, const int tile_offset,
                      const int B, const int C, const int OC) {

  int row = blockIdx.y * blockDim.y + threadIdx.y; // B
  int col = blockIdx.x * blockDim.x + threadIdx.x; // C
  int tx = threadIdx.x;
  int ty = threadIdx.y;

  // d_ht      e [B,      OC]
  // W         e [4*OC,   OC]
  // d_ifoc    e [T, B, 4*OC]



  extern __shared__ char _smem[];
  auto smem = reinterpret_cast<float*>(_smem);



  int offset = tile_offset;

  float tmp = 0.0f;
  // consider row as B and col as C
  
  __syncthreads();
  

#pragma unroll
  for (int i=0; i<ceilf(OC/(float)tile_size); ++i)
  {
    int _col = i*tile_size + tx;
    int _row = i*tile_size + ty;


    if( row<B  && _col<OC)
      smem[tx*tile_size +ty] = d_ifoc[_t*B*OC + row*OC + _col];
    else
      smem[tx*tile_size +ty] = 0;


    if(_row<OC &&  col<C)
      smem[offset+ty*tile_size +tx] = w[_row*C + col];
    else
      smem[offset+ty*tile_size +tx] = 0;
    

    __syncthreads();


#pragma unroll
    for(int j=0; j<tile_size; ++j)
      tmp += smem[j*tile_size +ty] * smem[offset+j*tile_size +tx];
    
    __syncthreads();
  }

  if(row<B && col<C)
    d_ht[row * C + col] = tmp;
}



__global__ void lstm_backward_dx_kernel(const float *d_ifoc,
                      float *dx, const float *w,
                      const int tile_size, const int tile_offset,
                      const int B, const int T, const int C, const int OC) {

  int row = blockIdx.y * blockDim.y + threadIdx.y; // BT
  int col = blockIdx.x * blockDim.x + threadIdx.x; // OC
  int tx = threadIdx.x;
  int ty = threadIdx.y;

  // d_ifoc e [T, B, 4*OC]
  // dx     e [B, T,    C]

  float sum = 0.0f;


  extern __shared__ char _smem[];
  auto smem = reinterpret_cast<float*>(_smem);



  int offset = tile_offset;

  float tmp = 0.0f;
  // consider row as BT and col as C
  
  __syncthreads();

  int b = row / T;
  int t = row % T;
  
#pragma unroll
  for (int i=0; i<ceilf(OC/(float)tile_size); ++i)
  {
    int _col = i*tile_size + tx;
    int _row = i*tile_size + ty;


    if( row<B*T  && _col<OC)
      smem[tx*tile_size +ty] = d_ifoc[(t*B + b)*OC + _col];
    else
      smem[tx*tile_size +ty] = 0;


    if(_row<OC &&  col<C)
      smem[offset+ty*tile_size +tx] = w[_row*C + col];
    else
      smem[offset+ty*tile_size +tx] = 0;
    

    __syncthreads();


#pragma unroll
    for(int j=0; j<tile_size; ++j)
      tmp += smem[j*tile_size +ty] * smem[offset+j*tile_size +tx];
    
    __syncthreads();
  }

  if(row<B*T && col<C)
    dx[(b*T + t)*C + col] = tmp;
}






__global__ void lstm_elementwise_ops_backward_kernel(const float *fused_out,
                      const float *ct,
                      float *d_ht, float *d_ct, float *d_ifoc, float *dB,
                      const float *w,
                      const int tile_size, const int tile_offset,
                      const int t, const int _t, const int T,
                      const int B, const int OC, const int fourX_OC,
                      const int f_offset, const int o_offset, const int c_offset) {
  // d_ht      e [B,      OC]
  // d_ct      e [B,      OC]
  // d_ifoc    e [T, B, 4*OC]
  // ct        e [T, B,   OC]
  // fused out e [T, B, 4*OC]

  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int x_block = blockIdx.x;
  int y_block = blockIdx.y;
  

  int row = y_block*tile_size + ty;
  int col = x_block*tile_size + tx;


  int tb_offset = (_t*B + row)*fourX_OC;

  if(row<B && col<OC)
  {
    float d_ct_aux, _ct, tanh_ct, _d_ht;

    _d_ht = d_ht[row*OC + col];
    _ct = ct[(_t*B + row)*OC + col];
    tanh_ct = tanhf(_ct);


    // ct = f*ct + i*c_
    // ht = o*tanh(ct)

    
    float i = fused_out[tb_offset + col];
    float f = fused_out[tb_offset + f_offset + col];
    float o = fused_out[tb_offset + o_offset + col];
    float c = fused_out[tb_offset + c_offset + col];

    float d_i, d_f, d_o, d_c;

    d_o = _d_ht * tanh_ct;

    d_ct_aux = o * _d_ht;
    d_ct_aux = (1 - tanh_ct*tanh_ct) * d_ct_aux;

    if(t!=0) // set to zero instead of accumulating on the first iter
      d_ct_aux += d_ct[row*OC + col];

    // ct = f*ct + i*c_

    d_i = c * d_ct_aux;
    d_c = i * d_ct_aux;

    d_f = _ct * d_ct_aux;
    d_ct[row*OC + col] = f * d_ct_aux;


    d_ifoc[tb_offset +            col] = (i*(1-i)) * d_i;
    d_ifoc[tb_offset + f_offset + col] = (f*(1-f)) * d_f;
    d_ifoc[tb_offset + o_offset + col] = (o*(1-o)) * d_o;
    d_ifoc[tb_offset + c_offset + col] = (1- c*c ) * d_c;


    //fused_out[_t*B*fourX_OC + row*fourX_OC + f_offset + col]
    //_ct = fused_out[_t*B*fourX_OC + row*fourX_OC + f_offset + col]*ct[_t*B*OC + row*OC + col] + fused_out[_t*B*fourX_OC + row*fourX_OC + col]*fused_out[_t*B*fourX_OC + row*fourX_OC + c_offset + col];

    /**/
    float *db = dB + col;
    atomicAdd(db,          d_ifoc[tb_offset +            col]);
    atomicAdd(db+f_offset, d_ifoc[tb_offset + f_offset + col]);
    atomicAdd(db+o_offset, d_ifoc[tb_offset + o_offset + col]);
    atomicAdd(db+c_offset, d_ifoc[tb_offset + c_offset + col]);
    
  }
  
  extern __shared__ char _smem[];
  auto smem = reinterpret_cast<float*>(_smem);

  int offset = tile_offset;

  float tmp = 0.0f;
  // consider row as B and col as C
  
  __syncthreads();
  
#pragma unroll
  for (int i=0; i<ceilf(fourX_OC/(float)tile_size); ++i)
  {
    int _col = i*tile_size + tx;
    int _row = i*tile_size + ty;


    if( row<B  && _col<fourX_OC)
      smem[tx*tile_size +ty] = d_ifoc[tb_offset + _col];
    else
      smem[tx*tile_size +ty] = 0;


    if(_row<fourX_OC &&  col<OC)
      smem[offset+ty*tile_size +tx] = w[_row*OC + col];
    else
      smem[offset+ty*tile_size +tx] = 0;
    

    __syncthreads();

#pragma unroll
    for(int j=0; j<tile_size; ++j)
      tmp += smem[j*tile_size +ty] * smem[offset+j*tile_size +tx];
    
    __syncthreads();
  }

  if(row<B && col<OC)
    d_ht[row*OC + col] = tmp;
}