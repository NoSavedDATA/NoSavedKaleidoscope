#pragma once

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <mma.h>

#include "../tensor/tensor_struct.h"
#include "utils.h"

using namespace nvcuda;



__global__ void mult_backwarddx(const float *w,
    float *dx, const float *dy,
    const int tile_size, const int tile_offset,
    const int B, const int C, const int OC);




__global__ void mult_backwarddw_acc(const float *x,
                      float *dw, const float *dy, const int tile_size, const int tile_offset,
                      int B, int C, int OC); 


__global__ void mult_backwarddw(const float *x,
                      float *dw, const float *dy, const int tile_size, const int tile_offset,
                      int B, int C, int OC); 



__global__ void mult_kernel(const float *x, const float *w,
                      float *out, const int tile_size, const int tile_offset, const int B, const int C, const int OC); 



template<int WMMA_T, int X_WARPS, int Y_WARPS>
__global__ void wmma_backwarddx_kernel(float *dx, const float *w,
                      const float *dy, const int B, const int C, const int OC)
                      {

  int laneId = ( threadIdx.y * blockDim.x + threadIdx.x) % warpSize;
  int mw = laneId / WMMA_T;
  int ml = laneId % WMMA_T;

  int warp_y = threadIdx.y;
  int warp_x = (threadIdx.x / 32);


  const uint32_t warpX{(blockIdx.x * blockDim.x + threadIdx.x) / warpSize};  // C
  const uint32_t warpY{blockIdx.y * blockDim.y + threadIdx.y};               // B

  // warpX = (oc*X_WARPS + warp_x)




  wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> x_frag;
  wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> w_frag;
  wmma::fragment<wmma::accumulator, 16, 16, 16, float> y_frag;



  wmma::fill_fragment(y_frag, 0.0f);


  extern __shared__ float smem[];
  float *out_smem = smem;
  __half *hsmem = reinterpret_cast<__half*>(smem + Y_WARPS*WMMA_T*(X_WARPS*WMMA_T));

  __half *x_smem     = hsmem;
  __half *w_smem     = hsmem + Y_WARPS*WMMA_T*(WMMA_T);
  
  
  

#pragma unroll
  for (int tile=0; tile<OC; tile+=WMMA_T)
  {

    /*
#pragma unroll
    for (int i=0; i<2; ++i)
    {
      // warp * mw_size * i_size + mw*i_size + i
      int row_aux1 = warp_x*((int)(warpSize/WMMA_T))*2 + mw*2+i;
      int row_aux2 = warp_y*((int)(warpSize/WMMA_T))*2 + mw*2+i;
      
      if (row_aux1<WMMA_T)
      {
        if ((warpY*WMMA_T+row_aux1)<B && (tile+ml)<OC)
          x_smem[(warp_y*WMMA_T+row_aux1)*WMMA_T + ml] = __float2half(*(dy + (warpY*WMMA_T+row_aux1)*OC + tile+ml));
        else
          x_smem[(warp_y*WMMA_T+row_aux1)*WMMA_T + ml] = 0;
      }

      if (row_aux2<WMMA_T)
      {
        if ((tile+ml)<OC && (warpX*WMMA_T+row_aux2)<C)
          w_smem[(warp_x*WMMA_T+row_aux2)*WMMA_T + ml] = __float2half(*(w + (tile+ml)*C + warpX*WMMA_T+row_aux2));
        else
          w_smem[(warp_x*WMMA_T+row_aux2)*WMMA_T + ml] = 0;
      }
    }
    */


    wmma::fill_fragment(x_frag, 0.0f);
    const auto func_x = [&](const unsigned* frag_index_list,
        const unsigned fragment_index_count,
        const unsigned i,
        const unsigned j) {

      if((warpY*WMMA_T+i)<B && (tile+j)<OC)
      {
        __half tmp = __float2half(*(dy + (warpY*WMMA_T+i)*OC + tile + j));
#pragma unroll
        for (unsigned f = 0; f < fragment_index_count; f++)
            x_frag.x[frag_index_list[f]] = tmp;
      } // else did not work, so fill_fragment is a workaround
    };
    
    __syncwarp();
    wmma_foreach_ij(
        x_frag,
        func_x
      );



    wmma::fill_fragment(w_frag, 0.0f);
    const auto func_w = [&](const unsigned* frag_index_list,
          const unsigned fragment_index_count,
          const unsigned i,
          const unsigned j) {
        
        if(((tile+i)<OC && warpX*WMMA_T+j)<C)
        { 
          __half tmp = __float2half(*(w + (tile+i)*C + warpX*WMMA_T+j));
  #pragma unroll
          for (unsigned f = 0; f < fragment_index_count; f++)
            w_frag.x[frag_index_list[f]] = tmp;
        }
      };

    __syncwarp();
    wmma_foreach_ij(
        w_frag,
        func_w
      );




    __syncwarp();
    if ((warpY*WMMA_T)<B && (warpX*WMMA_T)<C)
    {
      //wmma::load_matrix_sync(x_frag, x_smem+(warp_y*WMMA_T)*WMMA_T, WMMA_T);
      //wmma::load_matrix_sync(w_frag, w_smem+(warp_x*WMMA_T)*WMMA_T, WMMA_T);
      wmma::mma_sync(y_frag, x_frag, w_frag, y_frag);
    }
    
  }


  if ((warpY*WMMA_T)<B && (warpX*WMMA_T)<C && (warp_y*WMMA_T)<B && (warp_x*WMMA_T)<C)
  {

    float *_out = out_smem + warp_y*WMMA_T*(X_WARPS*WMMA_T) + warp_x*WMMA_T;
    wmma::store_matrix_sync(_out, y_frag, X_WARPS*WMMA_T, wmma::mem_row_major);

    __syncthreads();
    
#pragma unroll
    for (int tile=0; tile<std::ceil((WMMA_T*WMMA_T)/(float)warpSize); ++tile)
    {
      int tile_idx = tile*warpSize + laneId;

      int row = tile_idx / WMMA_T;
      int col = tile_idx % WMMA_T;

      if((warpY*WMMA_T+row)<B  &&  (warpX*WMMA_T+col)<C && row<WMMA_T)
        dx[(warpY*WMMA_T+row)*C + warpX*WMMA_T+col] = _out[row*(X_WARPS*WMMA_T)+col];

    }
  }
}


template<int WMMA_T, int X_WARPS, int Y_WARPS>
__global__ void wmma_backwarddw_kernel(float *dw, const float *x,
                      const float *dy, const int B, const int C, const int OC)
{

  int laneId = ( threadIdx.y * blockDim.x + threadIdx.x) % warpSize;
  int mw = laneId / WMMA_T;
  int ml = laneId % WMMA_T;

  int warp_y = threadIdx.y;
  int warp_x = (threadIdx.x / 32);


  const uint32_t warpX{(blockIdx.x * blockDim.x + threadIdx.x) / warpSize};  // C
  const uint32_t warpY{blockIdx.y * blockDim.y + threadIdx.y};               // OC

  // warpX = (oc*X_WARPS + warp_x)




  wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> x_frag;
  wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> w_frag;
  wmma::fragment<wmma::accumulator, 16, 16, 16, float> y_frag;



  wmma::fill_fragment(y_frag, 0.0f);


  extern __shared__ float smem[];
  float *out_smem = smem;
  __half *hsmem = reinterpret_cast<__half*>(smem + Y_WARPS*WMMA_T*(X_WARPS*WMMA_T));

  __half *x_smem     = hsmem;
  __half *w_smem     = hsmem + Y_WARPS*WMMA_T*(WMMA_T);
  
  
  

#pragma unroll
  for (int tile=0; tile<B; tile+=WMMA_T)
  {
    /*
#pragma unroll
    for (int i=0; i<2; ++i)
    {
      // warp * mw_size * i_size + mw*i_size + i
      int row_aux1 = warp_x*((int)(warpSize/WMMA_T))*2 + mw*2+i;
      int row_aux2 = warp_y*((int)(warpSize/WMMA_T))*2 + mw*2+i;
      
      if (row_aux1<WMMA_T)
      {
        if ((tile+ml)<B && (warpY*WMMA_T+row_aux1)<OC)
          x_smem[(warp_y*WMMA_T+row_aux1)*WMMA_T + ml] = __float2half(*(dy + (tile+ml)*OC + warpY*WMMA_T+row_aux1));
        else
          x_smem[(warp_y*WMMA_T+row_aux1)*WMMA_T + ml] = 0;
      }

      if (row_aux2<WMMA_T)
      {
        if ((tile+ml)<B && (warpX*WMMA_T+row_aux2)<C)
          w_smem[(warp_x*WMMA_T+row_aux2)*WMMA_T + ml] = __float2half(*(x + (tile+ml)*C + warpX*WMMA_T+row_aux2));
        else
          w_smem[(warp_x*WMMA_T+row_aux2)*WMMA_T + ml] = 0;
      }
    }
    */
    wmma::fill_fragment(x_frag, 0.0f);
    const auto func_x = [&](const unsigned* frag_index_list,
        const unsigned fragment_index_count,
        const unsigned i,
        const unsigned j) {

      if((tile+j)<B && (warpY*WMMA_T+i)<OC)
      {
        __half tmp = __float2half(*(dy + (tile+j)*OC + warpY*WMMA_T+i));
#pragma unroll
        for (unsigned f = 0; f < fragment_index_count; f++)
            x_frag.x[frag_index_list[f]] = tmp;
      } // else did not work, so fill_fragment is a workaround
    };
    
    __syncwarp();
    wmma_foreach_ij(
        x_frag,
        func_x
      );



    wmma::fill_fragment(w_frag, 0.0f);
    const auto func_w = [&](const unsigned* frag_index_list,
          const unsigned fragment_index_count,
          const unsigned i,
          const unsigned j) {
        
        if((tile+i)<B && (warpX*WMMA_T+j)<C)
        { 
          __half tmp = __float2half(*(x + (tile+i)*C + warpX*WMMA_T+j));
  #pragma unroll
          for (unsigned f = 0; f < fragment_index_count; f++)
            w_frag.x[frag_index_list[f]] = tmp;
        }
      };

    __syncwarp();
    wmma_foreach_ij(
        w_frag,
        func_w
      );



    __syncwarp();
    if ((warpY*WMMA_T)<OC && (warpX*WMMA_T)<C)
    {
      wmma::mma_sync(y_frag, x_frag, w_frag, y_frag);
    }
    
    
  }


  if ((warpY*WMMA_T)<OC && (warpX*WMMA_T)<C && (warp_y*WMMA_T)<OC && (warp_x*WMMA_T)<C)
  {
    float *_out = out_smem + warp_y*WMMA_T*(X_WARPS*WMMA_T) + warp_x*WMMA_T;
    wmma::store_matrix_sync(_out, y_frag, X_WARPS*WMMA_T, wmma::mem_row_major);

    __syncthreads();
    
#pragma unroll
    for (int tile=0; tile<std::ceil((WMMA_T*WMMA_T)/(float)warpSize); ++tile)
    {
      int tile_idx = tile*warpSize + laneId;

      int row = tile_idx / WMMA_T;
      int col = tile_idx % WMMA_T;

      if((warpY*WMMA_T+row)<OC  &&  (warpX*WMMA_T+col)<C && row<WMMA_T)
        dw[(warpY*WMMA_T+row)*C + warpX*WMMA_T+col] = _out[row*(X_WARPS*WMMA_T)+col];

    }
  }
}





template<int WMMA_T, int X_WARPS, int Y_WARPS>
__global__ void wmma_mult_kernel(const float *x, const float *w,
                      float *out, const int B, const int C, const int OC)
{

  int laneId = ( threadIdx.y * blockDim.x + threadIdx.x) % warpSize;
  int mw = laneId / WMMA_T;
  int ml = laneId % WMMA_T;

  int warp_y = threadIdx.y;
  int warp_x = (threadIdx.x / 32);


  const uint32_t warpX{(blockIdx.x * blockDim.x + threadIdx.x) / warpSize};  // OC
  const uint32_t warpY{blockIdx.y * blockDim.y + threadIdx.y};               // B

  // warpX = (oc*X_WARPS + warp_x)




  wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> x_frag;
  wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> w_frag;
  wmma::fragment<wmma::accumulator, 16, 16, 16, float> y_frag;

  using FRAG_T = wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major>;



  wmma::fill_fragment(y_frag, 0.0f);


  extern __shared__ float smem[];
  float *out_smem = smem;
  __half *hsmem = reinterpret_cast<__half*>(smem + Y_WARPS*WMMA_T*(X_WARPS*WMMA_T));

  __half *x_smem     = hsmem;
  __half *w_smem     = hsmem + Y_WARPS*WMMA_T*(WMMA_T);
  
  
  

#pragma unroll
  for (int tile=0; tile<C; tile+=WMMA_T)
  {

#pragma unroll
    for (int i=0; i<2; ++i)
    {
      // warp * mw_size * i_size + mw*i_size + i
      int row_aux1 = warp_x*((int)(warpSize/WMMA_T))*2 + mw*2+i;
      int row_aux2 = warp_y*((int)(warpSize/WMMA_T))*2 + mw*2+i;
      
      if (row_aux1<WMMA_T)
      {
        if ((warpY*WMMA_T+row_aux1)<B && (tile+ml)<C)
          x_smem[(warp_y*WMMA_T+row_aux1)*WMMA_T + ml] = __float2half(*(x + (warpY*WMMA_T+row_aux1)*C + tile+ml));
        else
          x_smem[(warp_y*WMMA_T+row_aux1)*WMMA_T + ml] = 0;
      }

      if (row_aux2<WMMA_T)
      {
        if ((warpX*WMMA_T+row_aux2)<OC && (tile+ml)<C)
          w_smem[(warp_x*WMMA_T+row_aux2)*WMMA_T + ml] = __float2half(*(w + (warpX*WMMA_T+row_aux2)*C + tile+ml));
        else
          w_smem[(warp_x*WMMA_T+row_aux2)*WMMA_T + ml] = 0;
      }
    }
    

    __syncthreads();


    if ((warpY*WMMA_T)<B && (warpX*WMMA_T)<OC)
    {

      //wmma::fill_fragment(x_frag, 0.0f);

      
      

      /*
      asm volatile(".reg .b32 x<8>;"
                  "wmma.load.a.sync.aligned.m16n16k16.shared.row.f16"
                  " {x0, x1, x2, x3, x4, x5, x6, x7}, [%0], 16;"
                  :    // output registers
                  : "r"(x_smem+warp_y*WMMA_T*WMMA_T));
      */



      /*
      asm volatile(".reg .b32 x<8>;"
                   ".reg .b32 w<8>;"
                  "wmma.load.a.sync.aligned.m16n16k16.shared.row.f16"
                  " {x0, x1, x2, x3, x4, x5, x6, x7}, [%0], 16;"
                  "wmma.load.b.sync.aligned.m16n16k16.shared.col.f16"
                  " {w0, w1, w2, w3, w4, w5, w6, w7}, [%1], 16;"
                  :    // output registers
                  : "r"(x_smem+warp_y*WMMA_T*WMMA_T), "r"(w_smem+warp_x*WMMA_T*WMMA_T));
      */




      wmma::load_matrix_sync(x_frag, x_smem+warp_y*WMMA_T*WMMA_T, WMMA_T);



      /*
      asm volatile(".reg .b32 x<8>;"
                  "wmma.load.a.sync.aligned.m16n16k16.shared.row.f16"
                  " {%0,%1,%2,%3,%4,%5,%6,%7}, [%8], 16;"
                  : "=r"(x_frag.x[0]), "=r"(x_frag.x[1]), "=r"(x_frag.x[2]), "=r"(x_frag.x[3]), "=r"(x_frag.x[4]), "=r"(x_frag.x[5]), "=r"(x_frag.x[6]), "=r"(x_frag.x[7])
                  : "r"(x_smem+warp_y*WMMA_T*WMMA_T));
      */
      


      wmma::load_matrix_sync(w_frag, w_smem+warp_x*WMMA_T*WMMA_T, WMMA_T);



      wmma::mma_sync(y_frag, x_frag, w_frag, y_frag);



      //y_frag.x[0] = x_frag.num_elements;
    }
    
    // __syncthreads();
  }


  if ((warpY*WMMA_T)<B && (warpX*WMMA_T)<OC && (warp_y*WMMA_T)<B && (warp_x*WMMA_T)<OC)
  { 

    float *_out = out_smem + warp_y*WMMA_T*(X_WARPS*WMMA_T) + warp_x*WMMA_T;
    wmma::store_matrix_sync(_out, y_frag, X_WARPS*WMMA_T, wmma::mem_row_major);

    // __syncthreads();
    
#pragma unroll
    for (int tile=0; tile<std::ceil((WMMA_T*WMMA_T)/(float)warpSize); ++tile)
    {
      int tile_idx = tile*warpSize + laneId;

      int row = tile_idx / WMMA_T;
      int col = tile_idx % WMMA_T;

      // if ((blockIdx.y+ warp_y+laneId)==0)
      //     printf("warpX: %d\t warpX offset: %d\t OC: %d\n", warpX, warpX*WMMA_T, OC);

      if((warpY*WMMA_T+row)<B  &&  (warpX*WMMA_T+col)<OC && row<WMMA_T)
        out[(warpY*WMMA_T+row)*OC + warpX*WMMA_T+col] = _out[row*(X_WARPS*WMMA_T)+col];

    }
  }
}
                    







template<int WMMA_T, int X_WARPS, int Y_WARPS>
__global__ void wmma_cp_async(const float *x, const float *w,
                      float *out, const int B, const int C, const int OC)
{

  int laneId = ( threadIdx.y * blockDim.x + threadIdx.x) % warpSize;
  int mw = laneId / 4;
  int ml = laneId % 4;

  int warp_y = threadIdx.y;
  int warp_x = (threadIdx.x / 32);


  const uint32_t warpX{(blockIdx.x * blockDim.x + threadIdx.x) / warpSize};  // OC
  const uint32_t warpY{blockIdx.y * blockDim.y + threadIdx.y};               // B

  // warpX = (oc*X_WARPS + warp_x)




  wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> x_frag;
  wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> w_frag;
  wmma::fragment<wmma::accumulator, 16, 16, 16, float> y_frag;

  



  wmma::fill_fragment(y_frag, 0.0f);


  extern __shared__ float smem[];
  float *out_smem = smem;
  // __half *hsmem = reinterpret_cast<__half*>(smem + Y_WARPS*WMMA_T*(X_WARPS*WMMA_T));

  // __half *x_smem     = hsmem;
  // __half *w_smem     = hsmem + Y_WARPS*WMMA_T*(WMMA_T);

  float *hsmem = smem;// + Y_WARPS*WMMA_T*(X_WARPS*WMMA_T);

  float *x_smem     = hsmem;
  float *w_smem     = hsmem + Y_WARPS*WMMA_T*(WMMA_T*2);
  
  
  int k_count=0;
  int k_stride;

  int xor_addr = smem_xor_cp_async(laneId);

#pragma unroll
  for (int tile=0; tile<C; tile+=WMMA_T)
  {
    // warp * mw_size * i_size + mw*i_size + i
    
    k_stride = k_count % 2;
    k_count++;




    
    // each block deals with a 64x16 tile
    

    int row_aux1 = warp_x*4 + ml;
    int row_aux2 = warp_y*4 + ml;
    
    
    if (k_stride==0) { // loads 2 strides simultaneously


      // if ((warpX+warpY+laneId)==0)
      // {
      //   for (int i=0; i<32; ++i)
      //   {
      //     printf("%d\t", (int)smem_xor_cp_async(i)/4);
      //     if((i+1)%8==0)
      //       printf("\n");
      //   }
      //   printf("\n\n");
      // }

      if (row_aux1<WMMA_T)
      {
        float const *gmem_ptr = x + (warpY*WMMA_T+row_aux1)*C + tile+mw*4;
        
        // extra *2 to accomodate 32 instead of 16 C (i.e, the whole warpSize)
        //       *4 is necessary as it needs to accomodate 4 consecutive floats
        uint32_t smem_int_ptr = cast_smem_ptr_to_uint(&x_smem[(warp_y*WMMA_T+ warp_x*4)*WMMA_T*2 + xor_addr]);
        
        asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], %2, %3;"
                      :: "r"(smem_int_ptr),
                         "l"(gmem_ptr),
                         "n"(16),
                         "r"(((warpY*WMMA_T+row_aux1)<B) ? std::min((( C-(tile+mw*4)) /4)*4, 16) : 0)); // incorrect 0 padding yet
                         // "r"(((warpY*WMMA_T+row_aux1)<B) ? 16 : 0)); // incorrect 0 padding yet
        
      }

      if (row_aux2<WMMA_T)
      {
        // w_smem[(warp_x*WMMA_T+row_aux2)*WMMA_T + ml] = *(w + (warpX*WMMA_T+row_aux2)*C + tile+ml);
        
        float const *gmem_ptr = w + (warpX*WMMA_T+row_aux2)*C + tile+mw*4;
        
        // extra 2 to accomodate 32 instead of 16 C
        uint32_t smem_int_ptr = cast_smem_ptr_to_uint(&w_smem[(warp_x*WMMA_T+ warp_y*4)*WMMA_T*2 + xor_addr]);
        
        asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], %2, %3;"
                      :: "r"(smem_int_ptr),
                         "l"(gmem_ptr),
                         "n"(16),
                         "r"(((warpX*WMMA_T+row_aux2)<OC) ? std::min((( C-(tile+mw*4)) /4)*4, 16) : 0));
                         // "r"(((warpX*WMMA_T+row_aux2)<OC) ? 16 : 0));
        
      }
    }
    // asm volatile("cp.async.commit_group;\n" ::);
    
    asm volatile("cp.async.wait_all;");
    
    __syncthreads();

    // if ((warpX+warpY+laneId)==0)
    // {
    //   for (int i=0; i<Y_WARPS*WMMA_T*32; ++i)
    //   {
    //     printf("%.2f, ", x_smem[i]*10000);
    //     if((i+1)%32==0)
    //       printf("\n");
    //   }
    //   printf("\n\n");
    // }

    if ((warpY*WMMA_T)<B && (warpX*WMMA_T)<OC)
    {




      // if ((warpX+warpY+laneId)==0)
      // {
      //   for (int i=0; i<4*8; ++i)
      //   {
      //     int aux_i = i/8;
      //     int aux_j = i%8;
      //     printf("%d\t", (int)smem_dexor_from_cp_async(aux_i, aux_j)/4);
      //     if((i+1)%8==0)
      //       printf("\n");
      //   }
      //   printf("\n\n");
      // }


        
      
      const auto func_x = [&](const unsigned* frag_index_list,
          const unsigned fragment_index_count,
          const unsigned i,
          const unsigned j) {


          int wi = i/4;
          int xi = i%4;

          int xj = j/4;
          int wj = j%4;

          int offset = smem_dexor_from_cp_async(xi, xj*2 + k_stride)+wj;

        
          __half tmp = __float2half(*(x_smem + (warp_y*WMMA_T+ wi*4)*WMMA_T*2 + offset));


  #pragma unroll
          for (unsigned f = 0; f < fragment_index_count; f++)
              x_frag.x[frag_index_list[f]] = tmp;
      };
      
      __syncwarp();
      wmma_foreach_ij(
          x_frag,
          func_x
        );


      // if(warpY==0&&warpX==0&&laneId==0)
      //   printf("\n\n");


      
      const auto func_w = [&](const unsigned* frag_index_list,
            const unsigned fragment_index_count,
            const unsigned i,
            const unsigned j) {
          

            int wj = j/4;
            int xj = j%4;
          
            int xi = i/4;
            int wi = i%4;


            int offset = smem_dexor_from_cp_async(xj, xi*2+k_stride)+wi;

          
            __half tmp = __float2half(*(w_smem + (warp_x*WMMA_T+wj*4)*WMMA_T*2 + offset));
    #pragma unroll
            for (unsigned f = 0; f < fragment_index_count; f++)
              w_frag.x[frag_index_list[f]] = tmp;
        };

      __syncwarp();
      wmma_foreach_ij(
          w_frag,
          func_w
        );


      wmma::mma_sync(y_frag, x_frag, w_frag, y_frag);


    }
    
  }

  __syncthreads();

  if ((warpY*WMMA_T)<B && (warpX*WMMA_T)<OC && (warp_y*WMMA_T)<B && (warp_x*WMMA_T)<OC)
  { 

    float *_out = out_smem + warp_y*WMMA_T*(X_WARPS*WMMA_T) + warp_x*WMMA_T;
    
    wmma::store_matrix_sync(_out, y_frag, X_WARPS*WMMA_T, wmma::mem_row_major);
  //   const auto func_y = [&](const unsigned* frag_index_list,
  //         const unsigned fragment_index_count,
  //         const unsigned i,
  //         const unsigned j) {
                  
          
  // #pragma unroll
  //         for (unsigned f = 0; f < fragment_index_count; f++)
  //           _out[i*X_WARPS*WMMA_T + j] = y_frag.x[frag_index_list[f]];
  //     };

  //   __syncwarp();
  //   wmma_foreach_ij(
  //       y_frag,
  //       func_y
  //     );








    
#pragma unroll
    for (int tile=0; tile<std::ceil((WMMA_T*WMMA_T)/(float)(warpSize)); ++tile)
    {
      int tile_idx = tile*warpSize + laneId;

      int row =  tile_idx / WMMA_T;
      int col = (tile_idx % WMMA_T);


      if((warpY*WMMA_T+row)<B  &&  (warpX*WMMA_T+col)<OC && row<WMMA_T)
        out[(warpY*WMMA_T+row)*OC + warpX*WMMA_T+col] = _out[row*(X_WARPS*WMMA_T)+col];

    }
  }
}







template<int WMMA_T, int X_WARPS, int Y_WARPS>
__global__ void wmma_pingpong(const float *x, const float *w,
                      float *out, const int B, const int C, const int OC)
                      {

  int laneId = ( threadIdx.y * blockDim.x + threadIdx.x) % warpSize;
  int mw = laneId / WMMA_T;
  int ml = laneId % WMMA_T;

  int warp_y = threadIdx.y;
  int warp_x = (threadIdx.x / warpSize);

  int s=2;
  int circular_smem_counter=0;


  uint32_t warpX;                                                     // OC
  uint32_t warpY{blockIdx.y * blockDim.y + threadIdx.y};               // B

  // warpX = (oc*X_WARPS + warp_x)





  wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> x_frag;
  wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> w_frag;
  wmma::fragment<wmma::accumulator, 16, 16, 16, float> y_frag;

  //using FRAG_T = wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major>;

  wmma::fill_fragment(y_frag, 0.0f);


  extern __shared__ float smem[];
  float *out_smem = smem;
  __half *hsmem = reinterpret_cast<__half*>(smem + Y_WARPS*WMMA_T*(X_WARPS*WMMA_T));

  __half *x_smem_base  = hsmem;
  __half *w_smem_base  = hsmem + (Y_WARPS*WMMA_T)*WMMA_T;
  
  __half *x_smem, *w_smem;
  





  if (warp_x>=4)
  {
    warp_x-=4;
    warpX = (blockIdx.x*(blockDim.x/2))/warpSize + warp_x;
    
    
  
    for (int tile=0; tile<C; tile+=WMMA_T)
    {

      int tgt_smem = circular_smem_counter % s;


      x_smem = x_smem_base + tgt_smem*((Y_WARPS+X_WARPS)*WMMA_T*WMMA_T);
      w_smem = w_smem_base + tgt_smem*((Y_WARPS+X_WARPS)*WMMA_T*WMMA_T);

      

  
      for (int i=0; i<2; ++i)
      {
        // warp*mw_size*i_size + mw*i_size + i
        int row_aux1 = warp_x*((int)(warpSize/WMMA_T))*2 + mw*2+i;
        int row_aux2 = warp_y*((int)(warpSize/WMMA_T))*2 + mw*2+i;
        
        if (row_aux1<WMMA_T)
        {
          if ((warpY*WMMA_T+row_aux1)<B && (tile+ml)<C)
            x_smem[(warp_y*WMMA_T+row_aux1)*WMMA_T + ml] = __float2half(*(x + (warpY*WMMA_T+row_aux1)*C + tile+ml));
          else
            x_smem[(warp_y*WMMA_T+row_aux1)*WMMA_T + ml] = 0;
        }

        if (row_aux2<WMMA_T)
        {
          if ((warpX*WMMA_T+row_aux2)<OC && (tile+ml)<C)
            w_smem[(warp_x*WMMA_T+row_aux2)*WMMA_T + ml] = __float2half(*(w + (warpX*WMMA_T+row_aux2)*C + tile+ml));
          else
            w_smem[(warp_x*WMMA_T+row_aux2)*WMMA_T + ml] = 0;
        }
      }
      

      
      asm volatile("bar.sync 0, 1024;"); // producer waits consumer
      
      asm volatile("bar.arrive 1, 1024;"); // producer ends


      circular_smem_counter++;
      
      

      // if ((blockIdx.x+blockIdx.y+warp_x+warp_y+laneId)==0)
      //   printf("Producer finished, tile: %d/%d.\n", tile, C);




      
      //asm volatile("bar.sync 2, 512;");
      // __syncthreads();

    }

    // if ((blockIdx.x+blockIdx.y+warp_x+warp_y+laneId)==0)
    //   printf("Producer exits.\n");

    return; // return is a must, otherwise the if below bugs
  }
  else if (warp_x<4)
  {
    warpX = (blockIdx.x*(blockDim.x/2))/warpSize + warp_x;


    asm volatile("bar.arrive 0, 1024;");

  
    for (int tile=0; tile<C; tile+=WMMA_T)
    {


      int tgt_smem = circular_smem_counter % s;

      x_smem = x_smem_base + tgt_smem*((Y_WARPS+X_WARPS)*WMMA_T*WMMA_T);
      w_smem = w_smem_base + tgt_smem*((Y_WARPS+X_WARPS)*WMMA_T*WMMA_T);


      


      // if ((blockIdx.x+blockIdx.y+warp_x+warp_y+laneId)==0)
      //   printf("\t\t\t\t\tConsumer wait %d.\n", tile);

      asm volatile("bar.sync 1, 1024;"); // consumer waits producer

      // if ((blockIdx.x+blockIdx.y+warp_x+warp_y+laneId)==0)
      //   printf("\t\t\t\t\tConsumer go %d.\n", tile);




      if ((warpY*WMMA_T)<B && (warpX*WMMA_T)<OC)
      {


        wmma::load_matrix_sync(x_frag, x_smem+warp_y*WMMA_T*WMMA_T, WMMA_T);
        wmma::load_matrix_sync(w_frag, w_smem+warp_x*WMMA_T*WMMA_T, WMMA_T);



        wmma::mma_sync(y_frag, x_frag, w_frag, y_frag);


      }
      

      asm volatile("bar.arrive 0, 1024;"); // consumer ends

      // if ((blockIdx.x+blockIdx.y+warp_x+warp_y+laneId)==0)
      //   printf("\t\t\t\t\tConsumer finished, tile: %d.\n", tile);
      
      
      circular_smem_counter++;
    }


    // if ((blockIdx.x+blockIdx.y+warp_x+warp_y+laneId)==0)
    //   printf("\t\tConsumer exits.\n");


    if ((warpY*WMMA_T)<B && (warpX*WMMA_T)<OC && (warp_y*WMMA_T)<B && (warp_x*WMMA_T)<OC)
    { 
      // if ((blockIdx.x+blockIdx.y+warp_x+warp_y+laneId)==0)
      //   printf("\t\t ------ NOW STORE OUTPUT ------\n");

      float *_out = out_smem + warp_y*WMMA_T*(X_WARPS*WMMA_T) + warp_x*WMMA_T;
      wmma::store_matrix_sync(_out, y_frag, X_WARPS*WMMA_T, wmma::mem_row_major);

      // if ((blockIdx.x+blockIdx.y+warp_x+warp_y+laneId)==0)
      //   printf("\t\t ------ post wmma store ------\n");


      
      
  
      for (int tile=0; tile<std::ceil((WMMA_T*WMMA_T)/(float)warpSize); ++tile)
      {
        int tile_idx = tile*warpSize + laneId;

        int row = tile_idx / WMMA_T;
        int col = tile_idx % WMMA_T;
        


        // if ((blockIdx.y+ warp_y+laneId)==0)
        //   printf("warpX: %d\t warpX offset: %d\t OC: %d\n", warpX, warpX*WMMA_T, OC);
        // if ((blockIdx.x+ warp_x+laneId)==0)
        //   printf("warpY: %d, out: %f\n", warpY);

        if((warpY*WMMA_T+row)<B  &&  (warpX*WMMA_T+col)<OC  &&  row<WMMA_T)
          out[(warpY*WMMA_T+row)*OC + warpX*WMMA_T+col] = _out[row*(X_WARPS*WMMA_T)+col];

      }
    }
    // if ((blockIdx.x+blockIdx.y+warp_x+warp_y+laneId)==0)
    //     printf("\t\t ------ post tiled ------\n");
    
  }
} 



template<int WMMA_T, int X_WARPS, int Y_WARPS>
__global__ void wmma_mult_kernel_(const float *x, const float *w,
                      float *out, const int B, const int C, const int OC) {

  int laneId = ( threadIdx.y * blockDim.x + threadIdx.x) % warpSize;
  int mw = laneId / WMMA_T;
  int ml = laneId % WMMA_T;

  int warp_y = threadIdx.y;
  int warp_x = (threadIdx.x / 32);


  const uint32_t warpX{(blockIdx.x * blockDim.x + threadIdx.x) / warpSize};  // OC
  const uint32_t warpY{blockIdx.y * blockDim.y + threadIdx.y};               // B

  // warpX = (oc*X_WARPS + warp_x)




  wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> x_frag;
  wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> w_frag;
  wmma::fragment<wmma::accumulator, 16, 16, 16, float> y_frag;

  using FRAG_T = wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major>;



  wmma::fill_fragment(y_frag, 0.0f);


  extern __shared__ float smem[];
  float *out_smem = smem;
  

  for (int tile=0; tile<C; tile+=WMMA_T)
  {
    

    
    wmma::fill_fragment(x_frag, 0.0f);
    const auto func_x = [&](const unsigned* frag_index_list,
        const unsigned fragment_index_count,
        const unsigned i,
        const unsigned j) {

      if((warpY*WMMA_T+i)<B && (tile+j)<C)
      {
        __half tmp = __float2half(*(x + (warpY*WMMA_T+i)*C + tile + j));
#pragma unroll
        for (unsigned f = 0; f < fragment_index_count; f++)
            x_frag.x[frag_index_list[f]] = tmp;
      } // else did not work, so fill_fragment is a workaround
    };
    
    __syncwarp();
    wmma_foreach_ij(
        x_frag,
        func_x
      );



    wmma::fill_fragment(w_frag, 0.0f);
    const auto func_w = [&](const unsigned* frag_index_list,
          const unsigned fragment_index_count,
          const unsigned i,
          const unsigned j) {
        
        if((warpX*WMMA_T+j)<OC && (tile+i)<C)
        { 
          __half tmp = __float2half(*(w + (warpX*WMMA_T+j)*C + tile+i));
  #pragma unroll
          for (unsigned f = 0; f < fragment_index_count; f++)
            w_frag.x[frag_index_list[f]] = tmp;
        }
      };

    __syncwarp();
    wmma_foreach_ij(
        w_frag,
        func_w
      );




    __syncwarp();
    if ((warpY*WMMA_T)<B && (warpX*WMMA_T)<OC)
      wmma::mma_sync(y_frag, x_frag, w_frag, y_frag);

    
  }


  if ((warpY*WMMA_T)<B && (warpX*WMMA_T)<OC && (warp_y*WMMA_T)<B && (warp_x*WMMA_T)<OC)
  { 

    float *_out = out_smem + warp_y*WMMA_T*(X_WARPS*WMMA_T) + warp_x*WMMA_T;
    wmma::store_matrix_sync(_out, y_frag, X_WARPS*WMMA_T, wmma::mem_row_major);

    __syncthreads();
    
#pragma unroll
    for (int tile=0; tile<std::ceil((WMMA_T*WMMA_T)/(float)warpSize); ++tile)
    {
      int tile_idx = tile*warpSize + laneId;

      int row = tile_idx / WMMA_T;
      int col = tile_idx % WMMA_T;

      if((warpY*WMMA_T+row)<B  &&  (warpX*WMMA_T+col)<OC && row<WMMA_T)
        out[(warpY*WMMA_T+row)*OC + warpX*WMMA_T+col] = _out[row*(X_WARPS*WMMA_T)+col];

    }
  }
}








void matmul_backward(Tensor *, Tensor *,
                    float *dinp, float *dw,
                    float *dout);


void matmul_forward(float* out,
                     float* inp, float* W,
                     int B, int C, int OC, int thread_id);